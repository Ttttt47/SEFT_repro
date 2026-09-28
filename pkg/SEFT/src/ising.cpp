#include <Rcpp.h>
using namespace Rcpp;

static inline double log_sum_exp(const std::vector<double> &v) {
  double m = -std::numeric_limits<double>::infinity();
  for (double x : v) if (x > m) m = x;
  double s = 0.0;
  for (double x : v) s += std::exp(x - m);
  return m + std::log(s);
}

static inline void inv2x2(double a00, double a01, double a10, double a11,
                          double &b00, double &b01, double &b10, double &b11,
                          double ridge = 1e-8) {
  double det = a00 * a11 - a01 * a10;
  if (!std::isfinite(det) || std::fabs(det) < ridge) {
    a00 += ridge; a11 += ridge;
    det = a00 * a11 - a01 * a10;
  }
  double invdet = 1.0 / det;
  b00 =  a11 * invdet;
  b01 = -a01 * invdet;
  b10 = -a10 * invdet;
  b11 =  a00 * invdet;
}

static inline double prob_abs_le(double a, double mu, double sd) {
  double pa = R::pnorm5(a,  mu, sd, 1, 0);
  double pb = R::pnorm5(-a, mu, sd, 1, 0);
  double p = pa - pb;
  if (!std::isfinite(p) || p < 1e-300) p = 1e-300;
  if (p > 1.0) p = 1.0;
  return p;
}

static inline double log_add_exp(double a, double b) {
  double m = (a > b) ? a : b;
  return m + std::log(std::exp(a - m) + std::exp(b - m));
}

static inline double logpdf_f0_absmax2_std(double y) {
  // f0(y) = 2 * phi(y) * P(|Z| <= |y|), Z~N(0,1)
  double ay = std::fabs(y);
  double logphi = R::dnorm4(y, 0.0, 1.0, 1);
  double p = prob_abs_le(ay, 0.0, 1.0);
  return std::log(2.0) + logphi + std::log(p);
}

static inline double logpdf_f1comp_absmax01(double y, double mu, double sd) {
  // Y = argmax(|X|,|Z|) with tie->X
  // X~N(mu,sd^2), Z~N(0,1), independent
  // f(y) = fX(y)*P(|Z|<=|y|) + fZ(y)*P(|X|<=|y|)
  double ay = std::fabs(y);

  double logfX = R::dnorm4(y, mu, sd, 1);
  double logfZ = R::dnorm4(y, 0.0, 1.0, 1);

  double pZ = prob_abs_le(ay, 0.0, 1.0);
  double pX = prob_abs_le(ay, mu, sd);

  double logA = logfX + std::log(pZ);
  double logB = logfZ + std::log(pX);

  return log_add_exp(logA, logB);
}

// log f(y; mu, sd) and gradients w.r.t mu and log(sd)
// where f is the absmax emission between X~N(mu,sd^2) and Z~N(0,1).
static inline void logpdf_grad_absmax01(double y, double mu, double sd,
                                       double &logf,
                                       double &g_mu,
                                       double &g_logsd) {
  const double tiny = 1e-300;

  double ay = std::fabs(y);

  // A(y) = P(|Z| <= |y|), Z~N(0,1)
  double A = prob_abs_le(ay, 0.0, 1.0);

  // fZ(y)
  double fZ = std::exp(R::dnorm4(y, 0.0, 1.0, 1));

  // X~N(mu,sd^2)
  sd = std::max(sd, 1e-6);
  double logfX = R::dnorm4(y, mu, sd, 1);
  double fX = std::exp(logfX);

  // B(y;mu,sd)=P(|X|<=|y|)
  double B = prob_abs_le(ay, mu, sd);

  // f(y)=A*fX + fZ*B
  double f = A * fX + fZ * B;
  f = std::max(f, tiny);
  logf = std::log(f);

  // Derivatives
  // dfX/dmu and dfX/dsd
  double diff = (y - mu);
  double sd2 = sd * sd;
  double dfX_dmu = fX * (diff / sd2);
  double dfX_dsd = fX * (-1.0 / sd + (diff * diff) / (sd2 * sd));

  // dB/dmu and dB/dsd
  // B = Phi(t1) - Phi(t2), t1=(ay-mu)/sd, t2=(-ay-mu)/sd
  double t1 = (ay - mu) / sd;
  double t2 = (-ay - mu) / sd;
  double phi1 = std::exp(R::dnorm4(t1, 0.0, 1.0, 1));
  double phi2 = std::exp(R::dnorm4(t2, 0.0, 1.0, 1));

  double dB_dmu = -(phi1 - phi2) / sd;
  double dB_dsd = (-t1 * phi1 + t2 * phi2) / sd;

  double df_dmu = A * dfX_dmu + fZ * dB_dmu;
  double df_dsd = A * dfX_dsd + fZ * dB_dsd;

  g_mu = df_dmu / f;
  g_logsd = (df_dsd / f) * sd;  // chain rule: d/dlogsd = sd * d/dsd
}

static inline double ig_prior_log_sig2(double sig2, double a, double b) {
  // Inverse-Gamma(a, b) prior on sig2: p(sig2) is proportional to sig2^{-(a+1)} exp(-b/sig2)
  sig2 = std::max(sig2, 1e-12);
  return -(a + 1.0) * std::log(sig2) - b / sig2;
}

static inline double ig_prior_grad_logsd(double sig2, double a, double b) {
  // Gradient of log prior w.r.t log(sd), where sig2 = sd^2.
  // log p(sig2) = -(a+1) log sig2 - b/sig2 + const
  // d/dlogsd = 2 * (-(a+1) + b/sig2)
  sig2 = std::max(sig2, 1e-12);
  return 2.0 * (-(a + 1.0) + b / sig2);
}

static inline double Q_absmax01_component(const NumericVector &x,
                                         const NumericVector &w,
                                         double mu, double logsd,
                                         double a, double b) {
  double sd = std::exp(logsd);
  double val = 0.0;
  for (int i = 0; i < x.size(); ++i) {
    double wi = w[i];
    if (wi <= 0) continue;
    double logf, gmu, gls;
    logpdf_grad_absmax01(x[i], mu, sd, logf, gmu, gls);
    val += wi * logf;
  }
  double sig2 = sd * sd;
  val += ig_prior_log_sig2(sig2, a, b);
  return val;
}

static inline void update_mu_sig2_absmax01_component(const NumericVector &x,
                                                     const NumericVector &w,
                                                     double &mu,
                                                     double &sig2,
                                                     double a, double b,
                                                     int n_iter = 25,
                                                     double step0 = 1.0,
                                                     int max_backtrack = 12) {
  const double tiny = 1e-12;

  // If the component has almost no mass, skip update
  double wsum = 0.0;
  for (int i = 0; i < w.size(); ++i) wsum += w[i];
  if (wsum < 1e-12) return;

  double logsd = 0.5 * std::log(std::max(sig2, 1e-6));

  for (int it = 0; it < n_iter; ++it) {
    double sd = std::exp(logsd);

    // Compute gradient of weighted log-likelihood
    double g_mu = 0.0, g_logsd = 0.0;
    for (int i = 0; i < x.size(); ++i) {
      double wi = w[i];
      if (wi <= 0) continue;
      double logf, dmu, dls;
      logpdf_grad_absmax01(x[i], mu, sd, logf, dmu, dls);
      g_mu += wi * dmu;
      g_logsd += wi * dls;
    }

    // Add IG(a,b) prior gradient on sig2
    double sig2_cur = sd * sd;
    g_logsd += ig_prior_grad_logsd(sig2_cur, a, b);

    double q0 = Q_absmax01_component(x, w, mu, logsd, a, b);
    double step = step0;
    bool accepted = false;

    for (int bt = 0; bt < max_backtrack; ++bt) {
      double mu_new = mu + step * g_mu;
      double logsd_new = logsd + step * g_logsd;

      // Keep sd in a reasonable range
      logsd_new = std::min(std::max(logsd_new, std::log(1e-3)), std::log(1e2));

      double q1 = Q_absmax01_component(x, w, mu_new, logsd_new, a, b);

      if (std::isfinite(q1) && q1 >= q0 - 1e-8) {
        mu = mu_new;
        logsd = logsd_new;
        accepted = true;
        break;
      }
      step *= 0.5;
    }

    if (!accepted) break;
  }

  double sd_final = std::exp(logsd);
  sig2 = std::max(sd_final * sd_final, 1e-6);
}

static inline void update_mu_sig2_absmax01_component_bhhh(
  const NumericVector &x,
  const NumericVector &w,
  double &mu,
  double &sig2,
  double a, double b,
  int n_iter = 30,
  double ridge = 1e-6,
  double stpmax = 5.0,
  int max_backtrack = 15,
  double armijo = 1e-4) {

double wsum = 0.0;
for (int i = 0; i < w.size(); ++i) wsum += w[i];
if (wsum < 1e-12) return;

double logsd = 0.5 * std::log(std::max(sig2, 1e-6));

for (int it = 0; it < n_iter; ++it) {
  double sd = std::exp(logsd);

  double g0 = 0.0, g1 = 0.0;
  double I00 = 0.0, I01 = 0.0, I11 = 0.0;

  for (int i = 0; i < x.size(); ++i) {
    double wi = w[i];
    if (wi <= 0) continue;

    double logf, dmu, dls;
    logpdf_grad_absmax01(x[i], mu, sd, logf, dmu, dls);

    g0 += wi * dmu;
    g1 += wi * dls;

    I00 += wi * dmu * dmu;
    I01 += wi * dmu * dls;
    I11 += wi * dls * dls;
  }

  double sig2_cur = sd * sd;
  g1 += ig_prior_grad_logsd(sig2_cur, a, b);

  I00 += ridge;
  I11 += ridge;

  double inv00, inv01, inv10, inv11;
  inv2x2(I00, I01, I01, I11, inv00, inv01, inv10, inv11, ridge);

  double dmu = inv00 * g0 + inv01 * g1;
  double dls = inv10 * g0 + inv11 * g1;

  double dn = std::sqrt(dmu * dmu + dls * dls);
  if (!std::isfinite(dn) || dn < 1e-12) break;
  if (dn > stpmax) {
    double s = stpmax / dn;
    dmu *= s; dls *= s;
  }

  double q0 = Q_absmax01_component(x, w, mu, logsd, a, b);
  double slope = g0 * dmu + g1 * dls;
  if (!std::isfinite(q0) || !std::isfinite(slope) || slope <= 0.0) break;

  double step = 1.0;
  bool accepted = false;

  for (int bt = 0; bt < max_backtrack; ++bt) {
    double mu_new = mu + step * dmu;
    double logsd_new = logsd + step * dls;

    logsd_new = std::min(std::max(logsd_new, std::log(1e-3)), std::log(1e2));

    double q1 = Q_absmax01_component(x, w, mu_new, logsd_new, a, b);

    if (std::isfinite(q1) && q1 >= q0 + armijo * step * slope) {
      mu = mu_new;
      logsd = logsd_new;
      accepted = true;
      break;
    }

    step *= 0.5;
  }

  if (!accepted) break;
}

double sd_final = std::exp(logsd);
sig2 = std::max(sd_final * sd_final, 1e-6);
}


// Build local-neighbor list inside mask using 6-neighborhood.
// Returns:
// - mask_idx: 1-based global indices of voxels in mask (length P)
// - neigh: P x 6 integer matrix of local neighbor indices (0..P-1) or -1 if none
// - dim: integer vector c(Lx, Ly, Lz)
// [[Rcpp::export]]
List hmrf_build_neighbors_6n(LogicalVector mask) {
  IntegerVector d = mask.attr("dim");
  if (d.size() != 3) stop("mask must be a 3D logical array.");
  int Lx = d[0], Ly = d[1], Lz = d[2];
  int N = Lx * Ly * Lz;
  if (mask.size() != N) stop("mask length mismatch.");

  std::vector<int> mask_pos;
  mask_pos.reserve(N);
  for (int i = 0; i < N; ++i) if (mask[i]) mask_pos.push_back(i);

  int P = (int)mask_pos.size();
  if (P == 0) stop("mask is empty.");

  std::vector<int> g2l(N, -1);
  for (int p = 0; p < P; ++p) g2l[mask_pos[p]] = p;

  IntegerMatrix neigh(P, 6);
  for (int p = 0; p < P; ++p) {
    for (int j = 0; j < 6; ++j) neigh(p, j) = -1;

    int g = mask_pos[p];
    int x = g % Lx;
    int y = (g / Lx) % Ly;
    int z = g / (Lx * Ly);

    int gx;

    // x-1
    if (x > 0) {
      gx = g - 1;
      if (g2l[gx] >= 0) neigh(p, 0) = g2l[gx];
    }
    // x+1
    if (x < Lx - 1) {
      gx = g + 1;
      if (g2l[gx] >= 0) neigh(p, 1) = g2l[gx];
    }
    // y-1
    if (y > 0) {
      gx = g - Lx;
      if (g2l[gx] >= 0) neigh(p, 2) = g2l[gx];
    }
    // y+1
    if (y < Ly - 1) {
      gx = g + Lx;
      if (g2l[gx] >= 0) neigh(p, 3) = g2l[gx];
    }
    // z-1
    if (z > 0) {
      gx = g - Lx * Ly;
      if (g2l[gx] >= 0) neigh(p, 4) = g2l[gx];
    }
    // z+1
    if (z < Lz - 1) {
      gx = g + Lx * Ly;
      if (g2l[gx] >= 0) neigh(p, 5) = g2l[gx];
    }
  }

  IntegerVector mask_idx(P);
  for (int p = 0; p < P; ++p) mask_idx[p] = mask_pos[p] + 1; // 1-based for R

  return List::create(
    _["mask_idx"] = mask_idx,
    _["neigh"] = neigh,
    _["dim"] = d
  );
}

// Compute log f1(x) - log f0(x) for x in mask.
// Default:
//   f0 = N(0,1), f1 = normal mixture.
// If f0_absmax2 = true:
//   f0 is baseline null: absmax between two iid N(0,1).
// If f1_absmax01 = true:
//   f1 is baseline alt: absmax between Z~N(0,1) and X~N(mu_l,sig2_l) per mixture component.
// [[Rcpp::export]]
NumericVector hmrf_llr_mixnorm(NumericVector x_mask,
                              NumericVector p,
                              NumericVector mu,
                              NumericVector sig2,
                              bool f0_absmax2 = false,
                              bool f1_absmax01 = false) {
  int P = x_mask.size();
  int L = p.size();
  if (mu.size() != L || sig2.size() != L) stop("mixture parameter length mismatch.");

  NumericVector llr(P);

  for (int i = 0; i < P; ++i) {
    double x = x_mask[i];

    double logf0 = f0_absmax2 ? logpdf_f0_absmax2_std(x)
      : R::dnorm4(x, 0.0, 1.0, 1);

    std::vector<double> log_terms(L);
    for (int l = 0; l < L; ++l) {
      double sd = std::sqrt(std::max(sig2[l], 1e-12));
      double logfl = f1_absmax01 ? logpdf_f1comp_absmax01(x, mu[l], sd)
          : R::dnorm4(x, mu[l], sd, 1);
      log_terms[l] = std::log(std::max(p[l], 1e-300)) + logfl;
    }
    double logf1 = log_sum_exp(log_terms);

    llr[i] = logf1 - logf0;
  }

  return llr;
}


static inline int neigh_sum_local(const IntegerVector &theta,
                                  const IntegerMatrix &neigh,
                                  int p) {
  int s = 0;
  for (int j = 0; j < 6; ++j) {
    int nb = neigh(p, j);
    if (nb >= 0) s += theta[nb];
  }
  return s;
}

static inline void gibbs_sweep_local(IntegerVector &theta,
                                     const IntegerMatrix &neigh,
                                     double beta,
                                     double h,
                                     const NumericVector *llr_ptr) {
  int P = theta.size();
  for (int p = 0; p < P; ++p) {
    int ns = neigh_sum_local(theta, neigh, p);
    double logit = beta * (double)ns + h;
    if (llr_ptr) logit += (*llr_ptr)[p];
    double p1 = 1.0 / (1.0 + std::exp(-logit));
    theta[p] = (R::runif(0.0, 1.0) < p1) ? 1 : 0;
  }
}

static inline void compute_H01(const IntegerVector &theta,
                               const IntegerMatrix &neigh,
                               double &H0,
                               double &H1) {
  int P = theta.size();
  double h0 = 0.0;
  double h1 = 0.0;
  for (int p = 0; p < P; ++p) {
    int ns = neigh_sum_local(theta, neigh, p);
    h0 += 0.5 * (double)theta[p] * (double)ns;
    h1 += (double)theta[p];
  }
  H0 = h0;
  H1 = h1;
}

static inline void mixture_update_cpp(const NumericVector &x_mask,
                                      const NumericVector &gamma1,
                                      NumericVector &p,
                                      NumericVector &mu,
                                      NumericVector &sig2,
                                      double a, double b,
                                      bool f1_absmax01,
                                      int inner_iter = 30,
                                      int max_backtrack = 15) {
  int P = x_mask.size();
  int L = p.size();

  std::vector<double> wsum(L, 0.0);
  std::vector<double> xw(L, 0.0);
  std::vector<double> x2w(L, 0.0);

  double gsum = 0.0;

  NumericMatrix W(P, L);

  for (int i = 0; i < P; ++i) {
    double x = x_mask[i];
    double g = gamma1[i];
    gsum += g;

    std::vector<double> logdens(L);
    for (int l = 0; l < L; ++l) {
      double sd = std::sqrt(std::max(sig2[l], 1e-12));
      double logfl = f1_absmax01 ? logpdf_f1comp_absmax01(x, mu[l], sd)
      : R::dnorm4(x, mu[l], sd, 1);
      logdens[l] = std::log(std::max(p[l], 1e-300)) + logfl;
    }
    double logmix = log_sum_exp(logdens);

    for (int l = 0; l < L; ++l) {
      double r_il = std::exp(logdens[l] - logmix);
      double w = g * r_il;

      W(i, l) = w;
      wsum[l] += w;
      xw[l] += w * x;
      x2w[l] += w * x * x;
    }
  }

  double tiny = 1e-12;
  double denom_p = std::max(gsum, tiny);

  for (int l = 0; l < L; ++l) p[l] = std::max(wsum[l], tiny) / denom_p;

  double ps = 0.0;
  for (int l = 0; l < L; ++l) ps += p[l];
  for (int l = 0; l < L; ++l) p[l] /= ps;

  if (!f1_absmax01) {
    for (int l = 0; l < L; ++l) {
      double denom = std::max(wsum[l], tiny);
      mu[l] = xw[l] / denom;

      double ex2 = x2w[l] / denom;
      double var_ml = std::max(ex2 - mu[l] * mu[l], 1e-8);

      sig2[l] = (2.0 * a + denom * var_ml) / (2.0 * b + denom);
      sig2[l] = std::max(sig2[l], 1e-6);
    }
    return;
  }

  for (int l = 0; l < L; ++l) {
    NumericVector wl(P);
    for (int i = 0; i < P; ++i) wl[i] = W(i, l);

    update_mu_sig2_absmax01_component_bhhh(
    x_mask, wl,
    mu[l], sig2[l],
    a, b,
    inner_iter, 1e-6, 5.0,
    max_backtrack, 1e-4
    );
  }
}


static inline List prior_stats_cpp(const IntegerMatrix &neigh,
                                  double beta, double h,
                                  IntegerVector &theta,
                                  int burnin, int sweeps) {
  int P = theta.size();

  for (int i = 0; i < burnin; ++i) {
    gibbs_sweep_local(theta, neigh, beta, h, nullptr);
  }

  std::vector<double> H0s(sweeps), H1s(sweeps);
  double m0 = 0.0, m1 = 0.0;

  for (int s = 0; s < sweeps; ++s) {
    gibbs_sweep_local(theta, neigh, beta, h, nullptr);
    double H0, H1;
    compute_H01(theta, neigh, H0, H1);
    H0s[s] = H0; H1s[s] = H1;
    m0 += H0; m1 += H1;
  }
  m0 /= (double)sweeps;
  m1 /= (double)sweeps;

  // Covariance of (H0,H1)
  double c00 = 0.0, c01 = 0.0, c11 = 0.0;
  for (int s = 0; s < sweeps; ++s) {
    double d0 = H0s[s] - m0;
    double d1 = H1s[s] - m1;
    c00 += d0 * d0;
    c01 += d0 * d1;
    c11 += d1 * d1;
  }
  double denom = std::max(1.0, (double)(sweeps - 1));
  c00 /= denom; c01 /= denom; c11 /= denom;

  // q2 = log sum exp(-beta*H0 - h*H1)
  std::vector<double> qv(sweeps);
  for (int s = 0; s < sweeps; ++s) qv[s] = -beta * H0s[s] - h * H1s[s];
  double q2 = log_sum_exp(qv);

  NumericVector H_mean = NumericVector::create(m0, m1);
  NumericMatrix Cov(2, 2);
  Cov(0,0) = c00; Cov(0,1) = c01;
  Cov(1,0) = c01; Cov(1,1) = c11;

  return List::create(
    _["H_mean"] = H_mean,
    _["Cov"] = Cov,
    _["q2"] = q2,
    _["theta_state"] = theta
  );
}

static inline List posterior_stats_cpp(const IntegerMatrix &neigh,
                                      double beta, double h,
                                      const NumericVector &llr,
                                      IntegerVector &theta,
                                      int burnin, int sweeps) {
  int P = theta.size();

  for (int i = 0; i < burnin; ++i) {
    gibbs_sweep_local(theta, neigh, beta, h, &llr);
  }

  NumericVector gamma1(P);
  double mH0 = 0.0, mH1 = 0.0;

  for (int s = 0; s < sweeps; ++s) {
    gibbs_sweep_local(theta, neigh, beta, h, &llr);

    for (int p = 0; p < P; ++p) gamma1[p] += (double)theta[p];

    double H0, H1;
    compute_H01(theta, neigh, H0, H1);
    mH0 += H0; mH1 += H1;
  }

  for (int p = 0; p < P; ++p) gamma1[p] /= (double)sweeps;
  mH0 /= (double)sweeps;
  mH1 /= (double)sweeps;

  return List::create(
    _["gamma1"] = gamma1,
    _["Hc_mean"] = NumericVector::create(mH0, mH1),
    _["theta_state"] = theta
  );
}

static inline List update_beta_h_backtracking_cpp(const IntegerMatrix &neigh,
                                                 const NumericVector &Hc_mean,
                                                 double beta, double h,
                                                 IntegerVector &theta_prior,
                                                 int sweep_b, int sweep_r,
                                                 double alpha,
                                                 double stpmax,
                                                 double ridge,
                                                 int max_backtrack) {
  List ps0 = prior_stats_cpp(neigh, beta, h, theta_prior, sweep_b, sweep_r);
  NumericVector Hm0 = ps0["H_mean"];
  NumericMatrix Cov0 = ps0["Cov"];
  double q2_old = as<double>(ps0["q2"]);
  theta_prior = as<IntegerVector>(ps0["theta_state"]);

  double U0 = Hc_mean[0] - Hm0[0];
  double U1 = Hc_mean[1] - Hm0[1];

  // inv(Cov) * U
  double b00, b01, b10, b11;
  inv2x2(Cov0(0,0), Cov0(0,1), Cov0(1,0), Cov0(1,1), b00, b01, b10, b11, ridge);

  double d_beta = b00 * U0 + b01 * U1;
  double d_h    = b10 * U0 + b11 * U1;

  double step = std::sqrt(d_beta * d_beta + d_h * d_h);
  if (step > stpmax) {
    double s = stpmax / step;
    d_beta *= s;
    d_h *= s;
  }

  double slope = U0 * d_beta + U1 * d_h;
  if (!std::isfinite(slope) || slope <= 0.0) {
    return List::create(
      _["beta"] = beta,
      _["h"] = h,
      _["q2"] = q2_old,
      _["H_mean"] = Hm0,
      _["U"] = NumericVector::create(U0, U1),
      _["delta"] = NumericVector::create(d_beta, d_h),
      _["accepted"] = false,
      _["theta_prior"] = theta_prior
    );
  }

  double lam = 1.0;
  bool ok = false;
  double beta_new = beta, h_new = h;
  NumericVector Hm_new = clone(Hm0);
  double q2_new = q2_old;

  for (int t = 0; t < max_backtrack; ++t) {
    beta_new = beta + lam * d_beta;
    h_new    = h    + lam * d_h;

    IntegerVector theta_try = clone(theta_prior);
    List ps1 = prior_stats_cpp(neigh, beta_new, h_new, theta_try, sweep_b, sweep_r);
    Hm_new = as<NumericVector>(ps1["H_mean"]);
    q2_new = as<double>(ps1["q2"]);
    theta_try = as<IntegerVector>(ps1["theta_state"]);

    double delta_q2 = (beta_new - beta) * Hc_mean[0] +
                      (h_new - h)       * Hc_mean[1] +
                      (q2_new - q2_old);

    if (delta_q2 >= alpha * lam * slope) {
      ok = true;
      theta_prior = theta_try;
      break;
    }
    lam *= 0.5;
  }

  double U0n = Hc_mean[0] - Hm_new[0];
  double U1n = Hc_mean[1] - Hm_new[1];

  return List::create(
    _["beta"] = ok ? beta_new : beta,
    _["h"] = ok ? h_new : h,
    _["q2"] = ok ? q2_new : q2_old,
    _["H_mean"] = ok ? Hm_new : Hm0,
    _["U"] = NumericVector::create(ok ? U0n : U0, ok ? U1n : U1),
    _["delta"] = NumericVector::create(d_beta, d_h),
    _["accepted"] = ok,
    _["theta_prior"] = theta_prior
  );
}

// Shu-style Monte Carlo GEM fit (unknown beta/h and mixture params), then return fitted params.
// nb must be the output from hmrf_build_neighbors_6n(mask).
// x3d must be a 3D numeric array with the same dim as mask.
// [[Rcpp::export]]
List hmrf_gem_fit(List nb,
                  NumericVector x3d,
                  int L,
                  int iter_max,
                  int sweep_b,
                  int sweep_r,
                  double a,
                  double b,
                  double alpha,
                  double stpmax,
                  int max_backtrack,
                  double tol,
                  double beta_init,
                  double h_init,
                  NumericVector p_init,
                  NumericVector mu_init,
                  NumericVector sig2_init,
                  double init_prob_theta,
                  int seed,
                  bool verbose = true,
                  bool f0_absmax2 = false,
                  bool f1_absmax01 = false,
                  bool fixing_beta = false
                ) {

  IntegerVector d = nb["dim"];
  if (d.size() != 3) stop("nb$dim must be length-3.");
  int N = d[0] * d[1] * d[2];

  if (x3d.size() != N) stop("x3d length mismatch with nb$dim.");
  IntegerVector mask_idx = nb["mask_idx"]; // 1-based global
  IntegerMatrix neigh = nb["neigh"];
  int P = mask_idx.size();
  if (neigh.nrow() != P || neigh.ncol() != 6) stop("nb$neigh dimension mismatch.");

  if (p_init.size() != L || mu_init.size() != L || sig2_init.size() != L) {
    stop("initial mixture parameter length mismatch with L.");
  }

  NumericVector x_mask(P);
  for (int p = 0; p < P; ++p) x_mask[p] = x3d[mask_idx[p] - 1];

  double beta = beta_init;
  double h = h_init;
  NumericVector p_mix = clone(p_init);
  NumericVector mu = clone(mu_init);
  NumericVector sig2 = clone(sig2_init);

  // Persistent chain states across GEM iterations
  IntegerVector theta_prior(P);
  IntegerVector theta_post(P);
  if (seed != 0) {
    Function set_seed("set.seed");
    set_seed(seed);
  }
  for (int i = 0; i < P; ++i) {
    theta_prior[i] = (R::runif(0.0, 1.0) < init_prob_theta) ? 1 : 0;
    theta_post[i]  = (R::runif(0.0, 1.0) < init_prob_theta) ? 1 : 0;
  }

  List trace(iter_max);

  for (int it = 0; it < iter_max; ++it) {
    // if use f0_absmax2, always evaluate llr with f1 as absmax with N(0,1)
    NumericVector llr = hmrf_llr_mixnorm(x_mask, p_mix, mu, sig2, f0_absmax2, f1_absmax01);

    List post = posterior_stats_cpp(neigh, beta, h, llr, theta_post, sweep_b, sweep_r);
    NumericVector gamma1 = post["gamma1"];
    NumericVector Hc_mean = post["Hc_mean"];
    theta_post = as<IntegerVector>(post["theta_state"]);

    // Mixture M-step
    // only f1_absmax01 affects the mixture update, if not,
    mixture_update_cpp(x_mask, gamma1, p_mix, mu, sig2, a, b, f1_absmax01, 30, max_backtrack);

    // beta/h M-step via backtracking
    List bt = update_beta_h_backtracking_cpp(
      neigh, Hc_mean, beta, h,
      theta_prior, sweep_b, sweep_r,
      alpha, stpmax, 1e-8, max_backtrack
    );
    double beta_new = beta;  // fixing beta
    if (!fixing_beta) {
      beta_new = as<double>(bt["beta"]);
    }
    double h_new = as<double>(bt["h"]);
    theta_prior = as<IntegerVector>(bt["theta_prior"]);

    // Convergence check
    double rel_beta = std::fabs(beta_new - beta) / (std::fabs(beta) + 1e-3);
    double rel_h    = std::fabs(h_new - h)       / (std::fabs(h) + 1e-3);

    double rel_p = 0.0, rel_mu = 0.0, rel_s2 = 0.0;
    for (int l = 0; l < L; ++l) {
      rel_p  = std::max(rel_p,  std::fabs(p_mix[l] - p_init[l])   / (std::fabs(p_init[l]) + 1e-3));
      rel_mu = std::max(rel_mu, std::fabs(mu[l]    - mu_init[l])  / (std::fabs(mu_init[l]) + 1e-3));
      rel_s2 = std::max(rel_s2, std::fabs(sig2[l]  - sig2_init[l])/ (std::fabs(sig2_init[l]) + 1e-3));
    }
    double rel = std::max(std::max(rel_beta, rel_h), std::max(rel_p, std::max(rel_mu, rel_s2)));

    trace[it] = List::create(
      _["iter"] = it + 1,
      _["beta"] = beta_new,
      _["h"] = h_new,
      _["p"] = clone(p_mix),
      _["mu"] = clone(mu),
      _["sig2"] = clone(sig2),
      _["Hc_mean"] = Hc_mean,
      _["accepted"] = bt["accepted"],
      _["rel"] = rel
    );

    beta = beta_new;
    h = h_new;

    // Update "previous" for rel_p computation in next loop
    p_init = clone(p_mix);
    mu_init = clone(mu);
    sig2_init = clone(sig2);

    if (verbose && (it % 5 == 0 || it == iter_max -1 || rel < tol)) {
      Rcpp::Rcout << "GEM iter " << (it + 1) << "/" << iter_max
                  << ": beta=" << beta
                  << ", h=" << h
                  << ", p_mix=" << p_mix
                  << ", mu=" << mu
                  << ", sig2=" << sig2
                  << ", rel=" << rel
                  << "\n";
    }

    if (rel < tol) {
      trace = trace[Range(0, it)];
      break;
    }
  }

  return List::create(
    _["beta"] = beta,
    _["h"] = h,
    _["p"] = p_mix,
    _["mu"] = mu,
    _["sig2"] = sig2,
    _["theta_post_state"] = theta_post,
    _["theta_prior_state"] = theta_prior,
    _["trace"] = trace
  );
}

// Numerically stable sigmoid.
static inline double inv_logit_stable(double x) {
  if (x >= 0.0) {
    double z = std::exp(-x);
    return 1.0 / (1.0 + z);
  } else {
    double z = std::exp(x);
    return z / (1.0 + z);
  }
}

// Fisher-Yates shuffle (uses R RNG).
static inline void fy_shuffle(IntegerVector &ord) {
  int n = ord.size();
  for (int i = n - 1; i >= 1; --i) {
    int j = (int)std::floor(R::runif(0.0, (double)(i + 1)));
    int tmp = ord[i];
    ord[i] = ord[j];
    ord[j] = tmp;
  }
}


// One sweep: optional random scan; optional RB accumulation of p1 (instead of theta).
static inline void gibbs_sweep_local_rb(IntegerVector &theta,
                                        const IntegerMatrix &neigh,
                                        double beta,
                                        double h,
                                        const NumericVector &llr,
                                        IntegerVector &ord,
                                        bool random_scan,
                                        NumericVector *acc_p1  // if non-null, accumulate p1
) {
  int P = theta.size();
  if (random_scan) fy_shuffle(ord);

  for (int t = 0; t < P; ++t) {
    int p = ord[t];
    int ns = neigh_sum_local(theta, neigh, p);
    double logit = beta * (double)ns + h + llr[p];
    double p1 = inv_logit_stable(logit);

    if (acc_p1) (*acc_p1)[p] += p1;              // RB: accumulate conditional probability
    theta[p] = (R::runif(0.0, 1.0) < p1) ? 1 : 0; // standard Gibbs sampling
  }
}


// [[Rcpp::export]]
List hmrf_estimate_gamma_base(List nb,
  NumericVector baseline3d,
  double beta,
  double h,
  NumericVector p,
  NumericVector mu,
  NumericVector sig2,
  int burnin,
  int sweeps,
  int thin,
  double init_prob_theta,
  int seed,
  int n_chains = 4,
  bool random_scan = true,
  int init_mode = 3,
  bool f0_absmax2 = false,
  bool f1_absmax01 = false,
  double clamp_eps = 1e-8) {

  RNGScope scope;

  IntegerVector d = nb["dim"];
  int N = d[0] * d[1] * d[2];
  if (baseline3d.size() != N) stop("baseline3d length mismatch with nb$dim.");

  IntegerVector mask_idx = nb["mask_idx"]; // 1-based global
  IntegerMatrix neigh = nb["neigh"];
  int Pm = mask_idx.size();
  if (neigh.nrow() != Pm || neigh.ncol() != 6) stop("nb$neigh dimension mismatch.");

  // Extract baseline on mask
  NumericVector base_mask(Pm);
  for (int i = 0; i < Pm; ++i) base_mask[i] = baseline3d[mask_idx[i] - 1];

  // Baseline LLR (emission term)
  NumericVector llr_base = hmrf_llr_mixnorm(base_mask, p, mu, sig2,
                f0_absmax2, f1_absmax01);

  // order for random scan
  IntegerVector ord(Pm);
  for (int i = 0; i < Pm; ++i) ord[i] = i;

  // multi-chain accumulate
  NumericVector gamma_acc(Pm); // sum over chains of mean(p1)
  NumericVector chain_means(n_chains);

  // set RNG seed once (inside, we will advance RNG anyway)
  if (seed != 0) {
    Function set_seed("set.seed");
    set_seed(seed);
  }

  auto init_theta = [&](IntegerVector &theta, int mode) {
    if (mode == 1) {
    for (int i = 0; i < Pm; ++i) theta[i] = 0;
    } else if (mode == 2) {
    for (int i = 0; i < Pm; ++i) theta[i] = 1;
    } else if (mode == 3) {
    // warm start: sign(llr + h)
    for (int i = 0; i < Pm; ++i) theta[i] = ((llr_base[i] + h) > 0.0) ? 1 : 0;
    } else {
    // random
    for (int i = 0; i < Pm; ++i) theta[i] = (R::runif(0.0, 1.0) < init_prob_theta) ? 1 : 0;
    }
  };

  if (thin < 1) thin = 1;
  if (sweeps < 1) stop("sweeps must be >= 1.");
  // total iterations for accumulation part
  int T = sweeps * thin;

  for (int c = 0; c < n_chains; ++c) {
    IntegerVector theta(Pm);
    init_theta(theta, init_mode == 3 ? 3 : (init_mode == 0 ? 0 : init_mode));

    // burnin
    for (int b = 0; b < burnin; ++b) {
      gibbs_sweep_local_rb(theta, neigh, beta, h, llr_base, ord, random_scan, nullptr);
    }

    // RB accumulation
    NumericVector acc_p1(Pm);
    int kept = 0;
    for (int t = 1; t <= T; ++t) {
      gibbs_sweep_local_rb(theta, neigh, beta, h, llr_base, ord, random_scan, nullptr);
      if (t % thin == 0) {
        gibbs_sweep_local_rb(theta, neigh, beta, h, llr_base, ord, random_scan, &acc_p1);
        kept++;
      }
    }

    // mean p1 = gamma for this chain
    for (int i = 0; i < Pm; ++i) {
    double g = acc_p1[i] / (double)(Pm * kept); //

    }

    for (int i = 0; i < Pm; ++i) {
      double g = acc_p1[i] / (double)kept;
      // clamp
      g = std::min(std::max(g, clamp_eps), 1.0 - clamp_eps);
      gamma_acc[i] += g;
    }

    // chain-level summary (for quick diagnostics)
    double m = 0.0;
    for (int i = 0; i < Pm; ++i) m += (gamma_acc[i] / (double)(c + 1));
    chain_means[c] = m / (double)Pm;
  }

  // average over chains
  NumericVector gamma_mask(Pm);
  for (int i = 0; i < Pm; ++i) {
    double g = gamma_acc[i] / (double)n_chains;
    g = std::min(std::max(g, clamp_eps), 1.0 - clamp_eps);
    gamma_mask[i] = g;
  }

  // expand to full
  NumericVector gamma_full(N, NA_REAL);
  for (int i = 0; i < Pm; ++i) gamma_full[mask_idx[i] - 1] = gamma_mask[i];
  gamma_full.attr("dim") = d;

  // quick diagnostics
  double gmin = 1.0, gmax = 0.0;
  for (int i = 0; i < Pm; ++i) { gmin = std::min(gmin, gamma_mask[i]); gmax = std::max(gmax, gamma_mask[i]); }

  return List::create(
  _["gamma_mask"] = gamma_mask,
  _["gamma_full"] = gamma_full,
  _["gamma_min"] = gmin,
  _["gamma_max"] = gmax,
  _["chain_means"] = chain_means
  );
}

// Abs-max PLIS site-replacement calibration used by SEFT-Ising.
//
// The HMRF is fitted to the abs-max baseline B, so its baseline emissions are
// g0/g1.  A site replaced by an observed or mirror candidate is instead
// evaluated under the ordinary f0/f1 emissions.
// [[Rcpp::export]]
NumericVector hmrf_plis_reweight_absmax(List nb,
  NumericVector candidate3d,
  NumericVector baseline3d,
  NumericVector gamma_base_in,
  NumericVector p,
  NumericVector mu,
  NumericVector sig2,
  double clamp_eps = 1e-10) {

  IntegerVector d = nb["dim"];
  int N = d[0] * d[1] * d[2];
  if (candidate3d.size() != N) stop("candidate3d length mismatch with nb$dim.");
  if (baseline3d.size() != N) stop("baseline3d length mismatch with nb$dim.");
  if (p.size() != mu.size() || p.size() != sig2.size() || p.size() < 1) {
    stop("p, mu, and sig2 must have equal positive length.");
  }

  IntegerVector mask_idx = nb["mask_idx"];
  int Pm = mask_idx.size();
  NumericVector gamma_mask(Pm);
  if (gamma_base_in.size() == Pm) {
    gamma_mask = clone(gamma_base_in);
  } else if (gamma_base_in.size() == N) {
    for (int i = 0; i < Pm; ++i) {
      gamma_mask[i] = gamma_base_in[mask_idx[i] - 1];
    }
  } else {
    stop("gamma_base_in must have length P(mask) or N(full).");
  }

  auto log_add_exp_local = [](double a, double b) {
    if (!R_finite(a)) return b;
    if (!R_finite(b)) return a;
    double m = (a > b) ? a : b;
    return m + std::log(std::exp(a - m) + std::exp(b - m));
  };

  double psum = 0.0;
  for (int l = 0; l < p.size(); ++l) psum += p[l];
  if (!(psum > 0.0) || !R_finite(psum)) stop("invalid mixture weights.");

  auto log_f1_candidate = [&](double x) {
    double answer = -INFINITY;
    for (int l = 0; l < p.size(); ++l) {
      double weight = p[l] / psum;
      if (!(weight > 0.0)) continue;
      double sd = std::sqrt(std::max(sig2[l], 1e-12));
      answer = log_add_exp_local(
        answer, std::log(weight) + R::dnorm4(x, mu[l], sd, 1)
      );
    }
    return answer;
  };

  auto log_g1_background = [&](double x) {
    double answer = -INFINITY;
    for (int l = 0; l < p.size(); ++l) {
      double weight = p[l] / psum;
      if (!(weight > 0.0)) continue;
      double sd = std::sqrt(std::max(sig2[l], 1e-12));
      answer = log_add_exp_local(
        answer,
        std::log(weight) + logpdf_f1comp_absmax01(x, mu[l], sd)
      );
    }
    return answer;
  };

  NumericVector lis_full(N, NA_REAL);
  for (int i = 0; i < Pm; ++i) {
    int g = mask_idx[i] - 1;
    double gamma = gamma_mask[i];
    if (!R_finite(gamma)) stop("gamma_base_in contains NA/NaN/Inf.");
    gamma = std::min(std::max(gamma, clamp_eps), 1.0 - clamp_eps);

    double x = candidate3d[g];
    double baseline = baseline3d[g];
    double log_r1 = log_f1_candidate(x) - log_g1_background(baseline);
    double log_r0 = R::dnorm4(x, 0.0, 1.0, 1) -
      logpdf_f0_absmax2_std(baseline);
    double log_num1 = std::log(gamma) + log_r1;
    double log_num0 = std::log1p(-gamma) + log_r0;
    double log_den = log_add_exp_local(log_num1, log_num0);
    double gamma_new = std::exp(log_num1 - log_den);
    lis_full[g] = std::min(
      std::max(1.0 - gamma_new, clamp_eps), 1.0 - clamp_eps
    );
  }
  lis_full.attr("dim") = d;
  return lis_full;
}
