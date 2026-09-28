#include <Rcpp.h>
#include <algorithm>
#include <cmath>

using namespace Rcpp;

namespace {

inline double clamp_double(const double x, const double lo, const double hi) {
    return std::max(lo, std::min(hi, x));
}

inline double expit_stable(const double x) {
    if (x >= 0.0) {
        const double e = std::exp(-std::min(x, 700.0));
        return 1.0 / (1.0 + e);
    }
    const double e = std::exp(std::max(x, -700.0));
    return e / (1.0 + e);
}

inline double log_expit_negative_stable(const double x) {
    // log{1 - expit(x)} = log{expit(-x)} without forming either
    // probability.  This remains finite when the posterior null
    // probability is far below the smallest representable double.
    if (x >= 0.0) {
        return -x - std::log1p(std::exp(-std::min(x, 700.0)));
    }
    return -std::log1p(std::exp(std::max(x, -700.0)));
}

inline double trapezoid(const NumericVector& x, const NumericVector& y) {
    double out = 0.0;
    for (R_xlen_t j = 0; j + 1 < x.size(); ++j) {
        out += (x[j + 1] - x[j]) * (y[j] + y[j + 1]) * 0.5;
    }
    return out;
}

inline double interpolate_linear(
    const NumericVector& grid,
    const NumericVector& density,
    const double value
) {
    const R_xlen_t n = grid.size();
    if (value <= grid[0]) return density[0];
    if (value >= grid[n - 1]) return density[n - 1];
    const NumericVector::const_iterator upper =
        std::upper_bound(grid.begin(), grid.end(), value);
    const R_xlen_t right = upper - grid.begin();
    const R_xlen_t left = right - 1;
    const double weight = (value - grid[left]) / (grid[right] - grid[left]);
    return density[left] + weight * (density[right] - density[left]);
}

inline R_xlen_t index3(
    const int x,
    const int y,
    const int z,
    const int nx,
    const int ny
) {
    return x + static_cast<R_xlen_t>(nx) *
        (y + static_cast<R_xlen_t>(ny) * z);
}

} // namespace

// [[Rcpp::export]]
List wm_predictive_recursion_cpp(
    const NumericVector& z,
    const NumericVector& grid,
    const IntegerVector& order_idx,
    const double mu0 = 0.0,
    const double sigma0 = 1.0,
    const double decay = -0.67
) {
    const R_xlen_t grid_size = grid.size();
    NumericVector theta(grid_size, 1.0 / static_cast<double>(grid_size));
    NumericVector joint(grid_size);
    double pi0 = 1.0;

    for (R_xlen_t iteration = 0; iteration < order_idx.size(); ++iteration) {
        const int index = order_idx[iteration] - 1;
        if (index < 0 || index >= z.size()) {
            stop("order_idx contains an invalid one-based index.");
        }
        const double value = z[index];
        const double weight = std::pow(4.0 + iteration, decay);
        for (R_xlen_t j = 0; j < grid_size; ++j) {
            joint[j] = R::dnorm4(
                grid[j], value - mu0, sigma0, false
            ) * theta[j];
        }
        const double m0 = pi0 * R::dnorm4(value, mu0, sigma0, false);
        const double m1 = trapezoid(grid, joint);
        const double mixture = std::max(m0 + m1, 1e-300);
        pi0 = (1.0 - weight) * pi0 + weight * m0 / mixture;
        for (R_xlen_t j = 0; j < grid_size; ++j) {
            theta[j] = (1.0 - weight) * theta[j] +
                weight * joint[j] / mixture;
        }
    }

    NumericVector signal_density(grid_size);
    const double alternative_mass = std::max(1.0 - pi0, 1e-8);
    for (R_xlen_t i = 0; i < grid_size; ++i) {
        for (R_xlen_t j = 0; j < grid_size; ++j) {
            joint[j] = R::dnorm4(
                grid[j], grid[i] - mu0, sigma0, false
            ) * theta[j];
        }
        signal_density[i] = std::max(
            trapezoid(grid, joint) / alternative_mass, 1e-12
        );
    }
    const double normalizer = trapezoid(grid, signal_density);
    for (R_xlen_t i = 0; i < grid_size; ++i) {
        signal_density[i] /= normalizer;
    }
    return List::create(
        _["grid"] = clone(grid),
        _["density"] = signal_density,
        _["pi0"] = pi0
    );
}

// [[Rcpp::export]]
List wm_tv_logistic_mstep_cpp(
    NumericVector beta,
    const NumericVector& posterior_signal,
    const IntegerVector& dims,
    const double lambda,
    const LogicalVector& mask,
    const int iterations = 120,
    const double tolerance = 1e-5
) {
    if (dims.size() != 3) stop("dims must have length three.");
    const int nx = dims[0], ny = dims[1], nz = dims[2];
    const R_xlen_t n = static_cast<R_xlen_t>(nx) * ny * nz;
    if (beta.size() != n || posterior_signal.size() != n || mask.size() != n) {
        stop("beta, posterior_signal, and mask must match dims.");
    }

    beta = clone(beta);
    NumericVector beta_bar = clone(beta);
    NumericVector beta_old(n), px(n), py(n), pz(n), adjoint(n);
    const double tau = 0.9;
    const double sigma = 0.08;
    int completed = 0;

    for (int iteration = 1; iteration <= iterations; ++iteration) {
        for (int z = 0; z < nz; ++z) {
            for (int y = 0; y < ny; ++y) {
                for (int x = 0; x < nx; ++x) {
                    const R_xlen_t i = index3(x, y, z, nx, ny);
                    if (x + 1 < nx) {
                        px[i] = clamp_double(
                            px[i] + sigma * (
                                beta_bar[index3(x + 1, y, z, nx, ny)] -
                                beta_bar[i]
                            ),
                            -lambda, lambda
                        );
                    }
                    if (y + 1 < ny) {
                        py[i] = clamp_double(
                            py[i] + sigma * (
                                beta_bar[index3(x, y + 1, z, nx, ny)] -
                                beta_bar[i]
                            ),
                            -lambda, lambda
                        );
                    }
                    if (z + 1 < nz) {
                        pz[i] = clamp_double(
                            pz[i] + sigma * (
                                beta_bar[index3(x, y, z + 1, nx, ny)] -
                                beta_bar[i]
                            ),
                            -lambda, lambda
                        );
                    }
                }
            }
        }

        std::fill(adjoint.begin(), adjoint.end(), 0.0);
        for (int z = 0; z < nz; ++z) {
            for (int y = 0; y < ny; ++y) {
                for (int x = 0; x < nx; ++x) {
                    const R_xlen_t i = index3(x, y, z, nx, ny);
                    if (x + 1 < nx) {
                        const R_xlen_t j = index3(x + 1, y, z, nx, ny);
                        adjoint[i] -= px[i];
                        adjoint[j] += px[i];
                    }
                    if (y + 1 < ny) {
                        const R_xlen_t j = index3(x, y + 1, z, nx, ny);
                        adjoint[i] -= py[i];
                        adjoint[j] += py[i];
                    }
                    if (z + 1 < nz) {
                        const R_xlen_t j = index3(x, y, z + 1, nx, ny);
                        adjoint[i] -= pz[i];
                        adjoint[j] += pz[i];
                    }
                }
            }
        }

        std::copy(beta.begin(), beta.end(), beta_old.begin());
        double change_sum = 0.0;
        R_xlen_t mask_count = 0;
        for (R_xlen_t i = 0; i < n; ++i) {
            if (mask[i] == TRUE) {
                const double likelihood_gradient =
                    expit_stable(beta_old[i]) - posterior_signal[i];
                beta[i] = clamp_double(
                    beta_old[i] - tau * (
                        likelihood_gradient + adjoint[i]
                    ),
                    -12.0, 12.0
                );
                change_sum += std::abs(beta[i] - beta_old[i]);
                ++mask_count;
            } else {
                beta[i] = 0.0;
            }
            beta_bar[i] = 2.0 * beta[i] - beta_old[i];
        }
        completed = iteration;
        if (iteration % 10 == 0 && mask_count > 0 &&
            change_sum / static_cast<double>(mask_count) < tolerance) {
            break;
        }
    }
    beta.attr("dim") = dims;
    return List::create(
        _["beta"] = beta,
        _["iterations"] = completed
    );
}

// [[Rcpp::export]]
List wm_fdr_posterior_objective_cpp(
    const NumericVector& values,
    const NumericVector& beta,
    const NumericVector& grid,
    const NumericVector& density,
    const LogicalVector& mask
) {
    const R_xlen_t n = values.size();
    if (beta.size() != n || mask.size() != n) {
        stop("values, beta, and mask must have equal length.");
    }
    NumericVector posterior(n);
    double objective = 0.0;
    for (R_xlen_t i = 0; i < n; ++i) {
        if (mask[i] != TRUE) {
            posterior[i] = 0.0;
            continue;
        }
        const double f1 = std::max(
            interpolate_linear(grid, density, values[i]), 1e-12
        );
        const double f0 = std::max(
            R::dnorm4(values[i], 0.0, 1.0, false), 1e-12
        );
        const double prior = expit_stable(beta[i]);
        posterior[i] = expit_stable(beta[i] + std::log(f1) - std::log(f0));
        const double mixture = std::max(
            prior * f1 + (1.0 - prior) * f0, 1e-300
        );
        objective -= std::log(mixture);
    }
    posterior.attr("dim") = values.attr("dim");
    return List::create(
        _["posterior_signal"] = posterior,
        _["objective"] = objective
    );
}

// [[Rcpp::export]]
NumericVector wm_score_candidates_cpp(
    const NumericVector& candidate,
    const NumericVector& beta,
    const NumericVector& grid,
    const NumericVector& density,
    const LogicalVector& mask,
    const double eps = 1e-10
) {
    const R_xlen_t n = candidate.size();
    if (beta.size() != n || mask.size() != n) {
        stop("candidate, beta, and mask must have equal length.");
    }
    NumericVector score(n);
    for (R_xlen_t i = 0; i < n; ++i) {
        if (mask[i] != TRUE) {
            score[i] = 1.0;
            continue;
        }
        const double f1 = std::max(
            interpolate_linear(grid, density, candidate[i]), 1e-12
        );
        const double f0 = std::max(
            R::dnorm4(candidate[i], 0.0, 1.0, false), 1e-12
        );
        const double posterior_signal = expit_stable(
            beta[i] + std::log(f1) - std::log(f0)
        );
        score[i] = clamp_double(1.0 - posterior_signal, eps, 1.0 - eps);
    }
    score.attr("dim") = candidate.attr("dim");
    return score;
}

// [[Rcpp::export]]
NumericVector wm_log_score_candidates_cpp(
    const NumericVector& candidate,
    const NumericVector& beta,
    const NumericVector& grid,
    const NumericVector& density,
    const LogicalVector& mask
) {
    const R_xlen_t n = candidate.size();
    if (beta.size() != n || mask.size() != n) {
        stop("candidate, beta, and mask must have equal length.");
    }
    NumericVector log_score(n);
    for (R_xlen_t i = 0; i < n; ++i) {
        if (mask[i] != TRUE) {
            log_score[i] = 0.0;
            continue;
        }
        const double f1 = std::max(
            interpolate_linear(grid, density, candidate[i]), 1e-12
        );
        const double f0 = std::max(
            R::dnorm4(candidate[i], 0.0, 1.0, false), 1e-12
        );
        const double log_signal_odds =
            beta[i] + std::log(f1) - std::log(f0);
        log_score[i] = log_expit_negative_stable(log_signal_odds);
    }
    log_score.attr("dim") = candidate.attr("dim");
    return log_score;
}


