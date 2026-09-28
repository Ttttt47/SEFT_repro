#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

namespace {

double logsumexp(const arma::vec& values) {
    const double maximum = values.max();
    if (maximum == -std::numeric_limits<double>::infinity()) return maximum;
    return maximum + std::log(arma::sum(arma::exp(values - maximum)));
}

arma::mat signed_log_difference(const arma::vec& x, const arma::vec& y) {
    arma::mat answer(x.n_elem, 2, arma::fill::zeros);
    for (arma::uword i = 0; i < x.n_elem; ++i) {
        const double larger = std::max(x(i), y(i));
        const double smaller = std::min(x(i), y(i));
        if (larger == -arma::datum::inf || larger == smaller) {
            answer(i, 1) = -arma::datum::inf;
            continue;
        }
        answer(i, 0) = x(i) >= y(i) ? 1.0 : -1.0;
        answer(i, 1) = larger + std::log1p(-std::exp(smaller - larger));
    }
    return answer;
}

} // namespace

// [[Rcpp::export]]
Rcpp::List cal_CLAW_scores_3d(
    arma::cube T_data, arma::cube T_data_til,
    arma::cube P, arma::cube P_til,
    double lambda = 0.5, double h = 2.5,
    double bandwidth = 2.5, int neighbor_range = 5,
    double c = 0.99
) {
    if (T_data.n_rows != T_data_til.n_rows || T_data.n_cols != T_data_til.n_cols ||
        T_data.n_slices != T_data_til.n_slices || T_data.n_rows != P.n_rows ||
        T_data.n_cols != P.n_cols || T_data.n_slices != P.n_slices ||
        T_data.n_rows != P_til.n_rows || T_data.n_cols != P_til.n_cols ||
        T_data.n_slices != P_til.n_slices) {
        Rcpp::stop("All score inputs must have the same dimensions.");
    }
    if (!(lambda >= 0.0 && lambda < 1.0) || !(h > 0.0) ||
        !(bandwidth > 0.0) || neighbor_range < 0 || !(c > 0.0 && c < 1.0)) {
        Rcpp::stop("Invalid CO tuning parameter.");
    }

    const int x_dim = T_data.n_rows;
    const int y_dim = T_data.n_cols;
    const int z_dim = T_data.n_slices;
    arma::cube pi(x_dim, y_dim, z_dim, arma::fill::zeros);
    arma::cube log_f(x_dim, y_dim, z_dim, arma::fill::value(-arma::datum::inf));
    arma::cube log_f_til(x_dim, y_dim, z_dim, arma::fill::value(-arma::datum::inf));
    arma::cube R(x_dim, y_dim, z_dim, arma::fill::value(-arma::datum::inf));
    arma::cube R_til(x_dim, y_dim, z_dim, arma::fill::value(-arma::datum::inf));
    arma::vec temporary(3);

    for (int i = 0; i < x_dim; ++i) {
        for (int j = 0; j < y_dim; ++j) {
            for (int k = 0; k < z_dim; ++k) {
                if (T_data(i, j, k) == 0.0) continue;
                double log_sum_w = -arma::datum::inf;
                for (int ii = i - neighbor_range; ii <= i + neighbor_range; ++ii) {
                    for (int jj = j - neighbor_range; jj <= j + neighbor_range; ++jj) {
                        for (int kk = k - neighbor_range; kk <= k + neighbor_range; ++kk) {
                            if (ii < 0 || ii >= x_dim || jj < 0 || jj >= y_dim ||
                                kk < 0 || kk >= z_dim || T_data(ii, jj, kk) == 0.0) continue;
                            const double distance = std::sqrt(
                                std::pow(i - ii, 2.0) + std::pow(j - jj, 2.0) + std::pow(k - kk, 2.0)
                            );
                            if (distance > neighbor_range) continue;
                            const double log_w = arma::log_normpdf<double>(distance, 0.0, bandwidth);
                            temporary(0) = log_sum_w;
                            temporary(1) = log_w;
                            log_sum_w = logsumexp(temporary.subvec(0, 1));

                            temporary(0) = log_w + arma::log_normpdf<double>(T_data(i, j, k) - T_data(ii, jj, kk), 0.0, h);
                            temporary(1) = log_w + arma::log_normpdf<double>(T_data(i, j, k) - T_data_til(ii, jj, kk), 0.0, h);
                            temporary(2) = log_f(i, j, k);
                            log_f(i, j, k) = logsumexp(temporary);
                            temporary(0) = log_w + arma::log_normpdf<double>(T_data_til(i, j, k) - T_data(ii, jj, kk), 0.0, h);
                            temporary(1) = log_w + arma::log_normpdf<double>(T_data_til(i, j, k) - T_data_til(ii, jj, kk), 0.0, h);
                            temporary(2) = log_f_til(i, j, k);
                            log_f_til(i, j, k) = logsumexp(temporary);
                            pi(i, j, k) += std::exp(log_w) *
                                (static_cast<int>(P(ii, jj, kk) > lambda) + static_cast<int>(P_til(ii, jj, kk) > lambda));
                        }
                    }
                }

                log_f(i, j, k) -= std::log(2.0) + log_sum_w;
                log_f_til(i, j, k) -= std::log(2.0) + log_sum_w;
                pi(i, j, k) = 1.0 - pi(i, j, k) / (2.0 * (1.0 - lambda) * std::exp(log_sum_w));
                pi(i, j, k) = std::max(0.0, std::min(0.5 - 1e-3, pi(i, j, k)));

                const double multiplier = (0.5 - pi(i, j, k)) / (1.0 - pi(i, j, k));
                double capped = std::min(
                    std::log(1.0 - pi(i, j, k)) + arma::log_normpdf(T_data(i, j, k)) - log_f(i, j, k),
                    std::log(c)
                );
                R(i, j, k) = multiplier * std::exp(capped) / (1.0 - std::exp(capped));
                capped = std::min(
                    std::log(1.0 - pi(i, j, k)) + arma::log_normpdf(T_data_til(i, j, k)) - log_f_til(i, j, k),
                    std::log(c)
                );
                R_til(i, j, k) = multiplier * std::exp(capped) / (1.0 - std::exp(capped));
            }
        }
    }
    return Rcpp::List::create(
        Rcpp::Named("pi") = pi,
        Rcpp::Named("log_f") = log_f,
        Rcpp::Named("log_f_til") = log_f_til,
        Rcpp::Named("R") = R,
        Rcpp::Named("R_til") = R_til
    );
}

// [[Rcpp::export]]
double Simes_PC_test_cpp(arma::vec p_values, int u) {
    const int n = p_values.n_elem;
    if (n < 1 || u < 1 || u > n) Rcpp::stop("u must be between 1 and the number of p-values.");
    const arma::vec ordered = arma::sort(p_values);
    const arma::vec retained = ordered.subvec(u - 1, n - 1);
    return arma::min((n - u + 1) * retained / arma::regspace(1, n - u + 1));
}

// [[Rcpp::export]]
double cal_ELIS_cpp(
    arma::mat log_PLIS_scores, int u,
    std::string method = "Sym_miny", double alpha_k = 0.05,
    std::string til_mth = "oneminus", bool signmin_flag = false
) {
    if (log_PLIS_scores.n_rows != 2 || log_PLIS_scores.n_cols < 1) {
        Rcpp::stop("log_PLIS_scores must be a non-empty 2-row matrix.");
    }
    if (u < 1 || u > static_cast<int>(log_PLIS_scores.n_cols)) {
        Rcpp::stop("u must be between 1 and the number of scores.");
    }
    if (method != "Sym_miny" || til_mth != "neglog" || signmin_flag) {
        Rcpp::stop("The package inference API uses Sym_miny with the neglog transformation.");
    }
    (void)alpha_k;
    const arma::mat transformed = arma::log(-log_PLIS_scores);
    const arma::mat difference = signed_log_difference(transformed.row(0).t(), transformed.row(1).t());
    const arma::vec log_magnitude = difference.col(1);
    const arma::uvec finite = arma::find_finite(log_magnitude);
    const double minimum = finite.n_elem ? arma::min(log_magnitude.elem(finite)) : 0.0;
    arma::vec statistics = (log_magnitude - minimum) % difference.col(0);
    statistics.elem(arma::find(difference.col(0) == 0.0)).zeros();

    double tau = std::abs(arma::min(statistics));
    const arma::vec absolute = arma::abs(statistics);
    if (arma::max(absolute) <= tau) return 0.0;
    const arma::vec larger = arma::sort(absolute.elem(arma::find(absolute > tau)));
    tau = larger(0);
    const double numerator = std::max(
        static_cast<double>(arma::sum(statistics >= tau)) - (u - 1.0), 0.0
    );
    return numerator / (1.0 + arma::sum(statistics <= -tau));
}
