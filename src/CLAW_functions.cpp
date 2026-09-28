#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
#include <iostream>
#include <algorithm>
#include <string>

using namespace arma;
using namespace Rcpp;

// [[Rcpp::export]]
double logsumexp_cpp(arma::vec logx) {
  double logxmax = max(logx);
  if (logxmax==-std::numeric_limits<double>::infinity()) {
    return -std::numeric_limits<double>::infinity();
  } else{
    return logxmax + log(sum(exp(logx - logxmax)));
  }
}

// [[Rcpp::export]]
arma::mat logsubtraction_cpp(const arma::vec& logx, const arma::vec& logy) {
    int n = logx.size();
    arma::mat result(n, 2);
    
    for (int i = 0; i < n; ++i) {
        double logx1 = std::max(logx(i), logy(i));
        double logx2 = std::min(logx(i), logy(i));
        double log_abs_value;
        if (logx1 == -arma::datum::inf) {
            log_abs_value = -arma::datum::inf;
        } else {
            log_abs_value = logx1 + std::log(1 - std::exp(logx2 - logx1));
        }
        if (log_abs_value == -arma::datum::inf) {
            result(i, 0) = 0;
        } else {
            result(i, 0) = (logx(i) - logy(i) >= 0) ? 1 : -1;
        }
        result(i, 1) = log_abs_value;
    }

    return result;
}


// [[Rcpp::export]]
Rcpp::List cal_CLAW_scores_3d(arma::cube T_data, arma::cube T_data_til, arma::cube P, arma::cube P_til, double lambda=0.5, double h=2.5, double bandwidth=2.5, int neighbor_range=5, double c=0.99) {
    int x_dim = T_data.n_rows;
    int y_dim = T_data.n_cols;
    int z_dim = T_data.n_slices;
    arma::cube pi(x_dim, y_dim, z_dim, arma::fill::value(0));
    arma::cube log_f(x_dim, y_dim, z_dim, arma::fill::value(-std::numeric_limits<double>::infinity()));
    arma::cube log_f_til(x_dim, y_dim, z_dim, arma::fill::value(-std::numeric_limits<double>::infinity()));
    arma::cube R(x_dim, y_dim, z_dim, arma::fill::value(-std::numeric_limits<double>::infinity()));
    arma::cube R_til(x_dim, y_dim, z_dim, arma::fill::value(-std::numeric_limits<double>::infinity()));
    double log_sum_w = -std::numeric_limits<double>::infinity();
    arma::vec temp(3);

    // for each voxel i, calculate w_ij, distance of voxel j & i for every voxel j in i's neighbor.
    const int progress_every = std::max(1, x_dim / 10);
    for (int i = 0; i < x_dim; i++) {
        if (i % progress_every == 0) cout << double(i)/x_dim*100 << "%" << endl;
        for (int j = 0; j < y_dim; j++) {
            for (int k = 0; k < z_dim; k++) {
                if (T_data(i,j,k)!=0){
                    log_sum_w = -std::numeric_limits<double>::infinity();
                    for (int ii = i-neighbor_range; ii < i+neighbor_range+1; ii++) {
                        for (int jj = j-neighbor_range; jj < j+neighbor_range+1; jj++) {
                            for (int kk = k-neighbor_range; kk < k+neighbor_range+1; kk++) {
                                if (ii >= 0 && ii < x_dim && jj >= 0 && jj < y_dim && kk >= 0 && kk < z_dim) {
                                    if (T_data(ii,jj,kk)!=0){
                                        
                                        temp[0] = i-ii;
                                        temp[1] = j-jj;
                                        temp[2] = k-kk;
                                        if (arma::norm(temp,2.0)>neighbor_range) continue;
                                        double log_w_ql = arma::log_normpdf<double>(arma::norm(temp,2.0), 0.0, bandwidth);
                                        
                                        temp[0] = log_sum_w;
                                        temp[1] = log_w_ql;
                                        log_sum_w = logsumexp_cpp(temp.subvec(0,1));
                                        
                                        // density f
                                        temp[0] = log_w_ql + arma::log_normpdf<double>(T_data(i,j,k)-T_data(ii,jj,kk),0.0,h);
                                        temp[1] = log_w_ql + arma::log_normpdf<double>(T_data(i,j,k)-T_data_til(ii,jj,kk),0.0,h);
                                        temp[2] = log_f(i,j,k);
                                        log_f(i,j,k) = logsumexp_cpp(temp);
                                        temp[0] = log_w_ql + arma::log_normpdf<double>(T_data_til(i,j,k)-T_data(ii,jj,kk),0.0,h);
                                        temp[1] = log_w_ql + arma::log_normpdf<double>(T_data_til(i,j,k)-T_data_til(ii,jj,kk),0.0,h);
                                        temp[2] = log_f_til(i,j,k);
                                        log_f_til(i,j,k) = logsumexp_cpp(temp);

                                        // signal spasity pi
                                        pi(i,j,k) = pi(i,j,k) + exp(log_w_ql)*(int(P(ii,jj,kk)>lambda)+int(P_til(ii,jj,kk)>lambda));
                                    }
                                }
                            }
                        }
                    }
                    
                    log_f(i,j,k) = log_f(i,j,k) - (log(2.0) + log_sum_w);
                    log_f_til(i,j,k) = log_f_til(i,j,k) - (log(2.0) + log_sum_w);

                    pi(i,j,k) = 1.0 - pi(i,j,k)/(2.0*(1.0-lambda)*exp(log_sum_w));
                    if (pi(i,j,k)<0) pi(i,j,k) = 0;
                    if (pi(i,j,k)>(0.5-1e-3)) pi(i,j,k) = 0.5-1e-3;

                    temp[0] = log(1-pi(i,j,k)) + arma::log_normpdf(T_data(i,j,k)) - log_f(i,j,k);
                    temp[1] = log(c);
                    temp[2] = min(temp.subvec(0,1));
                    R(i,j,k) = (0.5 - pi(i,j,k)) / (1-pi(i,j,k)) * exp(temp[2]) / (1-exp(temp[2]));


                    temp[0] = log(1-pi(i,j,k)) + arma::log_normpdf(T_data_til(i,j,k)) - log_f_til(i,j,k);
                    temp[1] = log(c);
                    temp[2] = min(temp.subvec(0,1));
                    R_til(i,j,k) = (0.5 - pi(i,j,k)) / (1-pi(i,j,k)) * exp(temp[2]) / (1.0-exp(temp[2]));
                    
                }
            }
        }
    }
    Rcpp::List result = Rcpp::List::create(Rcpp::Named("pi") = pi,
                                             Rcpp::Named("log_f") = log_f,
                                             Rcpp::Named("log_f_til") = log_f_til,
                                             Rcpp::Named("R") = R,
                                             Rcpp::Named("R_til") = R_til);
    return result;
}


// [[Rcpp::export]]
Rcpp::List cal_CLAW_scores_2d(arma::mat T_data, arma::mat T_data_til, arma::mat P, arma::mat P_til, double lambda=0.5, double h=2.5, double bandwidth=2.5, int neighbor_range=5, double c=0.99) {
    int x_dim = T_data.n_rows;
    int y_dim = T_data.n_cols;
    arma::mat pi(x_dim, y_dim, arma::fill::value(0));
    arma::mat log_f(x_dim, y_dim, arma::fill::value(-std::numeric_limits<double>::infinity()));
    arma::mat log_f_til(x_dim, y_dim, arma::fill::value(-std::numeric_limits<double>::infinity()));
    arma::mat R(x_dim, y_dim, arma::fill::value(-std::numeric_limits<double>::infinity()));
    arma::mat R_til(x_dim, y_dim, arma::fill::value(-std::numeric_limits<double>::infinity()));
    double log_sum_w = -std::numeric_limits<double>::infinity();
    arma::vec temp(3);
    // for each point q at grid (i,j), calculate w_ql, distance of point q & l for every point l in q's neighbor.
    const int progress_every = std::max(1, x_dim / 10);
    for (int i = 0; i < x_dim; i++) {
        if (i % progress_every == 0) cout << double(i)/x_dim*100 << "%" << endl;
        for (int j = 0; j < y_dim; j++) {
            if (T_data(i,j)!=0){
                log_sum_w = -std::numeric_limits<double>::infinity();
                for (int ii = i-neighbor_range; ii < i+neighbor_range+1; ii++) {
                    for (int jj = j-neighbor_range; jj < j+neighbor_range+1; jj++) {
                        if (ii >= 0 && ii < x_dim && jj >= 0 && jj < y_dim) {
                            if (T_data(ii,jj)!=0){
                                temp[0] = i-ii;
                                temp[1] = j-jj;
                                if (arma::norm(temp.subvec(0,1),2.0)>neighbor_range) continue;
                                double log_w_ql = arma::log_normpdf<double>(arma::norm(temp.subvec(0,1),2.0), 0, bandwidth);
                                
                                temp[0] = log_sum_w;
                                temp[1] = log_w_ql;
                                log_sum_w = logsumexp_cpp(temp.subvec(0,1));
                                
                                // density f
                                temp[0] = log_w_ql + arma::log_normpdf<double>(T_data(i,j)-T_data(ii,jj),0.0,h);
                                temp[1] = log_w_ql + arma::log_normpdf<double>(T_data(i,j)-T_data_til(ii,jj),0.0,h);
                                temp[2] = log_f(i,j);
                                log_f(i,j) = logsumexp_cpp(temp);
                                temp[0] = log_w_ql + arma::log_normpdf<double>(T_data_til(i,j)-T_data(ii,jj),0.0,h);
                                temp[1] = log_w_ql + arma::log_normpdf<double>(T_data_til(i,j)-T_data_til(ii,jj),0.0,h);
                                temp[2] = log_f_til(i,j);
                                log_f_til(i,j) = logsumexp_cpp(temp);

                                // signal spasity pi
                                pi(i,j) = pi(i,j) + exp(log_w_ql)*(int(P(ii,jj)>lambda)+int(P_til(ii,jj)>lambda));
                            }
                        }
                    }
                }
                // test if there is NAN, if there is, print the value
                if (std::isnan(log_sum_w) | std::isnan(log_f(i,j)) | std::isnan(log_f_til(i,j)) | std::isnan(pi(i,j))){
                    cout << "log_sum_w: " << log_sum_w << endl;
                    cout << "log_f: " << log_f(i,j) << endl;
                    cout << "log_f_til: " << log_f_til(i,j) << endl;
                    cout << "pi: " << pi(i,j) << endl;
                }
                log_f(i,j) = log_f(i,j) - (log(2.0) + log_sum_w);
                log_f_til(i,j) = log_f_til(i,j) - (log(2.0) + log_sum_w);

                pi(i,j) = 1 - pi(i,j)/(2.0*(1.0-lambda)*exp(log_sum_w));
                if (pi(i,j)<0) pi(i,j) = 0;
                if (pi(i,j)>(0.5-1e-3)) pi(i,j) = 0.5-1e-3;

                temp[0] = log(1-pi(i,j)) + arma::log_normpdf(T_data(i,j)) - log_f(i,j);
                temp[1] = log(c);
                temp[2] = min(temp.subvec(0,1));
                R(i,j) = (0.5 - pi(i,j)) / (1-pi(i,j)) * exp(temp[2]) / (1-exp(temp[2]));

                if (R(i,j) < 0){
                    cout << "pi: " << pi(i,j) << endl;
                    cout << "clfdr:" << exp(temp[2]) << endl;
                    cout << "R: " << R(i,j) << endl;
                }

                temp[0] = log(1-pi(i,j)) + arma::log_normpdf(T_data_til(i,j)) - log_f_til(i,j);
                temp[1] = log(c);
                temp[2] = min(temp.subvec(0,1));
                R_til(i,j) = (0.5 - pi(i,j)) / (1-pi(i,j)) * exp(temp[2]) / (1-exp(temp[2]));
                
                if (R_til(i,j) < 0){
                    cout << "pi: " << pi(i,j) << endl;
                    cout << "clfdr:" << exp(temp[2]) << endl;
                    cout << "R_til: " << R_til(i,j) << endl;
                }
            }
        }
    }
    Rcpp::List result = Rcpp::List::create(Rcpp::Named("pi") = pi,
                                             Rcpp::Named("log_f") = log_f,
                                             Rcpp::Named("log_f_til") = log_f_til,
                                             Rcpp::Named("R") = R,
                                             Rcpp::Named("R_til") = R_til);
    return result;
}


// [[Rcpp::export]]
double Simes_PC_test_cpp(arma::vec p_values, int u) {
    int n = p_values.n_elem;
    arma::vec sorted_p_values = arma::sort(p_values);
    arma::vec p_values_trunc = sorted_p_values.subvec(u - 1, n - 1);
    double PC_p_value = arma::min((n-u+1) * p_values_trunc / arma::regspace(1, n - u + 1));
    return PC_p_value;
}

// [[Rcpp::export]]
double Fisher_PC_test_cpp(arma::vec p_values, int u) {
    int n = p_values.n_elem;
    arma::vec sorted_p_values = arma::sort(p_values);
    arma::vec p_values_trunc = sorted_p_values.subvec(u - 1, n - 1);
    double PC_p_value = 1 - R::pchisq(-2 * arma::accu(arma::log(p_values_trunc)), 2 * (n - u + 1), 1, 0);
    return PC_p_value;
}

// [[Rcpp::export]]
double cal_AvgPLIS_cpp(arma::vec log_LIS_X_til, arma::vec log_LIS_Y_til, int u) {
    int n = log_LIS_X_til.n_elem;
    arma::uvec x_order = arma::sort_index(log_LIS_X_til);
    double temp = logsumexp_cpp(arma::join_cols(log_LIS_Y_til, arma::vec{arma::max(log_LIS_X_til)}));
    double log_AvgPLIS;
    if (temp == -arma::datum::inf) {
        return 0.0;
    } else {
        log_AvgPLIS = logsumexp_cpp(log_LIS_X_til(x_order.subvec(0, n - u))) - temp;
        return exp(log_AvgPLIS);
    }
}

// [[Rcpp::export]]
double cal_SymPLIS_cpp(arma::vec log_LIS_X_til, arma::vec log_LIS_Y_til, int u, std::string adaptive, double alpha_k, bool signmin_flag) {
    arma::vec sameorder_Ts;
    arma::vec abs_sameorder_Ts;
    if (adaptive == "miny") {
        if (signmin_flag) {
            arma::vec signs = arma::sign(arma::exp(log_LIS_X_til) - arma::exp(log_LIS_Y_til));
            sameorder_Ts = signs % arma::max(arma::exp(log_LIS_X_til), arma::exp(log_LIS_Y_til));
        } else {
            arma::mat log_sub = logsubtraction_cpp(log_LIS_X_til, log_LIS_Y_til);
            arma::vec log_sub_0 = log_sub.col(0);
            arma::vec log_sub_1 = log_sub.col(1);
            arma::uvec finite_inds = arma::find_finite(log_sub_1);
            double min_finite = 0.0;
            if (finite_inds.n_elem > 0) {
                min_finite = arma::min(log_sub_1.elem(finite_inds));
            }
            sameorder_Ts = (log_sub_1 - min_finite) % log_sub_0;
            sameorder_Ts.elem(arma::find(log_sub.col(0) == 0.0)).fill(0.0);
        }

        abs_sameorder_Ts = arma::abs(sameorder_Ts);
        double tau = std::abs(arma::min(sameorder_Ts));
        if (arma::max(abs_sameorder_Ts) > tau) {
            arma::vec sorted_abs_sameorder_Ts = arma::sort(abs_sameorder_Ts);
            arma::vec temp = sorted_abs_sameorder_Ts(arma::find(sorted_abs_sameorder_Ts - tau > 0));
            tau = temp[0];
            double R_t;
            arma::vec temp2(2);
            temp2[0] = double(arma::sum(sameorder_Ts >= tau)) - (u - 1.0);
            temp2[1] = 0.0;
            R_t = arma::max(temp2) / (1.0 + arma::sum((sameorder_Ts) <= -tau));

            return R_t;
        } else {
            return 0.0;
        }
    } else if (adaptive == "adapt") {
        if (signmin_flag) {
            arma::vec signs = arma::sign(arma::exp(log_LIS_X_til) - arma::exp(log_LIS_Y_til));
            sameorder_Ts = signs % arma::max(arma::exp(log_LIS_X_til), arma::exp(log_LIS_Y_til));
        } else {
            arma::mat log_sub = logsubtraction_cpp(log_LIS_X_til, log_LIS_Y_til);
            arma::vec log_sub_0 = log_sub.col(0);
            arma::vec log_sub_1 = log_sub.col(1);
            arma::uvec finite_inds = arma::find_finite(log_sub_1);
            double min_val = 0.0;
            if (finite_inds.n_elem > 0) {
                min_val = arma::min(log_sub_1.elem(finite_inds));
            }
            sameorder_Ts = (log_sub_1 - min_val) % log_sub_0;
            sameorder_Ts.elem(arma::find(log_sub.col(0) == 0.0)).fill(0.0);
        }

        abs_sameorder_Ts = arma::abs(sameorder_Ts);
        arma::vec temp2(2);

        // Efficient version: restrict candidates to the k-smallest negatives and choose tau as the next larger threshold from all absolute Ts
        arma::uvec neg_inds = arma::find(sameorder_Ts < 0);
        if (neg_inds.n_elem == 0) return sameorder_Ts.n_elem - 1;
        arma::vec sorted_abs_neg = arma::sort(arma::abs(sameorder_Ts.elem(neg_inds)));
        arma::vec sorted_all_abs = arma::sort(arma::abs(sameorder_Ts));
        double best_R = 0;
        double best_tau = std::numeric_limits<double>::infinity();
        temp2[0] = 5;
        temp2[1] = sorted_abs_neg.n_elem - 1;
        int start_idx = arma::min(temp2);
        for (int k = start_idx; k >= 0; k--) {
            if (sorted_abs_neg.n_elem - 1 - k < 0) break;  // Not enough negatives available.
            double current = sorted_abs_neg[sorted_abs_neg.n_elem - 1 - k];
            arma::uvec idx = arma::find(sorted_all_abs > current);
            double candidate_tau = (idx.n_elem > 0) ? sorted_all_abs[idx[0]] : std::numeric_limits<double>::infinity();
            temp2[0] = double(arma::sum(sameorder_Ts >= candidate_tau)) - (u - 1.0);
            temp2[1] = 0.0;
            double denominator = 1.0 + arma::sum(sameorder_Ts <= -candidate_tau);
            double R_t = arma::max(temp2) / denominator;
            if (R_t < best_R) {
                best_R = R_t;
                best_tau = candidate_tau;
                break; // Stop if the ratio is not improving.
            }             
            best_R = R_t;
            best_tau = candidate_tau;     
        }
        return best_R;

    } else if (adaptive == "adapt_grid") {
        if (signmin_flag) {
            arma::vec signs = arma::sign(arma::exp(log_LIS_X_til) - arma::exp(log_LIS_Y_til));
            sameorder_Ts = signs % arma::max(arma::exp(log_LIS_X_til), arma::exp(log_LIS_Y_til));
        } else {
            arma::mat log_sub = logsubtraction_cpp(log_LIS_X_til, log_LIS_Y_til);
            arma::vec log_sub_0 = log_sub.col(0);
            arma::vec log_sub_1 = log_sub.col(1);
            arma::uvec finite_inds = arma::find_finite(log_sub_1);
            double min_val = 0.0;
            if (finite_inds.n_elem > 0) {
                min_val = arma::min(log_sub_1.elem(finite_inds));
            }
            sameorder_Ts = (log_sub_1 - min_val) % log_sub_0;
            sameorder_Ts.elem(arma::find(log_sub.col(0) == 0.0)).fill(0.0);
        }
    
        abs_sameorder_Ts = arma::abs(sameorder_Ts);
        // New grid search: use B = total negatives = V_minus(0) and sweep along n grid points.
        arma::uvec neg_inds = arma::find(sameorder_Ts < 0);
        if (neg_inds.n_elem == 0) return sameorder_Ts.n_elem - 1;
        arma::vec sorted_abs_neg = arma::sort(arma::abs(sameorder_Ts.elem(neg_inds)));
        arma::vec sorted_all_abs = arma::sort(arma::abs(sameorder_Ts));
    
        int B = neg_inds.n_elem;              // total number of negatives (V_minus(0))
        int n_grid = 10;                     // number of grid points (modifiable)
        double best_R = 0;
        double best_tau = std::numeric_limits<double>::infinity();
    
        for (int i = 0; i < n_grid; i++) {
            // Determine grid index: choose candidate from sorted_abs_neg[ floor((i+1)*B/n_grid) - 1 ]
            int grid_idx = std::max(0, std::min((int)std::floor((i + 1) * B / double(n_grid)) - 1, B - 1));
            double current = sorted_abs_neg[grid_idx];
            arma::uvec idx = arma::find(sorted_all_abs > current);
            double candidate_tau = (idx.n_elem > 0) ? sorted_all_abs[idx[0]] : std::numeric_limits<double>::infinity();
    
            double numerator = double(arma::sum(sameorder_Ts >= candidate_tau)) - (u - 1.0);
            if (numerator < 0) {
                numerator = 0.0;
            }
            double denominator = 1.0 + arma::sum(sameorder_Ts <= -candidate_tau);
            double R_t = numerator / denominator;
            // Stop if ratio no longer improves:
            if (R_t < best_R) {
                best_R = R_t;
                best_tau = candidate_tau;
                return best_R;
            }
            best_R = R_t;
            best_tau = candidate_tau;
        }
        // if R_t is consistently improving, return the last R_t where V- first drop to 0.
        double current = sorted_abs_neg[sorted_abs_neg.n_elem - 1];
        arma::uvec idx = arma::find(sorted_all_abs > current);
        double candidate_tau = (idx.n_elem > 0) ? sorted_all_abs[idx[0]] : std::numeric_limits<double>::infinity();
        double numerator = double(arma::sum(sameorder_Ts >= candidate_tau)) - (u - 1.0);
        if (numerator < 0) {
            numerator = 0.0;
        }
        double denominator = 1.0 + arma::sum(sameorder_Ts <= -candidate_tau);
        double R_t = numerator / denominator;
        return R_t;
        
    } else if (adaptive == "fdr") {
        if (signmin_flag) {
            arma::vec signs = arma::sign(arma::exp(log_LIS_X_til) - arma::exp(log_LIS_Y_til));
            sameorder_Ts = signs % arma::max(arma::exp(log_LIS_X_til), arma::exp(log_LIS_Y_til));
        } else {
            arma::mat log_sub = logsubtraction_cpp(log_LIS_X_til, log_LIS_Y_til);
            arma::vec log_sub_0 = log_sub.col(0);
            arma::vec log_sub_1 = log_sub.col(1);
            arma::uvec finite_inds = arma::find_finite(log_sub_1);
            double min_val = 0.0;
            if (finite_inds.n_elem > 0) {
                min_val = arma::min(log_sub_1.elem(finite_inds));
            }
            sameorder_Ts = (log_sub_1 - min_val) % log_sub_0;
            (sameorder_Ts.elem(arma::find(log_sub_0 == 0.0))).fill(0.0);
        }
        double t;
        arma::vec sorted_abs_sameorder_Ts = arma::sort(arma::abs(sameorder_Ts));
        for (int i = 0; i < sorted_abs_sameorder_Ts.n_elem; i++) {
            t = sorted_abs_sameorder_Ts(i);
            double denominator = arma::sum(sameorder_Ts >= t);
            if (denominator != 0) {
                double Q_t = (1.0 + arma::sum(sameorder_Ts <= -t)) / denominator;
                if (Q_t <= alpha_k) {
                    break;
                }
            } else {
                return 0.0;
            }
            
        }
        arma::vec temp(2);
        temp[0] = arma::sum((sameorder_Ts) >= t) - (u - 1.0);
        temp[1] = 0.0;
        double s = arma::max(temp) / (1.0 + arma::sum((sameorder_Ts) <= -t));
        return s;
    } else {
        Rcpp::stop("Adaptive method not supported.");
        return 0.0;
    }
}

// [[Rcpp::export]]
double cal_ELIS_cpp(arma::mat log_PLIS_scores, int u, std::string method = "Sym_miny", double alpha_k = 0.05, std::string til_mth = "oneminus", bool signmin_flag = false) {
    arma::mat log_PLIS_scores_til;
    if (til_mth == "oneminus") {
        log_PLIS_scores_til = arma::log(1 - arma::exp(log_PLIS_scores));
    } else if (til_mth == "reciprocal") {
        log_PLIS_scores_til = -log_PLIS_scores;
    } else if (til_mth == "neglog") {
        log_PLIS_scores_til = arma::log(-log_PLIS_scores);
    } else {
        Rcpp::stop("Transformation method not supported.");
    }
    arma::vec log_PLIS_scores_til_0 = log_PLIS_scores_til.row(0).t();
    arma::vec log_PLIS_scores_til_1 = log_PLIS_scores_til.row(1).t();
    if (method == "Avg") {
        return cal_AvgPLIS_cpp(log_PLIS_scores_til_0, log_PLIS_scores_til_1, u);
    } else if (method == "Sym_miny") {
        return cal_SymPLIS_cpp(log_PLIS_scores_til_0, log_PLIS_scores_til_1, u, "miny", alpha_k, signmin_flag);
    } else if (method == "Sym_fdr") {
        return cal_SymPLIS_cpp(log_PLIS_scores_til_0, log_PLIS_scores_til_1, u, "fdr", alpha_k, signmin_flag);
    } else if (method == "Sym_adapt") {
        return cal_SymPLIS_cpp(log_PLIS_scores_til_0, log_PLIS_scores_til_1, u, "adapt", alpha_k, signmin_flag);
    } else if (method == "Sym_adapt_grid") {
        return cal_SymPLIS_cpp(log_PLIS_scores_til_0, log_PLIS_scores_til_1, u, "adapt_grid", alpha_k, signmin_flag);
    } else {
        Rcpp::stop("Method not supported.");
        return 0.0;
    }
    return 0.0;
}
