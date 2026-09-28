library(Rcpp)

resolve_project_file <- function(rel_path) {
  candidates <- c(
    rel_path,
    file.path("..", rel_path),
    file.path("..", "..", rel_path),
    file.path("..", "..", "..", rel_path)
  )
  hits <- candidates[file.exists(candidates)]
  if (length(hits) == 0) {
    stop(sprintf("Cannot locate required file: %s", rel_path))
  }
  normalizePath(hits[[1]], winslash = "/", mustWork = TRUE)
}

Rcpp::sourceCpp(resolve_project_file(file.path("src", "hmrf_gem.cpp")))

Rcpp::cppFunction(code = '
#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
IntegerVector ising3d_gibbs_01(int Lx, int Ly, int Lz,
                              double beta, double h,
                              int n_sweeps,
                              double init_prob = 0.5,
                              int seed = 0) {
  const int N = Lx * Ly * Lz;
  IntegerVector th(N);

  // RNG seed (optional)
  if (seed != 0) {
    // use base R RNG; set.seed in R is recommended, but we also provide this
    Function set_seed("set.seed");
    set_seed(seed);
  }

  // init: iid Bernoulli(init_prob)
  for (int idx = 0; idx < N; ++idx) {
    th[idx] = (R::runif(0.0, 1.0) < init_prob) ? 1 : 0;
  }

  auto idx3 = [Lx, Ly](int x, int y, int z) {
    return x + Lx * (y + Ly * z);
  };

  for (int sweep = 0; sweep < n_sweeps; ++sweep) {
    // fixed scan order (can randomize if needed)
    for (int z = 0; z < Lz; ++z) {
      for (int y = 0; y < Ly; ++y) {
        for (int x = 0; x < Lx; ++x) {
          int s = idx3(x, y, z);

          int neigh_sum = 0;
          // x-1, x+1
          if (x > 0)       neigh_sum += th[idx3(x-1, y, z)];
          if (x < Lx - 1)  neigh_sum += th[idx3(x+1, y, z)];
          // y-1, y+1
          if (y > 0)       neigh_sum += th[idx3(x, y-1, z)];
          if (y < Ly - 1)  neigh_sum += th[idx3(x, y+1, z)];
          // z-1, z+1
          if (z > 0)       neigh_sum += th[idx3(x, y, z-1)];
          if (z < Lz - 1)  neigh_sum += th[idx3(x, y, z+1)];

          // Conditional:
          // log P(theta_s=1|rest) - log P(theta_s=0|rest) = beta * neigh_sum + h
          double logit = beta * (double)neigh_sum + h;
          double p1 = 1.0 / (1.0 + std::exp(-logit));

          th[s] = (R::runif(0.0, 1.0) < p1) ? 1 : 0;
        }
      }
    }
  }

  return th;
}
')

print('Ising model cpp functions loaded.')

source(resolve_project_file(file.path("R", "core", "CLAW_functions.R")))


simulate_emission_mixnorm <- function(n, weights, means, vars) {
  w <- weights / sum(weights)
  k <- sample.int(length(w), size = n, replace = TRUE, prob = w)
  rnorm(n, mean = means[k], sd = sqrt(vars[k]))
}

simulate_hmrf_data <- function(L = 20,
                               beta = 0.8, h = -2.5,
                               burnin_theta = 1000, sweeps_theta = 500,
                               f0_mean = 0, f0_var = 1,
                               f1_weights = c(1),
                               f1_means = c(3),
                               f1_vars = c(1),
                               seed = 1) {
  if (!exists("ising3d_gibbs_01")) stop("ising3d_gibbs_01() not found.")
  set.seed(seed)

  th_vec <- ising3d_gibbs_01(L, L, L,
                            beta = beta, h = h,
                            n_sweeps = burnin_theta + sweeps_theta,
                            init_prob = 0.5, seed = 0)
  theta <- array(th_vec, dim = c(L, L, L))

  n <- L^3
  x <- numeric(n)
  idx1 <- which(th_vec == 1L)
  idx0 <- which(th_vec == 0L)

  x[idx0] <- rnorm(length(idx0), mean = f0_mean, sd = sqrt(f0_var))
  x[idx1] <- simulate_emission_mixnorm(length(idx1), f1_weights, f1_means, f1_vars)

  list(theta = theta, x3d = array(x, dim = c(L, L, L)))
}


