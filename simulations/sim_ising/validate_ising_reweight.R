#!/usr/bin/env Rscript

# Exact 2x2x2 audit of the formal f/g single-site update and its swap
# equivariance. This test uses the same abs-max emissions as the paper run.

find_repro_root <- function() {
  file_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  candidates <- c(
    getwd(),
    if (length(file_arg)) file.path(
      dirname(sub("^--file=", "", file_arg[[1]])), "..", ".."
    )
  )
  for (candidate in candidates) {
    candidate <- normalizePath(candidate, mustWork = FALSE)
    if (file.exists(file.path(candidate, "R", "core", "Ising_functions.R"))) {
      return(candidate)
    }
  }
  stop("Cannot locate SEFT_repro.")
}

repro_root <- find_repro_root()
setwd(repro_root)
environment_bin <- normalizePath(
  file.path(R.home(), "..", "..", "bin"), mustWork = FALSE
)
Sys.setenv(PATH = paste(
  environment_bin, Sys.getenv("PATH"), sep = .Platform$path.sep
))
source(file.path("R", "core", "Ising_functions.R"))

L <- 2L
n <- L^3
beta <- 0.65
h <- -1.7
mu <- 2.1
sig2 <- 1
set.seed(20260731)
x <- array(rnorm(n, mean = 0.4), dim = rep(L, 3L))
y <- array(rnorm(n), dim = rep(L, 3L))
w <- array(ifelse(abs(x) >= abs(y), x, y), dim = dim(x))

f0 <- function(z) dnorm(z)
f1 <- function(z) dnorm(z, mu, sqrt(sig2))
F0abs <- function(t) pnorm(t) - pnorm(-t)
F1abs <- function(t) pnorm(t, mu, sqrt(sig2)) - pnorm(-t, mu, sqrt(sig2))
g0 <- function(z) 2 * f0(z) * F0abs(abs(z))
g1 <- function(z) f1(z) * F0abs(abs(z)) + f0(z) * F1abs(abs(z))

index <- function(i, j, k) i + L * (j - 1L) + L^2 * (k - 1L)
edges <- matrix(integer(), ncol = 2L)
for (k in seq_len(L)) for (j in seq_len(L)) for (i in seq_len(L)) {
  here <- index(i, j, k)
  if (i < L) edges <- rbind(edges, c(here, index(i + 1L, j, k)))
  if (j < L) edges <- rbind(edges, c(here, index(i, j + 1L, k)))
  if (k < L) edges <- rbind(edges, c(here, index(i, j, k + 1L)))
}
states <- do.call(rbind, lapply(0:(2^n - 1L), function(value) {
  as.integer(intToBits(value))[seq_len(n)]
}))
log_prior <- apply(states, 1L, function(theta) {
  beta * sum(theta[edges[, 1L]] * theta[edges[, 2L]]) + h * sum(theta)
})

posterior_signal <- function(candidate_site = NULL, candidate_value = NULL) {
  log_weight <- log_prior
  for (site in seq_len(n)) {
    if (!is.null(candidate_site) && site == candidate_site) {
      log_weight <- log_weight + ifelse(
        states[, site] == 1L,
        log(f1(candidate_value)), log(f0(candidate_value))
      )
    } else {
      log_weight <- log_weight + ifelse(
        states[, site] == 1L,
        log(g1(w[[site]])), log(g0(w[[site]]))
      )
    }
  }
  weight <- exp(log_weight - max(log_weight))
  colSums(states * weight) / sum(weight)
}

gamma_w <- posterior_signal()
nb <- hmrf_build_neighbors_6n(array(TRUE, dim = rep(L, 3L)))
lis_x <- hmrf_plis_reweight_absmax(
  nb, x, w, array(gamma_w, dim = dim(x)), 1, mu, sig2, 1e-12
)
lis_y <- hmrf_plis_reweight_absmax(
  nb, y, w, array(gamma_w, dim = dim(x)), 1, mu, sig2, 1e-12
)
exact_lis_x <- exact_lis_y <- numeric(n)
for (site in seq_len(n)) {
  exact_lis_x[[site]] <- 1 - posterior_signal(site, x[[site]])[[site]]
  exact_lis_y[[site]] <- 1 - posterior_signal(site, y[[site]])[[site]]
}
formula_error <- max(
  abs(as.vector(lis_x) - exact_lis_x),
  abs(as.vector(lis_y) - exact_lis_y)
)

swap_site <- 3L
x_swapped <- x
y_swapped <- y
x_swapped[[swap_site]] <- y[[swap_site]]
y_swapped[[swap_site]] <- x[[swap_site]]
w_swapped <- array(
  ifelse(abs(x_swapped) >= abs(y_swapped), x_swapped, y_swapped),
  dim = dim(x)
)
baseline_error <- max(abs(w_swapped - w))
lis_x_swapped <- hmrf_plis_reweight_absmax(
  nb, x_swapped, w_swapped, array(gamma_w, dim = dim(x)), 1, mu, sig2, 1e-12
)
lis_y_swapped <- hmrf_plis_reweight_absmax(
  nb, y_swapped, w_swapped, array(gamma_w, dim = dim(x)), 1, mu, sig2, 1e-12
)
swap_error <- max(
  abs(lis_x_swapped[[swap_site]] - lis_y[[swap_site]]),
  abs(lis_y_swapped[[swap_site]] - lis_x[[swap_site]]),
  abs(lis_x_swapped[-swap_site] - lis_x[-swap_site]),
  abs(lis_y_swapped[-swap_site] - lis_y[-swap_site])
)

output_dir <- file.path(
  "outputs", "intermediate", "simulation_results",
  "ising_B100_seed20260731", "validation"
)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
audit <- list(
  lattice = "2x2x2",
  enumerated_states = nrow(states),
  formula_max_abs_error = formula_error,
  baseline_swap_max_abs_error = baseline_error,
  score_swap_max_abs_error = swap_error,
  passed = formula_error < 1e-10 && baseline_error == 0 && swap_error < 1e-12
)
saveRDS(audit, file.path(output_dir, "exact_reweight_and_swap_audit.rds"))
writeLines(capture.output(str(audit)), file.path(
  output_dir, "exact_reweight_and_swap_audit.log"
))
if (!audit$passed) stop("Ising reweighting audit failed.")
cat(sprintf(
  "PASS formula_error=%.3g baseline_swap_error=%.3g score_swap_error=%.3g\n",
  formula_error, baseline_error, swap_error
))
