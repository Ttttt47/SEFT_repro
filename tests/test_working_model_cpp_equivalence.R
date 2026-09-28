#!/usr/bin/env Rscript

project_root <- normalizePath(
    file.path(dirname(sub(
        "^--file=", "",
        grep("^--file=", commandArgs(FALSE), value = TRUE)[[1]]
    )), ".."),
    mustWork = TRUE
)
setwd(project_root)
Sys.setenv(SEFT_WORKING_MODEL_CPP = file.path(
    project_root, "src", "working_models",
    "working_model_scores.cpp"
))
source(file.path(
    project_root, "R", "working_models",
    "working_model_scores.R"
))

set.seed(20260723)
z <- rnorm(160)
grid <- seq(-7, 7, length.out = 81)
pr_r <- .wm_predictive_recursion_r(z, grid, sweeps = 2L, seed = 91L)
pr_cpp <- .wm_predictive_recursion(z, grid, sweeps = 2L, seed = 91L)
stopifnot(
    max(abs(pr_r$density - pr_cpp$density)) < 1e-12,
    abs(pr_r$pi0 - pr_cpp$pi0) < 1e-12
)

dims <- c(7L, 6L, 5L)
beta <- array(rnorm(prod(dims), sd = 0.2), dims)
posterior <- array(runif(prod(dims)), dims)
mask <- array(runif(prod(dims)) > 0.08, dims)
tv_r <- .wm_tv_logistic_mstep_r(
    beta, posterior, lambda = 0.1, mask = mask, iterations = 30L
)
tv_cpp <- .wm_tv_logistic_mstep(
    beta, posterior, lambda = 0.1, mask = mask, iterations = 30L
)
stopifnot(max(abs(tv_r - tv_cpp)) < 1e-12)

candidate <- array(rnorm(prod(dims)), dims)
score_cpp <- wm_score_candidates_cpp(
    candidate, tv_cpp, grid, pr_cpp$density, as.logical(mask), 1e-10
)
f1_r <- pmax(approx(
    grid, pr_cpp$density, xout = candidate,
    rule = 2, ties = "ordered"
)$y, 1e-12)
f0_r <- pmax(dnorm(candidate), 1e-12)
score_r <- 1 - .wm_expit(tv_cpp + log(f1_r) - log(f0_r))
score_r <- pmax(pmin(score_r, 1 - 1e-10), 1e-10)
score_r[!mask] <- 1
stopifnot(max(abs(score_r - score_cpp)) < 1e-12)

log_score_cpp <- wm_log_score_candidates_cpp(
    candidate, tv_cpp, grid, pr_cpp$density, as.logical(mask)
)
eta_r <- tv_cpp + log(f1_r) - log(f0_r)
log_score_r <- ifelse(
    eta_r >= 0,
    -eta_r - log1p(exp(-eta_r)),
    -log1p(exp(eta_r))
)
log_score_r[!mask] <- 0
stopifnot(max(abs(log_score_r - log_score_cpp)) < 1e-12)

cat("R/C++ equivalence tests passed.\n")
