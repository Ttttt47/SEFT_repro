#!/usr/bin/env Rscript

source(file.path("R", "working_models", "working_model_dispatch.R"))

stub_scores <- function(x, x_til, mask, ..., model = "stub") {
    background <- .seft_absmax_background(x, x_til, mask)
    log_r <- -abs(x - background)
    log_r_til <- -abs(x_til - background)
    list(
        R = exp(log_r), R_til = exp(log_r_til),
        log_R = log_r, log_R_til = log_r_til,
        background = background, fit = list(ok = TRUE), model = model
    )
}

cal_CLAW_scores_3d <- function(x, x_til, p, p_til, ...) {
    result <- stub_scores(x, x_til, array(TRUE, dim = dim(x)), model = "co_stub")
    result[c("R", "R_til")]
}
cal_fdr_smoothing_scores_3d_absmax <- function(x, x_til, mask, ...) {
    stub_scores(x, x_til, mask, model = "fdr_smoothing_stub")
}
cal_deepfdr_scores_3d_absmax <- function(x, x_til, mask, ...) {
    stub_scores(x, x_til, mask, model = "deepfdr_stub")
}
cal_fchmrf_scores_3d_absmax <- function(x, x_til, mask, ...) {
    stub_scores(x, x_til, mask, model = "fchmrf_stub")
}
cal_ising_scores_3d_absmax <- function(x, x_til, mask, ...) {
    stub_scores(x, x_til, mask, model = "ising_stub")
}

set.seed(7)
x <- array(rnorm(5 * 6 * 7), dim = c(5, 6, 7))
x_til <- array(rnorm(length(x)), dim = dim(x))
mask <- array(FALSE, dim = dim(x))
mask[2:5, 2:6, 2:7] <- TRUE

for (model in seft_working_model_choices()) {
    result <- run_seft_working_model(
        x, x_til, mask = mask, working_model = model, seed = 13L,
        density_bandwidth = 1, spatial_bandwidth = 5,
        lambda = 0.5, neighbor_range = 10L, score_clip_c = 0.99
    )
    stopifnot(
        identical(dim(result$R), dim(x)),
        identical(dim(result$R_til), dim(x)),
        identical(dim(result$log_R), dim(x)),
        identical(dim(result$log_R_til), dim(x)),
        identical(dim(result$background), dim(x)),
        identical(result$model_identifier, model),
        all(is.finite(result$log_R)),
        all(is.finite(result$log_R_til)),
        all(result$R[!mask] == 1),
        all(result$R_til[!mask] == 1)
    )
}

stopifnot(
    normalize_seft_working_model("covariate-adaptive") == "co",
    normalize_seft_working_model("fdrs") == "fdr-smoothing"
)
cat("PASS unified working-model dispatcher\n")
