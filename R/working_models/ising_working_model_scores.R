# Abs-max PLIS adaptation of the six-neighbour 3-D Ising/HMRF working model.

.wm_load_ising_cpp <- function() {
    if (exists(
        "hmrf_plis_reweight_absmax",
        mode = "function", inherits = TRUE
    )) return(invisible(TRUE))
    environment_bin <- normalizePath(
        file.path(R.home(), "..", "..", "bin"), mustWork = FALSE
    )
    Sys.setenv(PATH = paste(
        environment_bin, Sys.getenv("PATH"), sep = .Platform$path.sep
    ))
    candidates <- c(
        file.path(getwd(), "src", "hmrf_gem.cpp"),
        file.path(getwd(), "src", "hmrf_gem.cpp")
    )
    candidates <- candidates[file.exists(candidates)]
    if (!length(candidates)) stop("Cannot locate SEFT_repro/src/hmrf_gem.cpp.")
    suppressPackageStartupMessages(requireNamespace("Rcpp", quietly = TRUE))
    Rcpp::sourceCpp(normalizePath(candidates[[1]]), rebuild = FALSE)
    invisible(TRUE)
}

.wm_load_ising_cpp()

#' Six-neighbour Ising/HMRF score with a swap-invariant abs-max baseline.
#'
#' The fitted baseline posterior is shared by the observed and mirror
#' candidates. Candidate emissions are ordinary f0/f1, while baseline
#' emissions are the corresponding abs-max g0/g1.
cal_ising_scores_3d_absmax <- function(
    x,
    x_til,
    mask = NULL,
    seed = 1L,
    iter_max = 20L,
    sweep_b = 1L,
    sweep_r = 2L,
    burnin_lis = 20L,
    sweep_lis = 50L,
    n_chains = 2L,
    mixture_components = 1L,
    fixing_beta = FALSE,
    fit_attempts = 1L,
    basic_convergence_tol = Inf,
    basic_convergence_max = Inf,
    require_basic_convergence = FALSE,
    verbose = FALSE
) {
    if (!exists(".wm_validate_pair", mode = "function", inherits = TRUE)) {
        stop("Source working_model_scores.R before this file.")
    }
    mask <- .wm_validate_pair(x, x_til, mask)
    background <- make_plis_background(x, x_til, mask)
    nb <- hmrf_build_neighbors_6n(mask)
    components <- as.integer(mixture_components)
    if (components < 1L) stop("mixture_components must be positive.")
    p_init <- rep(1 / components, components)
    fit_attempts <- as.integer(fit_attempts)
    if (fit_attempts < 1L) stop("fit_attempts must be positive.")
    fit_started <- proc.time()[["elapsed"]]
    fits <- vector("list", fit_attempts)
    fit_audits <- vector("list", fit_attempts)
    selected_attempt <- NA_integer_
    for (attempt in seq_len(fit_attempts)) {
        attempt_seed <- as.integer(seed + (attempt - 1L) * 1000003L)
        candidate <- hmrf_gem_fit(
            nb = nb,
            x3d = background,
            L = components,
            iter_max = as.integer(iter_max),
            sweep_b = as.integer(sweep_b),
            sweep_r = as.integer(sweep_r),
            a = 1, b = 2,
            alpha = 1e-3,
            stpmax = 1,
            max_backtrack = 10L,
            tol = 1e-4,
            beta_init = 0.5,
            h_init = -2,
            p_init = p_init,
            mu_init = rep(1, components),
            sig2_init = rep(1, components),
            init_prob_theta = 0.8,
            seed = attempt_seed,
            verbose = isTRUE(verbose),
            f0_absmax2 = TRUE,
            f1_absmax01 = TRUE,
            fixing_beta = isTRUE(fixing_beta)
        )
        rel <- vapply(
            candidate$trace,
            function(value) as.numeric(value$rel),
            numeric(1)
        )
        tail_rel <- tail(rel[is.finite(rel)], 5L)
        tail_median <- if (length(tail_rel)) median(tail_rel) else Inf
        tail_max <- if (length(tail_rel)) max(tail_rel) else Inf
        basic_converged <- (
            tail_median <= basic_convergence_tol &&
                tail_max <= basic_convergence_max
        )
        fits[[attempt]] <- candidate
        fit_audits[[attempt]] <- data.frame(
            attempt = attempt,
            seed = attempt_seed,
            iterations = length(rel),
            last_rel = if (length(rel)) tail(rel, 1L) else NA_real_,
            tail5_median_rel = tail_median,
            tail5_max_rel = tail_max,
            basic_converged = basic_converged
        )
        if (basic_converged) {
            selected_attempt <- attempt
            break
        }
    }
    attempted <- which(!vapply(fits, is.null, logical(1)))
    fit_audit <- do.call(rbind, fit_audits[attempted])
    if (is.na(selected_attempt)) {
        selected_attempt <- attempted[[which.min(
            fit_audit$tail5_median_rel
        )]]
    }
    fit <- fits[[selected_attempt]]
    basic_converged <- isTRUE(
        fit_audit$basic_converged[fit_audit$attempt == selected_attempt]
    )
    if (isTRUE(require_basic_convergence) && !basic_converged) {
        stop(
            "Ising GEM did not reach basic convergence after ",
            length(attempted), " attempts; best tail-5 median rel=",
            signif(min(fit_audit$tail5_median_rel), 4),
            ", best tail-5 max rel=",
            signif(fit_audit$tail5_max_rel[
                which.min(fit_audit$tail5_median_rel)
            ], 4)
        )
    }
    gamma_fit <- hmrf_estimate_gamma_base(
        nb = nb,
        baseline3d = background,
        beta = fit$beta,
        h = fit$h,
        p = fit$p,
        mu = fit$mu,
        sig2 = fit$sig2,
        burnin = as.integer(burnin_lis),
        sweeps = as.integer(sweep_lis),
        thin = 1L,
        init_prob_theta = 0.5,
        seed = as.integer(seed + 10000L),
        n_chains = as.integer(n_chains),
        random_scan = TRUE,
        init_mode = 3L,
        f0_absmax2 = TRUE,
        f1_absmax01 = TRUE,
        clamp_eps = 1e-10
    )
    fit_trace <- fit$trace
    gem_iterations <- length(fit_trace)
    last_rel <- if (gem_iterations) {
        as.numeric(fit_trace[[gem_iterations]]$rel)
    } else {
        NA_real_
    }
    gem_converged <- is.finite(last_rel) && last_rel < 1e-4
    score_x <- hmrf_plis_reweight_absmax(
        nb, x, background, gamma_fit$gamma_full,
        fit$p, fit$mu, fit$sig2, 1e-10
    )
    score_til <- hmrf_plis_reweight_absmax(
        nb, x_til, background, gamma_fit$gamma_full,
        fit$p, fit$mu, fit$sig2, 1e-10
    )
    score_x[!mask] <- 1
    score_til[!mask] <- 1
    log_x <- log(pmax(score_x, 1e-300))
    log_til <- log(pmax(score_til, 1e-300))
    list(
        R = score_x,
        R_til = score_til,
        log_R = log_x,
        log_R_til = log_til,
        background = background,
        model = "ising_absmax_plis_adaptation",
        fit = list(
            beta = fit$beta,
            h = fit$h,
            p = fit$p,
            mu = fit$mu,
            sig2 = fit$sig2,
            gamma_min = gamma_fit$gamma_min,
            gamma_max = gamma_fit$gamma_max,
            gamma_mean = mean(gamma_fit$gamma_mask),
            gamma_chain_range = diff(range(gamma_fit$chain_means)),
            gamma_chain_means = as.numeric(gamma_fit$chain_means),
            gem_iterations = as.integer(gem_iterations),
            gem_last_rel = last_rel,
            gem_converged = gem_converged,
            gem_basic_converged = basic_converged,
            gem_selected_attempt = as.integer(selected_attempt),
            gem_fit_attempts_used = nrow(fit_audit),
            gem_tail5_median_rel = fit_audit$tail5_median_rel[
                fit_audit$attempt == selected_attempt
            ],
            gem_tail5_max_rel = fit_audit$tail5_max_rel[
                fit_audit$attempt == selected_attempt
            ],
            gem_attempt_audit = fit_audit,
            iter_max = as.integer(iter_max),
            sweep_b = as.integer(sweep_b),
            sweep_r = as.integer(sweep_r),
            burnin_lis = as.integer(burnin_lis),
            sweep_lis = as.integer(sweep_lis),
            n_chains = as.integer(n_chains),
            fixing_beta = isTRUE(fixing_beta),
            elapsed_seconds = proc.time()[["elapsed"]] - fit_started,
            candidate_emission = "ordinary f0/f1",
            baseline_emission = "abs-max g0/g1"
        )
    )
}
