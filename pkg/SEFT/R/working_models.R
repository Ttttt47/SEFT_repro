.validate_pair <- function(x, x_til, mask) {
    if (!is.array(x) || length(dim(x)) != 3L || !identical(dim(x), dim(x_til))) {
        stop("Working-model inputs must be three-dimensional arrays with identical dimensions.", call. = FALSE)
    }
    if (is.null(mask)) mask <- array(TRUE, dim = dim(x))
    if (!identical(dim(x), dim(mask))) stop("Working-model mask dimensions differ from the inputs.", call. = FALSE)
    mask <- array(as.logical(mask), dim = dim(x))
    mask[is.na(mask)] <- FALSE
    if (!any(mask)) stop("Working-model mask is empty.", call. = FALSE)
    if (any(!is.finite(x[mask])) || any(!is.finite(x_til[mask]))) stop("Working-model inputs must be finite inside the mask.", call. = FALSE)
    mask
}

.absmax_background <- function(x, x_til, mask) {
    mask <- .validate_pair(x, x_til, mask)
    output <- array(0, dim = dim(x))
    choose_x <- abs(x) > abs(x_til)
    choose_til <- abs(x_til) > abs(x)
    ties <- !(choose_x | choose_til)
    output[choose_x & mask] <- x[choose_x & mask]
    output[choose_til & mask] <- x_til[choose_til & mask]
    output[ties & mask] <- pmax(x[ties & mask], x_til[ties & mask])
    output
}

.model_crop <- function(x, x_til, mask, seed) {
    coordinates <- which(mask, arr.ind = TRUE)
    lower <- apply(coordinates, 2L, min)
    upper <- apply(coordinates, 2L, max)
    extent <- as.integer(upper - lower + 1L)
    work_dim <- as.integer(ceiling(extent / 4L) * 4L)
    relative <- sweep(coordinates, 2L, lower - 1L, "-")
    work_indices <- relative[, 1L] + (relative[, 2L] - 1L) * work_dim[[1L]] + (relative[, 3L] - 1L) * work_dim[[1L]] * work_dim[[2L]]
    set.seed(as.integer(seed))
    x_work <- array(stats::rnorm(prod(work_dim)), dim = work_dim)
    x_til_work <- array(stats::rnorm(prod(work_dim)), dim = work_dim)
    x_work[work_indices] <- x[mask]
    x_til_work[work_indices] <- x_til[mask]
    list(
        x = x_work, x_til = x_til_work, mask_indices = which(mask),
        work_indices = as.integer(work_indices), lower = as.integer(lower),
        upper = as.integer(upper), work_dim = work_dim,
        padding_voxels = prod(work_dim) - sum(mask)
    )
}

.expand_scores <- function(scores, crop, full_dim, model) {
    expand <- function(values, outside) {
        output <- array(outside, dim = full_dim)
        output[crop$mask_indices] <- as.numeric(values[crop$work_indices])
        output
    }
    scores$fit$crop_lower_1based <- crop$lower
    scores$fit$crop_upper_1based <- crop$upper
    scores$fit$working_dimensions <- crop$work_dim
    scores$fit$padding_voxels <- crop$padding_voxels
    list(
        R = expand(scores$R, 1), R_til = expand(scores$R_til, 1),
        log_R = expand(scores$log_R, 0), log_R_til = expand(scores$log_R_til, 0),
        background = expand(scores$background, 0), fit = scores$fit,
        model_identifier = paste0("seft_", model)
    )
}

.run_working_model <- function(x, x_til, mask, model, seed, density_bandwidth, bandwidth, neighbor_range, lambda, score_clip, verbose) {
    mask <- .validate_pair(x, x_til, mask)
    if (model == "co") {
        p_x <- 2 * stats::pnorm(-abs(x))
        p_til <- 2 * stats::pnorm(-abs(x_til))
        p_x[!mask] <- 1
        p_til[!mask] <- 1
        scores <- cal_CLAW_scores_3d(
            x, x_til, p_x, p_til, lambda = lambda, h = density_bandwidth,
            bandwidth = bandwidth, neighbor_range = as.integer(neighbor_range), c = score_clip
        )
        scores$R[!mask] <- 1
        scores$R_til[!mask] <- 1
        scores$log_R <- log(pmax(scores$R, .Machine$double.xmin))
        scores$log_R_til <- log(pmax(scores$R_til, .Machine$double.xmin))
        scores$background <- .absmax_background(x, x_til, mask)
        scores$fit <- list(
            density_bandwidth = density_bandwidth, spatial_bandwidth = bandwidth,
            neighbor_range = as.integer(neighbor_range), lambda = lambda,
            score_clip = score_clip
        )
        scores$model_identifier <- "seft_co"
        return(scores)
    }
    crop <- .model_crop(x, x_til, mask, as.integer(seed))
    full_mask <- array(TRUE, dim = crop$work_dim)
    scores <- switch(
        model,
        ising = .ising_scores(
            crop$x, crop$x_til, full_mask,
            seed = as.integer(seed + 53L), verbose = verbose
        ),
        `fdr-smoothing` = .fdr_smoothing_scores(
            crop$x, crop$x_til, full_mask, seed = as.integer(seed + 1L)
        ),
        deepfdr = .ml_scores(
            crop$x, crop$x_til, full_mask, model = "deepfdr", seed = 0L
        ),
        fchmrf = .ml_scores(
            crop$x, crop$x_til, full_mask, model = "fchmrf", seed = as.integer(seed + 37L)
        )
    )
    if (any(!is.finite(scores$log_R)) || any(!is.finite(scores$log_R_til))) stop(model, " working model returned non-finite log scores.", call. = FALSE)
    .expand_scores(scores, crop, dim(x), model)
}

.ising_scores <- function(
    x, x_til, mask, seed = 1L, iter_max = 20L, sweep_b = 1L,
    sweep_r = 2L, burnin_lis = 20L, sweep_lis = 50L,
    n_chains = 2L, mixture_components = 1L, verbose = FALSE
) {
    mask <- .validate_pair(x, x_til, mask)
    background <- .absmax_background(x, x_til, mask)
    neighbours <- hmrf_build_neighbors_6n(mask)
    components <- as.integer(mixture_components)
    started <- proc.time()[["elapsed"]]
    fit <- hmrf_gem_fit(
        nb = neighbours, x3d = background, L = components,
        iter_max = as.integer(iter_max), sweep_b = as.integer(sweep_b),
        sweep_r = as.integer(sweep_r), a = 1, b = 2, alpha = 1e-3,
        stpmax = 1, max_backtrack = 10L, tol = 1e-4,
        beta_init = 0.5, h_init = -2, p_init = rep(1 / components, components),
        mu_init = rep(1, components), sig2_init = rep(1, components),
        init_prob_theta = 0.8, seed = as.integer(seed), verbose = isTRUE(verbose),
        f0_absmax2 = TRUE, f1_absmax01 = TRUE, fixing_beta = FALSE
    )
    gamma_fit <- hmrf_estimate_gamma_base(
        nb = neighbours, baseline3d = background, beta = fit$beta, h = fit$h,
        p = fit$p, mu = fit$mu, sig2 = fit$sig2,
        burnin = as.integer(burnin_lis), sweeps = as.integer(sweep_lis), thin = 1L,
        init_prob_theta = 0.5, seed = as.integer(seed + 10000L),
        n_chains = as.integer(n_chains), random_scan = TRUE, init_mode = 3L,
        f0_absmax2 = TRUE, f1_absmax01 = TRUE, clamp_eps = 1e-10
    )
    score_x <- hmrf_plis_reweight_absmax(neighbours, x, background, gamma_fit$gamma_full, fit$p, fit$mu, fit$sig2, 1e-10)
    score_til <- hmrf_plis_reweight_absmax(neighbours, x_til, background, gamma_fit$gamma_full, fit$p, fit$mu, fit$sig2, 1e-10)
    score_x[!mask] <- 1
    score_til[!mask] <- 1
    trace_rel <- vapply(fit$trace, function(value) as.numeric(value$rel), numeric(1))
    list(
        R = score_x, R_til = score_til,
        log_R = log(pmax(score_x, 1e-300)),
        log_R_til = log(pmax(score_til, 1e-300)),
        background = background,
        fit = list(
            beta = fit$beta, h = fit$h, p = fit$p, mu = fit$mu,
            sig2 = fit$sig2, gamma_min = gamma_fit$gamma_min,
            gamma_max = gamma_fit$gamma_max,
            gamma_mean = mean(gamma_fit$gamma_mask),
            gamma_chain_means = as.numeric(gamma_fit$chain_means),
            gem_iterations = length(trace_rel),
            gem_last_rel = if (length(trace_rel)) tail(trace_rel, 1L) else NA_real_,
            iter_max = as.integer(iter_max), burnin_lis = as.integer(burnin_lis),
            sweep_lis = as.integer(sweep_lis), n_chains = as.integer(n_chains),
            elapsed_seconds = proc.time()[["elapsed"]] - started,
            candidate_emission = "ordinary f0/f1",
            baseline_emission = "absolute-max g0/g1"
        )
    )
}

.wavelet_denoise <- function(z, mask) {
    if (!requireNamespace("waveslim", quietly = TRUE)) {
        stop("denoise='wavelet' requires the optional package 'waveslim'.", call. = FALSE)
    }
    original_dim <- dim(z)
    expanded_dim <- rep(2^ceiling(log2(max(original_dim))), 3L)
    expanded <- array(0, dim = expanded_dim)
    expanded[seq_len(original_dim[[1L]]), seq_len(original_dim[[2L]]), seq_len(original_dim[[3L]])] <- z
    transform <- waveslim::dwt.3d(expanded, wf = "d8", J = min(7L, as.integer(log2(min(expanded_dim)))))
    threshold_scale <- sqrt(2 * log(max(2L, sum(expanded != 0))))
    for (name in names(transform)) {
        level <- suppressWarnings(as.integer(sub(".*([0-9]+)$", "\\1", name)))
        if (!is.na(level) && level == 1L) {
            nonzero <- transform[[name]][transform[[name]] != 0]
            if (length(nonzero)) {
                threshold <- threshold_scale * stats::median(abs(nonzero)) / 0.6745
                transform[[name]] <- ifelse(abs(transform[[name]]) > threshold, transform[[name]], 0)
            }
        }
    }
    output <- waveslim::idwt.3d(transform)
    output <- output[seq_len(original_dim[[1L]]), seq_len(original_dim[[2L]]), seq_len(original_dim[[3L]])]
    output <- array(output, dim = original_dim)
    output[!mask] <- 0
    output
}
