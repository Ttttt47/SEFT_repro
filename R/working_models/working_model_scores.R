# PLIS-type working-model scores for SEFT.
#
# Both public functions return a list containing R and R_til.  These arrays are
# local indices of significance (posterior null probabilities), so smaller
# values indicate stronger evidence against the point null.

.wm_load_cpp <- function() {
    if (exists("wm_predictive_recursion_cpp", mode = "function", inherits = TRUE)) {
        return(invisible(TRUE))
    }
    environment_bin <- normalizePath(
        file.path(R.home(), "..", "..", "bin"),
        mustWork = FALSE
    )
    if (dir.exists(environment_bin)) {
        Sys.setenv(PATH = paste(
            environment_bin,
            Sys.getenv("PATH"),
            sep = .Platform$path.sep
        ))
    }
    configured <- Sys.getenv("SEFT_WORKING_MODEL_CPP", unset = "")
    candidates <- unique(c(
        configured,
        file.path(getwd(), "src", "working_models", "working_model_scores.cpp"),
        file.path(getwd(), "src", "working_model_scores.cpp"),
        file.path(getwd(), "..", "src", "working_model_scores.cpp")
    ))
    candidates <- candidates[nzchar(candidates) & file.exists(candidates)]
    if (length(candidates) == 0L) {
        stop(
            "Cannot locate src/working_model_scores.cpp. Set ",
            "SEFT_WORKING_MODEL_CPP to its absolute path."
        )
    }
    suppressPackageStartupMessages(requireNamespace("Rcpp", quietly = TRUE))
    Rcpp::sourceCpp(normalizePath(candidates[[1]]), rebuild = FALSE)
    invisible(TRUE)
}

.wm_load_cpp()

.wm_expit <- function(x) {
    x <- pmax(pmin(x, 35), -35)
    1 / (1 + exp(-x))
}

.wm_logit <- function(x) {
    x <- pmax(pmin(x, 1 - 1e-10), 1e-10)
    log(x) - log1p(-x)
}

.wm_validate_pair <- function(x, x_til, mask) {
    if (!is.array(x) || length(dim(x)) != 3L) {
        stop("x must be a three-dimensional array.")
    }
    if (!identical(dim(x), dim(x_til))) {
        stop("x and x_til must have identical dimensions.")
    }
    if (is.null(mask)) {
        mask <- array(TRUE, dim = dim(x))
    }
    if (!identical(dim(x), dim(mask))) {
        stop("mask must have the same dimensions as x.")
    }
    mask <- array(as.logical(mask), dim = dim(x))
    mask[is.na(mask)] <- FALSE
    if (any(!is.finite(x[mask])) || any(!is.finite(x_til[mask]))) {
        stop("Non-finite values are not allowed inside mask.")
    }
    mask
}

#' Construct the shared, swap-invariant PLIS background.
#'
#' The entry with larger absolute magnitude is retained.  Exact absolute-value
#' ties are resolved with numeric max, which makes the construction invariant
#' even for discrete or rounded inputs.
make_plis_background <- function(x, x_til, mask = NULL) {
    mask <- .wm_validate_pair(x, x_til, mask)
    out <- array(0, dim = dim(x))
    choose_x <- abs(x) > abs(x_til)
    choose_til <- abs(x_til) > abs(x)
    ties <- !(choose_x | choose_til)
    out[choose_x & mask] <- x[choose_x & mask]
    out[choose_til & mask] <- x_til[choose_til & mask]
    out[ties & mask] <- pmax(x[ties & mask], x_til[ties & mask])
    out
}

.wm_periodic_gaussian_spectrum <- function(dims, fwhm) {
    if (!is.finite(fwhm) || fwhm <= 0) {
        stop("fwhm must be positive.")
    }
    ell <- fwhm / sqrt(8 * log(2))
    axes <- lapply(dims, function(n) {
        idx <- 0:(n - 1)
        pmin(idx, n - idx)
    })
    cov_arr <- array(0, dim = dims)
    cov_arr[] <- exp(
        -0.5 * (
            outer(axes[[1]]^2, rep(1, dims[2] * dims[3])) +
            rep(outer(axes[[2]]^2, rep(1, dims[3])), each = dims[1]) +
            rep(axes[[3]]^2, each = dims[1] * dims[2])
        ) / ell^2
    )
    eig <- pmax(Re(fft(cov_arr)), 0)
    eig / mean(eig)
}

.wm_spatial_fdr_fit <- function(
    background,
    mask,
    signal_fwhm_grid,
    error_fwhm,
    noise_variance,
    correlated_noise_grid,
    signal_variance_grid
) {
    dims <- dim(background)
    y <- background
    mu0 <- mean(y[mask])
    y[!mask] <- mu0
    yc <- y - mu0
    n <- length(yc)
    periodogram <- Mod(fft(yc))^2 / n

    error_eig <- .wm_periodic_gaussian_spectrum(dims, error_fwhm)
    best <- NULL
    best_nll <- Inf

    for (signal_fwhm in signal_fwhm_grid) {
        signal_eig <- .wm_periodic_gaussian_spectrum(dims, signal_fwhm)
        for (r_noise in correlated_noise_grid) {
            error_spec <- noise_variance * ((1 - r_noise) + r_noise * error_eig)
            for (tau2 in signal_variance_grid) {
                signal_spec <- tau2 * signal_eig
                total_spec <- pmax(signal_spec + error_spec, 1e-10)
                nll <- 0.5 * sum(log(total_spec) + periodogram / total_spec)
                if (is.finite(nll) && nll < best_nll) {
                    best_nll <- nll
                    best <- list(
                        mu0 = mu0,
                        signal_fwhm = signal_fwhm,
                        error_fwhm = error_fwhm,
                        noise_variance = noise_variance,
                        correlated_noise = r_noise,
                        signal_variance = tau2,
                        signal_spec = signal_spec,
                        error_spec = error_spec
                    )
                }
            }
        }
    }
    if (is.null(best)) {
        stop("Spatial-FDR spectral parameter fit failed.")
    }

    total_spec <- pmax(best$signal_spec + best$error_spec, 1e-10)
    gain <- best$signal_spec / total_spec
    post_spec <- best$signal_spec * best$error_spec / total_spec
    post_mean <- best$mu0 + Re(fft(gain * fft(yc), inverse = TRUE)) / n
    post_var <- max(mean(post_spec), 1e-8)

    best$gain_diagonal <- mean(gain)
    best$posterior_variance <- post_var
    best$posterior_mean <- post_mean
    best$nll <- best_nll
    best
}

#' Sun et al. (2015) spatial-FDR score in PLIS form.
#'
#' A stationary GRF signal plus correlated-GRF error working model is fitted to
#' the shared background by Whittle empirical Bayes.  The marginal posterior
#' null probability P(mu_i <= null_boundary | data) is then reweighted exactly
#' for replacing background_i by x_i or x_til_i.
cal_spatial_fdr_scores_3d_whittle_approx <- function(
    x,
    x_til,
    mask = NULL,
    null_boundary = 0,
    signal_fwhm_grid = c(6, 10, 14),
    error_fwhm = 4,
    noise_variance = 1,
    correlated_noise_grid = c(0, 0.5, 0.8, 0.95, 0.995),
    signal_variance_grid = c(0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16),
    score_eps = 1e-10
) {
    mask <- .wm_validate_pair(x, x_til, mask)
    background <- make_plis_background(x, x_til, mask)
    fit <- .wm_spatial_fdr_fit(
        background = background,
        mask = mask,
        signal_fwhm_grid = signal_fwhm_grid,
        error_fwhm = error_fwhm,
        noise_variance = noise_variance,
        correlated_noise_grid = correlated_noise_grid,
        signal_variance_grid = signal_variance_grid
    )

    log_score_candidate <- function(candidate) {
        post_mean <- fit$posterior_mean +
            fit$gain_diagonal * (candidate - background)
        out <- pnorm(
            null_boundary,
            mean = post_mean,
            sd = sqrt(fit$posterior_variance),
            log.p = TRUE
        )
        out[!mask] <- 0
        array(out, dim = dim(x))
    }

    log_R <- log_score_candidate(x)
    log_R_til <- log_score_candidate(x_til)
    diagnostics <- fit
    diagnostics$posterior_mean <- NULL
    diagnostics$signal_spec <- NULL
    diagnostics$error_spec <- NULL
    list(
        R = exp(log_R),
        R_til = exp(log_R_til),
        log_R = log_R,
        log_R_til = log_R_til,
        background = background,
        fit = diagnostics,
        model = "spatial_fdr_whittle_approx"
    )
}

#' Refuse to label the local Whittle approximation as official Spatial FDR.
cal_spatial_fdr_scores_3d <- function(...) {
    stop(
        "No verified official Sun et al. (2015) backend is available. The ",
        "authors' published software URL is no longer accessible, and the ",
        "local FFT/Whittle model is materially different from their MCMC ",
        "hierarchical GP. Call cal_spatial_fdr_scores_3d_whittle_approx() ",
        "only when that distinction is explicit."
    )
}

.wm_trapezoid <- function(x, y) {
    sum(diff(x) * (head(y, -1) + tail(y, -1)) / 2)
}

.wm_predictive_recursion_r <- function(
    z,
    grid,
    sweeps = 3L,
    seed = 1L,
    mu0 = 0,
    sigma0 = 1,
    decay = -0.67
) {
    set.seed(seed)
    order_idx <- unlist(
        replicate(sweeps, sample.int(length(z)), simplify = FALSE),
        use.names = FALSE
    )
    theta_subdensity <- rep(1 / length(grid), length(grid))
    pi0 <- 1

    for (iteration in seq_along(order_idx)) {
        value <- z[order_idx[[iteration]]]
        weight <- (3 + iteration)^decay
        joint1 <- dnorm(grid, mean = value - mu0, sd = sigma0) *
            theta_subdensity
        m0 <- pi0 * dnorm(value, mean = mu0, sd = sigma0)
        m1 <- .wm_trapezoid(grid, joint1)
        mixture <- max(m0 + m1, 1e-300)
        pi0 <- (1 - weight) * pi0 + weight * m0 / mixture
        theta_subdensity <- (1 - weight) * theta_subdensity +
            weight * joint1 / mixture
    }

    signal_density <- vapply(grid, function(value) {
        joint1 <- dnorm(grid, mean = value - mu0, sd = sigma0) *
            theta_subdensity
        .wm_trapezoid(grid, joint1) / max(1 - pi0, 1e-8)
    }, numeric(1))
    signal_density <- pmax(signal_density, 1e-12)
    signal_density <- signal_density /
        .wm_trapezoid(grid, signal_density)
    list(grid = grid, density = signal_density, pi0 = pi0)
}

.wm_predictive_recursion <- function(
    z,
    grid,
    sweeps = 3L,
    seed = 1L,
    mu0 = 0,
    sigma0 = 1,
    decay = -0.67
) {
    set.seed(seed)
    order_idx <- unlist(
        replicate(sweeps, sample.int(length(z)), simplify = FALSE),
        use.names = FALSE
    )
    wm_predictive_recursion_cpp(
        z = as.numeric(z),
        grid = as.numeric(grid),
        order_idx = as.integer(order_idx),
        mu0 = mu0,
        sigma0 = sigma0,
        decay = decay
    )
}

.wm_gradient_3d <- function(beta) {
    dims <- dim(beta)
    gx <- gy <- gz <- array(0, dim = dims)
    if (dims[1] > 1) {
        gx[1:(dims[1] - 1), , ] <-
            beta[2:dims[1], , ] - beta[1:(dims[1] - 1), , ]
    }
    if (dims[2] > 1) {
        gy[, 1:(dims[2] - 1), ] <-
            beta[, 2:dims[2], ] - beta[, 1:(dims[2] - 1), ]
    }
    if (dims[3] > 1) {
        gz[, , 1:(dims[3] - 1)] <-
            beta[, , 2:dims[3]] - beta[, , 1:(dims[3] - 1)]
    }
    list(x = gx, y = gy, z = gz)
}

.wm_gradient_adjoint_3d <- function(p) {
    dims <- dim(p$x)
    out <- array(0, dim = dims)
    if (dims[1] > 1) {
        out[1:(dims[1] - 1), , ] <-
            out[1:(dims[1] - 1), , ] - p$x[1:(dims[1] - 1), , ]
        out[2:dims[1], , ] <-
            out[2:dims[1], , ] + p$x[1:(dims[1] - 1), , ]
    }
    if (dims[2] > 1) {
        out[, 1:(dims[2] - 1), ] <-
            out[, 1:(dims[2] - 1), ] - p$y[, 1:(dims[2] - 1), ]
        out[, 2:dims[2], ] <-
            out[, 2:dims[2], ] + p$y[, 1:(dims[2] - 1), ]
    }
    if (dims[3] > 1) {
        out[, , 1:(dims[3] - 1)] <-
            out[, , 1:(dims[3] - 1)] - p$z[, , 1:(dims[3] - 1)]
        out[, , 2:dims[3]] <-
            out[, , 2:dims[3]] + p$z[, , 1:(dims[3] - 1)]
    }
    out
}

.wm_tv_logistic_mstep_r <- function(
    beta,
    posterior_signal,
    lambda,
    mask,
    iterations = 120L,
    tolerance = 1e-5
) {
    p <- list(
        x = array(0, dim = dim(beta)),
        y = array(0, dim = dim(beta)),
        z = array(0, dim = dim(beta))
    )
    beta_bar <- beta
    tau <- 0.9
    sigma <- 0.08

    for (iteration in seq_len(iterations)) {
        grad_bar <- .wm_gradient_3d(beta_bar)
        p$x <- pmax(pmin(p$x + sigma * grad_bar$x, lambda), -lambda)
        p$y <- pmax(pmin(p$y + sigma * grad_bar$y, lambda), -lambda)
        p$z <- pmax(pmin(p$z + sigma * grad_bar$z, lambda), -lambda)

        beta_old <- beta
        likelihood_gradient <- .wm_expit(beta) - posterior_signal
        likelihood_gradient[!mask] <- 0
        beta <- beta - tau * (
            likelihood_gradient + .wm_gradient_adjoint_3d(p)
        )
        beta <- pmax(pmin(beta, 12), -12)
        beta[!mask] <- 0
        beta_bar <- 2 * beta - beta_old

        if (iteration %% 10L == 0L &&
                mean(abs(beta - beta_old)[mask]) < tolerance) {
            break
        }
    }
    beta
}

.wm_tv_logistic_mstep <- function(
    beta,
    posterior_signal,
    lambda,
    mask,
    iterations = 120L,
    tolerance = 1e-5
) {
    wm_tv_logistic_mstep_cpp(
        beta = beta,
        posterior_signal = posterior_signal,
        dims = as.integer(dim(beta)),
        lambda = lambda,
        mask = as.logical(mask),
        iterations = as.integer(iterations),
        tolerance = tolerance
    )$beta
}

.wm_fdr_smoothing_fit_native_approx <- function(
    background,
    mask,
    lambda,
    em_iterations,
    tv_iterations,
    pr_sweeps,
    density_sample_size,
    density_grid_size,
    seed,
    tolerance
) {
    z_all <- background[mask]
    set.seed(seed)
    if (length(z_all) > density_sample_size) {
        z_fit <- sample(z_all, density_sample_size)
    } else {
        z_fit <- z_all
    }
    bound <- max(6, min(12, max(abs(stats::quantile(
        z_fit, c(0.001, 0.999), names = FALSE
    ))) + 1))
    grid <- seq(-bound, bound, length.out = density_grid_size)
    pr <- .wm_predictive_recursion(
        z_fit,
        grid,
        sweeps = pr_sweeps,
        seed = seed
    )
    prior_signal <- pmin(pmax(1 - pr$pi0, 0.02), 0.5)
    beta <- array(.wm_logit(prior_signal), dim = dim(background))
    beta[!mask] <- 0
    previous_objective <- Inf
    objective_trace <- numeric()

    for (em_iteration in seq_len(em_iterations)) {
        posterior_fit <- wm_fdr_posterior_objective_cpp(
            values = background,
            beta = beta,
            grid = pr$grid,
            density = pr$density,
            mask = as.logical(mask)
        )
        posterior_signal <- posterior_fit$posterior_signal
        beta <- .wm_tv_logistic_mstep(
            beta,
            posterior_signal,
            lambda = lambda,
            mask = mask,
            iterations = tv_iterations,
            tolerance = tolerance
        )
        posterior_fit <- wm_fdr_posterior_objective_cpp(
            values = background,
            beta = beta,
            grid = pr$grid,
            density = pr$density,
            mask = as.logical(mask)
        )
        objective <- posterior_fit$objective
        objective_trace <- c(objective_trace, objective)
        if (is.finite(previous_objective) &&
                abs(previous_objective - objective) /
                    max(1, abs(previous_objective)) < tolerance) {
            break
        }
        previous_objective <- objective
    }

    posterior_signal <- wm_fdr_posterior_objective_cpp(
        values = background,
        beta = beta,
        grid = pr$grid,
        density = pr$density,
        mask = as.logical(mask)
    )$posterior_signal
    list(
        beta = beta,
        posterior_signal = posterior_signal,
        predictive_recursion = pr,
        objective_trace = objective_trace,
        lambda = lambda
    )
}

#' Tansey et al. (2018) FDR-smoothing score in PLIS form.
#'
#' The implementation follows the paper's two-stage structure: predictive
#' recursion estimates the alternative density and graph total variation
#' smooths the site-specific prior log odds.  The fitted background posterior
#' is then reweighted for each member of the real/mirror pair.
cal_fdr_smoothing_scores_3d_native_approx <- function(
    x,
    x_til,
    mask = NULL,
    lambda = 0.1,
    em_iterations = 8L,
    tv_iterations = 120L,
    pr_sweeps = 3L,
    density_sample_size = 12000L,
    density_grid_size = 121L,
    seed = 1L,
    tolerance = 1e-4,
    score_eps = 1e-10
) {
    mask <- .wm_validate_pair(x, x_til, mask)
    background <- make_plis_background(x, x_til, mask)
    fit <- .wm_fdr_smoothing_fit_native_approx(
        background = background,
        mask = mask,
        lambda = lambda,
        em_iterations = em_iterations,
        tv_iterations = tv_iterations,
        pr_sweeps = pr_sweeps,
        density_sample_size = density_sample_size,
        density_grid_size = density_grid_size,
        seed = seed,
        tolerance = tolerance
    )

    log_score_candidate <- function(candidate) {
        wm_log_score_candidates_cpp(
            candidate = candidate,
            beta = fit$beta,
            grid = fit$predictive_recursion$grid,
            density = fit$predictive_recursion$density,
            mask = as.logical(mask)
        )
    }

    log_R <- log_score_candidate(x)
    log_R_til <- log_score_candidate(x_til)
    list(
        R = exp(log_R),
        R_til = exp(log_R_til),
        log_R = log_R,
        log_R_til = log_R_til,
        background = background,
        fit = list(
            lambda = fit$lambda,
            pi0 = fit$predictive_recursion$pi0,
            density_grid = fit$predictive_recursion$grid,
            signal_density = fit$predictive_recursion$density,
            objective_trace = fit$objective_trace,
            prior_signal_range = range(.wm_expit(fit$beta)[mask]),
            posterior_signal_range = range(fit$posterior_signal[mask])
        ),
        model = "fdr_smoothing_native_approx"
    )
}

.wm_resolve_fdrs_official_script <- function() {
    configured <- Sys.getenv("SEFT_FDRS_OFFICIAL_SCRIPT", unset = "")
    candidates <- unique(c(
        configured,
        file.path(
            getwd(), "python", "working_models",
            "fdr_smoothing_official.py"
        ),
        file.path(getwd(), "python", "fdr_smoothing_official.py"),
        file.path(getwd(), "..", "python", "fdr_smoothing_official.py")
    ))
    candidates <- candidates[nzchar(candidates) & file.exists(candidates)]
    if (!length(candidates)) {
        stop(
            "Cannot locate python/fdr_smoothing_official.py. Set ",
            "SEFT_FDRS_OFFICIAL_SCRIPT."
        )
    }
    normalizePath(candidates[[1]], mustWork = TRUE)
}

.wm_resolve_fdrs_absmax_script <- function() {
    configured <- Sys.getenv("SEFT_FDRS_ABSMAX_SCRIPT", unset = "")
    candidates <- unique(c(
        configured,
        file.path(
            getwd(), "python", "working_models",
            "fdr_smoothing_absmax.py"
        ),
        file.path(getwd(), "python", "fdr_smoothing_absmax.py"),
        file.path(getwd(), "..", "python", "fdr_smoothing_absmax.py")
    ))
    candidates <- candidates[nzchar(candidates) & file.exists(candidates)]
    if (!length(candidates)) {
        stop(
            "Cannot locate python/fdr_smoothing_absmax.py. Set ",
            "SEFT_FDRS_ABSMAX_SCRIPT."
        )
    }
    normalizePath(candidates[[1]], mustWork = TRUE)
}

.wm_resolve_fdrs_python <- function() {
    configured <- Sys.getenv("SEFT_FDRS_PYTHON", unset = "")
    candidates <- unique(c(configured, Sys.which("python3")))
    candidates <- candidates[nzchar(candidates) & file.exists(candidates)]
    if (!length(candidates)) {
        stop("Cannot locate the official smoothfdr Python environment.")
    }
    normalizePath(candidates[[1]], mustWork = TRUE)
}

.wm_write_binary_volume_fdrs <- function(x, path) {
    connection <- file(path, open = "wb")
    on.exit(close(connection), add = TRUE)
    writeBin(as.double(x), connection, size = 8L, endian = "little")
}

#' Official Tansey et al. FDR-smoothing score in PLIS form.
#'
#' The shared background is fitted by the authors' smoothfdr 0.9.5
#' ``easy.smooth_fdr`` entry point. Its repository defaults are retained:
#' empirical null, 220-point predictive-recursion grid, 10 sweeps, 30-lambda
#' solution path, BIC selection, greedy trails, and the pygfl TrailSolver.
cal_fdr_smoothing_scores_3d <- function(
    x,
    x_til,
    mask = NULL,
    seed = 1L,
    num_sweeps = 10L,
    fdr_level_fit_only = 0.1,
    verbose = 0L,
    keep_work_dir = FALSE
) {
    mask <- .wm_validate_pair(x, x_til, mask)
    if (!all(mask)) {
        stop(
            "Official smoothfdr mode currently requires a complete rectangular ",
            "3-D lattice. Use cal_fdr_smoothing_scores_3d_native_approx() only ",
            "if an explicitly labelled approximation is acceptable."
        )
    }
    if (as.integer(num_sweeps) != 10L) {
        warning(
            "num_sweeps differs from the smoothfdr repository default (10). ",
            "Some later comparison papers used 20."
        )
    }
    background <- make_plis_background(x, x_til, mask)
    work_dir <- tempfile("seft_fdrs_official_")
    dir.create(work_dir, recursive = TRUE, showWarnings = FALSE)
    if (!keep_work_dir) {
        on.exit(unlink(work_dir, recursive = TRUE, force = TRUE), add = TRUE)
    }
    paths <- file.path(
        work_dir,
        c(
            "background.bin", "x.bin", "x_til.bin", "log_R.bin",
            "log_R_til.bin", "diagnostics.json", "python.log"
        )
    )
    names(paths) <- c(
        "background", "x", "x_til", "log_R", "log_R_til",
        "diagnostics", "log"
    )
    .wm_write_binary_volume_fdrs(background, paths[["background"]])
    .wm_write_binary_volume_fdrs(x, paths[["x"]])
    .wm_write_binary_volume_fdrs(x_til, paths[["x_til"]])
    arguments <- c(
        .wm_resolve_fdrs_official_script(),
        "--background", paths[["background"]],
        "--candidate-x", paths[["x"]],
        "--candidate-til", paths[["x_til"]],
        "--dims", paste(dim(x), collapse = ","),
        "--out-log-r", paths[["log_R"]],
        "--out-log-rtil", paths[["log_R_til"]],
        "--diagnostics", paths[["diagnostics"]],
        "--seed", as.character(as.integer(seed)),
        "--num-sweeps", as.character(as.integer(num_sweeps)),
        "--fdr-level", format(fdr_level_fit_only, scientific = FALSE),
        "--verbose", as.character(as.integer(verbose))
    )
    status <- system2(
        .wm_resolve_fdrs_python(),
        args = arguments,
        stdout = paths[["log"]],
        stderr = paths[["log"]]
    )
    if (!identical(status, 0L) ||
            !file.exists(paths[["log_R"]]) ||
            !file.exists(paths[["log_R_til"]])) {
        log_text <- if (file.exists(paths[["log"]])) {
            paste(tail(readLines(paths[["log"]], warn = FALSE), 60L),
                  collapse = "\n")
        } else {
            "(Python log was not created.)"
        }
        stop(sprintf(
            "Official smoothfdr backend failed (status %s):\n%s",
            status, log_text
        ))
    }
    read_score <- function(path) {
        connection <- file(path, open = "rb")
        on.exit(close(connection), add = TRUE)
        values <- readBin(
            connection, what = "double", n = length(x),
            size = 8L, endian = "little"
        )
        if (length(values) != length(x)) {
            stop("Official smoothfdr backend returned an incomplete volume.")
        }
        array(values, dim = dim(x))
    }
    log_R <- read_score(paths[["log_R"]])
    log_R_til <- read_score(paths[["log_R_til"]])
    diagnostics <- jsonlite::fromJSON(paths[["diagnostics"]])
    list(
        R = exp(log_R),
        R_til = exp(log_R_til),
        log_R = log_R,
        log_R_til = log_R_til,
        background = background,
        fit = diagnostics,
        model = "fdr_smoothing_official_0.9.5",
        work_dir = if (keep_work_dir) work_dir else NULL
    )
}

#' Abs-max PLIS adaptation of Tansey et al. FDR smoothing.
#'
#' The official graph, greedy trails, GFL solver, lambda path, and BIC rule
#' are retained.  Predictive recursion and the fitted two-groups likelihood
#' use the abs-max background emissions g0/g1, while point replacement uses
#' the ordinary candidate emissions f0/f1.
cal_fdr_smoothing_scores_3d_absmax <- function(
    x,
    x_til,
    mask = NULL,
    seed = 1L,
    num_sweeps = 10L,
    fdr_level_fit_only = 0.1,
    pr_table_size = 524288L,
    trail_cache = Sys.getenv("SEFT_FDRS_TRAIL_CACHE", unset = file.path(tempdir(), "seft_smoothfdr_trails")),
    trail_mode = c("greedy", "axis"),
    solver_mode = c("official", "persistent", "grid_primal_dual"),
    gfl_threads = 1L,
    grid_maxsteps = 400L,
    grid_converge = 1e-5,
    lambda_min = 0.2,
    lambda_max = 1.5,
    lambda_bins = 30L,
    verbose = 0L,
    keep_work_dir = FALSE
) {
    trail_mode <- match.arg(trail_mode)
    solver_mode <- match.arg(solver_mode)
    gfl_threads <- as.integer(gfl_threads)
    if (length(gfl_threads) != 1L || is.na(gfl_threads) ||
            gfl_threads < 1L) {
        stop("gfl_threads must be one positive integer.")
    }
    mask <- .wm_validate_pair(x, x_til, mask)
    if (!all(mask)) {
        stop(
            "The abs-max FDR-smoothing backend currently requires a complete ",
            "rectangular 3-D lattice."
        )
    }
    if (as.integer(num_sweeps) != 10L) {
        stop("The official smoothfdr repository default is num_sweeps=10.")
    }
    background <- make_plis_background(x, x_til, mask)
    work_dir <- tempfile("seft_fdrs_absmax_")
    dir.create(work_dir, recursive = TRUE, showWarnings = FALSE)
    if (!keep_work_dir) {
        on.exit(unlink(work_dir, recursive = TRUE, force = TRUE), add = TRUE)
    }
    paths <- file.path(
        work_dir,
        c(
            "background.bin", "x.bin", "x_til.bin", "log_R.bin",
            "log_R_til.bin", "diagnostics.json", "python.log"
        )
    )
    names(paths) <- c(
        "background", "x", "x_til", "log_R", "log_R_til",
        "diagnostics", "log"
    )
    .wm_write_binary_volume_fdrs(background, paths[["background"]])
    .wm_write_binary_volume_fdrs(x, paths[["x"]])
    .wm_write_binary_volume_fdrs(x_til, paths[["x_til"]])
    arguments <- c(
        .wm_resolve_fdrs_absmax_script(),
        "--background", paths[["background"]],
        "--candidate-x", paths[["x"]],
        "--candidate-til", paths[["x_til"]],
        "--dims", paste(dim(x), collapse = ","),
        "--out-log-r", paths[["log_R"]],
        "--out-log-rtil", paths[["log_R_til"]],
        "--diagnostics", paths[["diagnostics"]],
        "--seed", as.character(as.integer(seed)),
        "--num-sweeps", as.character(as.integer(num_sweeps)),
        "--fdr-level", format(fdr_level_fit_only, scientific = FALSE),
        "--pr-table-size", as.character(as.integer(pr_table_size)),
        "--trail-cache", normalizePath(trail_cache, mustWork = FALSE),
        "--trail-mode", trail_mode,
        "--solver-mode", solver_mode,
        "--gfl-threads", as.character(gfl_threads),
        "--grid-maxsteps", as.character(as.integer(grid_maxsteps)),
        "--grid-converge", format(grid_converge, scientific = TRUE),
        "--lambda-min", format(lambda_min, scientific = FALSE),
        "--lambda-max", format(lambda_max, scientific = FALSE),
        "--lambda-bins", as.character(as.integer(lambda_bins)),
        "--verbose", as.character(as.integer(verbose))
    )
    status <- system2(
        .wm_resolve_fdrs_python(),
        args = arguments,
        stdout = paths[["log"]],
        stderr = paths[["log"]]
    )
    if (!identical(status, 0L) ||
            !file.exists(paths[["log_R"]]) ||
            !file.exists(paths[["log_R_til"]])) {
        log_text <- if (file.exists(paths[["log"]])) {
            paste(tail(readLines(paths[["log"]], warn = FALSE), 80L),
                  collapse = "\n")
        } else {
            "(Python log was not created.)"
        }
        stop(sprintf(
            "Abs-max FDR-smoothing backend failed (status %s):\n%s",
            status, log_text
        ))
    }
    read_score <- function(path) {
        connection <- file(path, open = "rb")
        on.exit(close(connection), add = TRUE)
        values <- readBin(
            connection, what = "double", n = length(x),
            size = 8L, endian = "little"
        )
        if (length(values) != length(x)) {
            stop("Abs-max FDR-smoothing returned an incomplete volume.")
        }
        array(values, dim = dim(x))
    }
    log_R <- read_score(paths[["log_R"]])
    log_R_til <- read_score(paths[["log_R_til"]])
    diagnostics <- jsonlite::fromJSON(paths[["diagnostics"]])
    list(
        R = exp(log_R),
        R_til = exp(log_R_til),
        log_R = log_R,
        log_R_til = log_R_til,
        background = background,
        fit = diagnostics,
        model = if (trail_mode == "greedy" &&
                solver_mode == "official") {
            "fdr_smoothing_absmax_adaptation"
        } else if (trail_mode == "axis" &&
                solver_mode == "official") {
            "fdr_smoothing_absmax_axis_trails_adaptation"
        } else if (solver_mode == "persistent") {
            "fdr_smoothing_absmax_axis_persistent_adaptation"
        } else if (as.integer(lambda_bins) == 1L) {
            "fdr_smoothing_absmax_grid_fixed_lambda_adaptation"
        } else {
            "fdr_smoothing_absmax_grid_primal_dual_adaptation"
        },
        work_dir = if (keep_work_dir) work_dir else NULL
    )
}

#' Numerically verify the pairwise swap equivariance of a score backend.
audit_pairwise_exchangeability <- function(
    score_function,
    x,
    x_til,
    swap_fraction = 0.1,
    seed = 1L,
    tolerance = 1e-10,
    ...
) {
    set.seed(seed)
    swap <- array(
        runif(length(x)) < swap_fraction,
        dim = dim(x)
    )
    call_args <- c(list(x = x, x_til = x_til), list(...))
    if ("seed" %in% names(formals(score_function))) {
        call_args$seed <- seed
    }
    original <- do.call(score_function, call_args)
    x_swapped <- x
    x_til_swapped <- x_til
    x_swapped[swap] <- x_til[swap]
    x_til_swapped[swap] <- x[swap]
    call_args$x <- x_swapped
    call_args$x_til <- x_til_swapped
    swapped <- do.call(score_function, call_args)

    expected_R <- original$R
    expected_R_til <- original$R_til
    expected_R[swap] <- original$R_til[swap]
    expected_R_til[swap] <- original$R[swap]
    background_error <- max(abs(
        original$background - swapped$background
    ))
    score_error <- max(
        abs(expected_R - swapped$R),
        abs(expected_R_til - swapped$R_til)
    )
    list(
        passed = is.finite(background_error) &&
            is.finite(score_error) &&
            background_error <= tolerance &&
            score_error <= tolerance,
        background_max_abs_error = background_error,
        score_max_abs_error = score_error,
        tolerance = tolerance,
        swapped_points = sum(swap)
    )
}

