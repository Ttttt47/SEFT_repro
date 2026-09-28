# Unified dispatcher for the working models exposed by the SEFT command line.

seft_working_model_choices <- function() {
    c("co", "fdr-smoothing", "deepfdr", "fchmrf", "ising")
}

normalize_seft_working_model <- function(model) {
    aliases <- c(
        "covariate-adaptive" = "co",
        "co" = "co",
        "fdr-smoothing" = "fdr-smoothing",
        "fdrs" = "fdr-smoothing",
        "deepfdr" = "deepfdr",
        "fchmrf" = "fchmrf",
        "ising" = "ising"
    )
    normalized <- unname(aliases[[tolower(model)]])
    if (is.null(normalized)) {
        stop(
            "working-model must be one of: ",
            paste(seft_working_model_choices(), collapse = ", ")
        )
    }
    normalized
}

.seft_absmax_background <- function(x, x_til, mask) {
    background <- array(0, dim = dim(x))
    choose_x <- abs(x) > abs(x_til)
    choose_til <- abs(x_til) > abs(x)
    ties <- !(choose_x | choose_til)
    background[choose_x & mask] <- x[choose_x & mask]
    background[choose_til & mask] <- x_til[choose_til & mask]
    background[ties & mask] <- pmax(x[ties & mask], x_til[ties & mask])
    background
}

.seft_model_crop <- function(x, x_til, mask, seed) {
    coordinates <- which(mask, arr.ind = TRUE)
    if (!nrow(coordinates)) stop("The working-model mask is empty.")
    lower <- apply(coordinates, 2L, min)
    upper <- apply(coordinates, 2L, max)
    extent <- as.integer(upper - lower + 1L)
    work_dim <- as.integer(ceiling(extent / 4L) * 4L)
    relative <- sweep(coordinates, 2L, lower - 1L, "-")
    work_indices <- relative[, 1L] +
        (relative[, 2L] - 1L) * work_dim[[1L]] +
        (relative[, 3L] - 1L) * work_dim[[1L]] * work_dim[[2L]]

    set.seed(seed)
    x_work <- array(rnorm(prod(work_dim)), dim = work_dim)
    x_til_work <- array(rnorm(prod(work_dim)), dim = work_dim)
    x_work[work_indices] <- x[mask]
    x_til_work[work_indices] <- x_til[mask]
    list(
        x = x_work,
        x_til = x_til_work,
        mask_indices = which(mask),
        work_indices = as.integer(work_indices),
        lower = as.integer(lower),
        upper = as.integer(upper),
        work_dim = work_dim,
        padding_voxels = prod(work_dim) - sum(mask)
    )
}

.seft_expand_scores <- function(scores, crop, full_dim, mask, model_key) {
    expand <- function(values, outside) {
        output <- array(outside, dim = full_dim)
        output[crop$mask_indices] <- as.numeric(values[crop$work_indices])
        output
    }
    fit <- scores$fit
    fit$crop_lower_1based <- crop$lower
    fit$crop_upper_1based <- crop$upper
    fit$working_dimensions <- crop$work_dim
    fit$padding_voxels <- crop$padding_voxels
    list(
        R = expand(scores$R, 1),
        R_til = expand(scores$R_til, 1),
        log_R = expand(scores$log_R, 0),
        log_R_til = expand(scores$log_R_til, 0),
        background = expand(scores$background, 0),
        fit = fit,
        model = scores$model,
        model_identifier = model_key
    )
}

run_seft_working_model <- function(
    x,
    x_til,
    mask = NULL,
    working_model = "co",
    seed = 1L,
    density_bandwidth,
    spatial_bandwidth = 5,
    lambda = 0.5,
    neighbor_range = 10L,
    score_clip_c = 0.99
) {
    if (!is.array(x) || length(dim(x)) != 3L || !identical(dim(x), dim(x_til))) {
        stop("x and x_til must be three-dimensional arrays with identical dimensions.")
    }
    if (is.null(mask)) mask <- array(TRUE, dim = dim(x))
    mask <- array(as.logical(mask), dim = dim(x))
    mask[is.na(mask)] <- FALSE
    if (!any(mask)) stop("The working-model mask is empty.")
    if (any(!is.finite(x[mask])) || any(!is.finite(x_til[mask]))) {
        stop("Working-model inputs must be finite inside the mask.")
    }

    model <- normalize_seft_working_model(working_model)
    if (model == "co") {
        p_x <- 2 * (1 - pnorm(abs(x)))
        p_til <- 2 * (1 - pnorm(abs(x_til)))
        p_x[!mask] <- 1
        p_til[!mask] <- 1
        scores <- cal_CLAW_scores_3d(
            x, x_til, p_x, p_til,
            h = density_bandwidth,
            bandwidth = spatial_bandwidth,
            lambda = lambda,
            neighbor_range = as.integer(neighbor_range),
            c = score_clip_c
        )
        scores$R[!mask] <- 1
        scores$R_til[!mask] <- 1
        scores$log_R <- log(pmax(scores$R, .Machine$double.xmin))
        scores$log_R_til <- log(pmax(scores$R_til, .Machine$double.xmin))
        scores$background <- .seft_absmax_background(x, x_til, mask)
        scores$fit <- list(
            density_bandwidth = density_bandwidth,
            spatial_bandwidth = spatial_bandwidth,
            lambda = lambda,
            neighbor_range = as.integer(neighbor_range),
            score_clip_c = score_clip_c
        )
        scores$model <- "seft_co"
        scores$model_identifier <- model
        return(scores)
    }

    crop <- .seft_model_crop(x, x_til, mask, as.integer(seed))
    full_mask <- array(TRUE, dim = crop$work_dim)
    scores <- switch(
        model,
        "fdr-smoothing" = cal_fdr_smoothing_scores_3d_absmax(
            crop$x, crop$x_til, mask = full_mask,
            seed = as.integer(seed + 1L),
            solver_mode = "grid_primal_dual",
            gfl_threads = 1L,
            pr_table_size = 65536L
        ),
        "deepfdr" = cal_deepfdr_scores_3d_absmax(
            crop$x, crop$x_til, mask = full_mask,
            seed = 0L,
            device = Sys.getenv("SEFT_DEEPFDR_DEVICE", unset = "auto")
        ),
        "fchmrf" = cal_fchmrf_scores_3d_absmax(
            crop$x, crop$x_til, mask = full_mask,
            appearance_mode = "background",
            seed = as.integer(seed + 37L)
        ),
        "ising" = cal_ising_scores_3d_absmax(
            crop$x, crop$x_til, mask = full_mask,
            seed = as.integer(seed + 53L)
        )
    )
    required <- c("R", "R_til", "log_R", "log_R_til", "background", "fit", "model")
    missing <- setdiff(required, names(scores))
    if (length(missing)) {
        stop("Working model omitted fields: ", paste(missing, collapse = ", "))
    }
    if (any(!is.finite(scores$log_R)) || any(!is.finite(scores$log_R_til))) {
        stop("Working model returned non-finite log scores.")
    }
    .seft_expand_scores(scores, crop, dim(x), mask, model)
}
