#' Run SEFT inference
#'
#' Run region-level spatial e-value inference from a signed Z map or a t
#' statistic map. Inputs may be NIfTI paths, `RNifti` images, or numeric
#' three-dimensional arrays.
#'
#' @param zmap Signed Z map. Exactly one of `zmap` and `tstat` is required.
#' @param tstat Student t statistic map.
#' @param atlas Integer-valued region atlas.
#' @param atlas_labels Optional label file, data frame with `region_id` and
#'   `label` columns, or named character vector.
#' @param mask Optional analysis mask.
#' @param df Residual degrees of freedom for `tstat`.
#' @param design Design matrix or FSL VEST design-matrix path. Used to derive
#'   residual degrees of freedom for `tstat`.
#' @param working_model One or more of `"co"`, `"ising"`,
#'   `"fdr-smoothing"`, `"deepfdr"`, and `"fchmrf"`. The last three require
#'   the optional working-model environment.
#' @param alpha Multiple-testing level.
#' @param pc_levels Partial-conjunction proportions.
#' @param denoise Either `"none"` or `"wavelet"`.
#' @param bandwidth Spatial bandwidth in voxels for the CO working model.
#' @param neighbor_range CO neighbourhood radius in voxels.
#' @param lambda CO sparsity tuning parameter.
#' @param score_clip Upper clipping constant used by the CO score.
#' @param simes Include the BHe+Simes comparator.
#' @param seed Random seed used for the shared mirror statistics and model fits.
#' @param out_dir Optional output directory. If supplied, tables, maps, an RDS
#'   object, and a regional PDF are written.
#' @param prefix Output filename prefix.
#' @param keep_scores Retain voxelwise working-model scores in the returned
#'   object and RDS output.
#' @param verbose Print progress messages.
#'
#' @return An object of class `seft_result`.
#' @export
seft <- function(
    zmap = NULL,
    tstat = NULL,
    atlas,
    atlas_labels = NULL,
    mask = NULL,
    df = NULL,
    design = NULL,
    working_model = "co",
    alpha = 0.1,
    pc_levels = c(0.1, 0.2, 0.3),
    denoise = c("none", "wavelet"),
    bandwidth = 5,
    neighbor_range = 10L,
    lambda = 0.5,
    score_clip = 0.99,
    simes = FALSE,
    seed = 1L,
    out_dir = NULL,
    prefix = "seft",
    keep_scores = FALSE,
    verbose = interactive()
) {
    call <- match.call()
    denoise <- match.arg(denoise)
    .validate_scalar(alpha, "alpha", lower = 0, upper = 1, lower_open = TRUE)
    .validate_scalar(bandwidth, "bandwidth", lower = 0, lower_open = TRUE)
    .validate_scalar(lambda, "lambda", lower = 0, upper = 1, lower_open = TRUE, upper_open = TRUE)
    .validate_scalar(score_clip, "score_clip", lower = 0, upper = 1, lower_open = TRUE, upper_open = TRUE)
    if (length(neighbor_range) != 1L || is.na(neighbor_range) || neighbor_range < 1 || neighbor_range != as.integer(neighbor_range)) {
        stop("neighbor_range must be a positive integer.", call. = FALSE)
    }
    if (!length(pc_levels) || any(!is.finite(pc_levels)) || any(pc_levels <= 0 | pc_levels > 1)) {
        stop("pc_levels must contain values in (0, 1].", call. = FALSE)
    }
    pc_levels <- sort(unique(as.numeric(pc_levels)))
    if (length(seed) != 1L || is.na(seed) || seed != as.integer(seed)) {
        stop("seed must be a single integer.", call. = FALSE)
    }
    if (length(prefix) != 1L || !nzchar(prefix) || grepl("[/\\\\]", prefix)) {
        stop("prefix must be a non-empty filename stem.", call. = FALSE)
    }
    models <- .normalize_models(working_model)

    supplied <- c(!is.null(zmap), !is.null(tstat))
    if (sum(supplied) != 1L) {
        stop("Supply exactly one of zmap and tstat.", call. = FALSE)
    }
    if (!is.null(zmap) && (!is.null(df) || !is.null(design))) {
        stop("df and design are only used with tstat.", call. = FALSE)
    }
    if (!is.null(tstat) && is.null(df) && is.null(design)) {
        stop("tstat requires either df or design.", call. = FALSE)
    }
    if (!is.null(tstat) && !is.null(df) && !is.null(design)) {
        stop("Supply df or design for tstat, not both.", call. = FALSE)
    }

    stat_input <- .read_volume(if (is.null(zmap)) tstat else zmap, "statistic map")
    atlas_input <- .read_volume(atlas, "atlas")
    .assert_geometry(stat_input, atlas_input, "statistic map", "atlas")
    stat_data <- stat_input$data
    atlas_raw <- atlas_input$data
    if (any(!is.finite(atlas_raw))) stop("atlas contains non-finite values.", call. = FALSE)
    if (any(atlas_raw < 0) || any(abs(atlas_raw - round(atlas_raw)) > 1e-6)) {
        stop("atlas must contain non-negative integer region identifiers.", call. = FALSE)
    }
    atlas_data <- array(as.integer(round(atlas_raw)), dim = dim(atlas_raw))

    residual_df <- NULL
    if (!is.null(tstat)) {
        residual_df <- if (!is.null(df)) .validate_df(df) else .design_df(design)
        stat_data <- .t_to_z(stat_data, residual_df)
    }
    signed_z <- array(as.numeric(stat_data), dim = dim(stat_data))
    signed_z[!is.finite(signed_z)] <- 0

    analysis_mask <- atlas_data > 0L & is.finite(stat_data)
    mask_input <- NULL
    if (!is.null(mask)) {
        mask_input <- .read_volume(mask, "mask")
        .assert_geometry(stat_input, mask_input, "statistic map", "mask")
        analysis_mask <- analysis_mask & is.finite(mask_input$data) & mask_input$data > 0
    }
    if (!any(analysis_mask)) stop("No atlas voxels overlap the analysis mask.", call. = FALSE)
    region_map <- atlas_data
    region_map[!analysis_mask] <- 0L
    region_ids <- sort(unique(region_map[region_map > 0L]))
    labels <- .read_labels(atlas_labels, region_ids)

    model_z <- signed_z
    model_z[!analysis_mask] <- 0
    model_z[analysis_mask & model_z == 0] <- 1e-12
    if (denoise == "wavelet") {
        model_z <- .wavelet_denoise(model_z, analysis_mask)
        model_z[!analysis_mask] <- 0
        model_z[analysis_mask & model_z == 0] <- 1e-12
    }

    rng_state <- .preserve_rng()
    on.exit(.restore_rng(rng_state), add = TRUE)
    set.seed(as.integer(seed))
    mirror_z <- array(stats::rnorm(length(model_z)), dim = dim(model_z))
    mirror_z[!analysis_mask] <- 0
    mirror_z[analysis_mask & mirror_z == 0] <- 1e-12
    density_sample <- c(mirror_z[analysis_mask], model_z[analysis_mask])
    if (length(density_sample) > 1000L) density_sample <- sample(density_sample, 1000L)
    density_bandwidth <- tryCatch(stats::density(density_sample, bw = "nrd0")$bw, error = function(e) NA_real_)
    if (!is.finite(density_bandwidth) || density_bandwidth <= 0) density_bandwidth <- 1

    if (isTRUE(verbose)) {
        message(sprintf("SEFT: %d regions and %d voxels; models: %s", length(region_ids), sum(analysis_mask), paste(models, collapse = ", ")))
    }
    score_results <- setNames(vector("list", length(models)), models)
    for (model in models) {
        score_results[[model]] <- .run_working_model(
            model_z, mirror_z, analysis_mask, model,
            seed = as.integer(seed), density_bandwidth = density_bandwidth,
            bandwidth = bandwidth, neighbor_range = as.integer(neighbor_range),
            lambda = lambda, score_clip = score_clip, verbose = verbose
        )
    }

    regions <- .regional_inference(
        score_results, model_z, region_map, region_ids, labels,
        pc_levels = pc_levels, alpha = alpha, simes = isTRUE(simes)
    )
    summaries <- .summarise_regions(regions, alpha, seed)
    model_objects <- lapply(score_results, function(x) {
        result <- list(model_identifier = x$model_identifier, fit = x$fit)
        if (isTRUE(keep_scores)) result$scores <- x[c("R", "R_til", "log_R", "log_R_til", "background")]
        result
    })
    metadata <- list(
        package_version = tryCatch(as.character(utils::packageVersion("SEFT")), error = function(e) "0.0.0.9000"),
        statistic = if (is.null(zmap)) "signed Z converted from t" else "signed Z",
        residual_df = residual_df,
        dimensions = dim(signed_z),
        affine = stat_input$affine,
        n_mask_voxels = sum(analysis_mask),
        n_regions = length(region_ids),
        working_models = models,
        alpha = alpha,
        pc_levels = pc_levels,
        denoise = denoise,
        bandwidth = bandwidth,
        neighbor_range = as.integer(neighbor_range),
        lambda = lambda,
        score_clip = score_clip,
        simes = isTRUE(simes),
        seed = as.integer(seed),
        input_paths = list(statistic = stat_input$path, atlas = atlas_input$path, mask = if (is.null(mask_input)) NULL else mask_input$path)
    )
    result <- structure(list(
        call = call,
        regions = regions,
        summary = summaries,
        models = model_objects,
        metadata = metadata,
        output_files = NULL
    ), class = "seft_result")

    if (!is.null(out_dir)) {
        result$output_files <- .write_result(
            result, out_dir, prefix, signed_z, region_map,
            template = stat_input$image %||% atlas_input$image
        )
        saveRDS(result, result$output_files$rds, compress = "xz")
    }
    result
}

`%||%` <- function(x, y) if (is.null(x)) y else x

.validate_scalar <- function(x, name, lower = -Inf, upper = Inf, lower_open = FALSE, upper_open = FALSE) {
    valid <- length(x) == 1L && is.finite(x)
    if (valid) valid <- if (lower_open) x > lower else x >= lower
    if (valid) valid <- if (upper_open) x < upper else x <= upper
    if (!valid) stop(name, " is outside its allowed range.", call. = FALSE)
    invisible(as.numeric(x))
}

.validate_df <- function(df) {
    if (length(df) != 1L || !is.finite(df) || df <= 0) stop("df must be finite and positive.", call. = FALSE)
    as.numeric(df)
}

.normalize_models <- function(models) {
    if (length(models) == 1L && grepl(",", models, fixed = TRUE)) models <- strsplit(models, ",", fixed = TRUE)[[1L]]
    models <- trimws(tolower(as.character(models)))
    models[models == "covariate-adaptive"] <- "co"
    models[models == "fdrs"] <- "fdr-smoothing"
    models <- unique(models[nzchar(models)])
    unsupported <- setdiff(models, c("co", "ising", "fdr-smoothing", "deepfdr", "fchmrf"))
    if (!length(models) || length(unsupported)) {
        stop("working_model must contain co, ising, fdr-smoothing, deepfdr, and/or fchmrf.", call. = FALSE)
    }
    models
}

.preserve_rng <- function() {
    if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) {
        list(exists = TRUE, value = get(".Random.seed", envir = .GlobalEnv, inherits = FALSE))
    } else list(exists = FALSE, value = NULL)
}

.restore_rng <- function(state) {
    if (isTRUE(state$exists)) assign(".Random.seed", state$value, envir = .GlobalEnv)
    else if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) rm(".Random.seed", envir = .GlobalEnv)
    invisible(NULL)
}
