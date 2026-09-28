#!/usr/bin/env Rscript

parse_args <- function(args) {
    out <- list()
    i <- 1
    while (i <= length(args)) {
        key <- args[[i]]
        if (!startsWith(key, "--")) {
            stop(sprintf("Unexpected positional argument: %s", key))
        }
        name <- substring(key, 3)
        if (name %in% c("simes", "save-internals")) {
            out[[name]] <- TRUE
            i <- i + 1
        } else {
            if (i == length(args)) {
                stop(sprintf("Missing value for %s", key))
            }
            out[[name]] <- args[[i + 1]]
            i <- i + 2
        }
    }
    out
}

require_arg <- function(opts, name) {
    value <- opts[[name]]
    if (is.null(value) || !nzchar(value)) {
        stop(sprintf("Missing required argument --%s", name))
    }
    value
}

parse_float_list <- function(value) {
    as.numeric(strsplit(value, ",", fixed = TRUE)[[1]])
}

read_labels <- function(path) {
    if (is.null(path) || !file.exists(path)) {
        return(data.frame(region_id = integer(), label = character()))
    }
    first <- readLines(path, n = 1, warn = FALSE)
    if (grepl("region_id", first, ignore.case = TRUE)) {
        df <- utils::read.delim(path, stringsAsFactors = FALSE, check.names = FALSE)
        return(data.frame(region_id = as.integer(df$region_id), label = as.character(df$label)))
    }
    rows <- readLines(path, warn = FALSE)
    ids <- integer()
    labels <- character()
    for (line in rows) {
        line <- trimws(line)
        if (!nzchar(line) || startsWith(line, "#")) next
        pieces <- strsplit(line, "\\s+")[[1]]
        if (length(pieces) < 2) next
        id <- suppressWarnings(as.integer(as.numeric(pieces[[1]])))
        if (is.na(id)) next
        ids <- c(ids, id)
        labels <- c(labels, pieces[[2]])
    }
    data.frame(region_id = ids, label = labels, stringsAsFactors = FALSE)
}

bh_decision <- function(p_values, alpha) {
    p_values[!is.finite(p_values)] <- 1
    p_values <- pmin(pmax(p_values, 0), 1)
    order_idx <- order(p_values)
    m <- length(p_values)
    rejected <- rep(0L, m)
    for (i in seq(m, 1)) {
        if (p_values[order_idx][i] <= alpha * i / m) {
            rejected[order_idx][seq_len(i)] <- 1L
            break
        }
    }
    rejected
}

seft_pc_evalue <- function(results, region_mask, u) {
    log_scores <- matrix(
        c(results$log_R[region_mask], results$log_R_til[region_mask]),
        nrow = 2, byrow = TRUE
    )
    score_max <- suppressWarnings(max(log_scores))
    if (!is.finite(score_max)) {
        return(NA_real_)
    }
    value <- cal_ELIS_cpp(
        log_scores - score_max,
        u,
        method = "Sym_miny",
        til_mth = "neglog",
        signmin_flag = FALSE
    )
    value
}

opts <- parse_args(commandArgs(trailingOnly = TRUE))

zmap_path <- normalizePath(require_arg(opts, "zmap"), winslash = "/", mustWork = TRUE)
atlas_path <- normalizePath(require_arg(opts, "atlas"), winslash = "/", mustWork = TRUE)
mask_path <- if (!is.null(opts$mask)) normalizePath(opts$mask, winslash = "/", mustWork = TRUE) else NULL
labels_path <- if (!is.null(opts[["atlas-labels"]])) normalizePath(opts[["atlas-labels"]], winslash = "/", mustWork = FALSE) else NULL
out_dir <- normalizePath(require_arg(opts, "out-dir"), winslash = "/", mustWork = FALSE)
prefix <- require_arg(opts, "prefix")
working_model <- ifelse(is.null(opts[["working-model"]]), "co", opts[["working-model"]])
alpha <- as.numeric(ifelse(is.null(opts$alpha), "0.1", opts$alpha))
pc_levels <- parse_float_list(ifelse(is.null(opts[["pc-levels"]]), "0.01,0.05,0.1,0.2,0.3", opts[["pc-levels"]]))
seed <- as.integer(ifelse(is.null(opts$seed), "1", opts$seed))
denoise <- ifelse(is.null(opts$denoise), "none", opts$denoise)
bandwidth <- as.numeric(ifelse(is.null(opts$bandwidth), "5", opts$bandwidth))
lambda <- as.numeric(ifelse(is.null(opts$lambda), "0.5", opts$lambda))
neighbor_range <- as.integer(ifelse(is.null(opts[["neighbor-range"]]), "10", opts[["neighbor-range"]]))
score_clip_c <- as.numeric(ifelse(is.null(opts[["score-clip-c"]]), "0.99", opts[["score-clip-c"]]))
include_simes <- isTRUE(opts$simes)
save_internals <- isTRUE(opts[["save-internals"]])
seft_repro_dir <- normalizePath(require_arg(opts, "seft-repro-dir"), winslash = "/", mustWork = TRUE)

if (!is.finite(alpha) || alpha <= 0 || alpha > 1) stop("alpha must be in (0, 1].")
if (!length(pc_levels) || any(!is.finite(pc_levels)) || any(pc_levels <= 0 | pc_levels > 1)) {
    stop("Every PC level must be finite and in (0, 1].")
}
if (!is.finite(bandwidth) || bandwidth <= 0) stop("bandwidth must be finite and positive.")
if (!is.finite(lambda) || lambda <= 0 || lambda >= 1) stop("lambda must be in (0, 1).")
if (is.na(neighbor_range) || neighbor_range < 1) stop("neighbor-range must be positive.")
if (!is.finite(score_clip_c) || score_clip_c <= 0 || score_clip_c >= 1) {
    stop("score-clip-c must be in (0, 1).")
}

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

suppressPackageStartupMessages({
    library(oro.nifti)
    library(data.table)
    library(Rcpp)
    library(RcppArmadillo)
})

oldwd <- getwd()
setwd(seft_repro_dir)
source(file.path("R", "core", "CLAW_functions.R"))
source(file.path("R", "working_models", "working_model_dispatch.R"))
working_model <- normalize_seft_working_model(working_model)
if (working_model != "co") {
    Sys.setenv(
        SEFT_FDRS_ABSMAX_SCRIPT = file.path(
            seft_repro_dir, "python", "working_models", "fdr_smoothing_absmax.py"
        ),
        SEFT_ML_WORKING_MODEL_SCRIPT = file.path(
            seft_repro_dir, "python", "working_models", "ml_working_model.py"
        ),
        SEFT_FDRS_PYTHON = Sys.getenv("SEFT_ML_PYTHON", unset = Sys.which("python3")),
        SEFT_ML_PYTHON = Sys.getenv("SEFT_ML_PYTHON", unset = Sys.which("python3"))
    )
    source(file.path("R", "working_models", "working_model_scores.R"))
    if (working_model %in% c("deepfdr", "fchmrf")) {
        source(file.path("R", "working_models", "ml_working_model_scores.R"))
    }
    if (working_model == "ising") {
        source(file.path("R", "working_models", "ising_working_model_scores.R"))
    }
}
setwd(oldwd)

z_img <- readNIfTI(zmap_path, reorient = FALSE)
atlas_img <- readNIfTI(atlas_path, reorient = FALSE)
z_data <- drop(z_img@.Data)
atlas_raw <- drop(atlas_img@.Data)
atlas_data <- array(as.integer(round(atlas_raw)), dim = dim(atlas_raw))
if (!identical(dim(z_data), dim(atlas_data))) {
    stop("Z-map and atlas dimensions differ after Python geometry validation.")
}

analysis_mask <- is.finite(z_data) & atlas_data > 0
if (!is.null(mask_path)) {
    mask_img <- readNIfTI(mask_path, reorient = FALSE)
    mask_data <- drop(mask_img@.Data) > 0
    analysis_mask <- analysis_mask & mask_data
}
z_data[!analysis_mask] <- 0
z_data[analysis_mask & z_data == 0] <- 1e-12
region_masks <- atlas_data
region_masks[!analysis_mask] <- 0L
region_ids <- sort(unique(as.integer(region_masks[region_masks > 0])))

if (length(region_ids) == 0) {
    stop("No non-zero atlas regions overlap the analysis mask.")
}

if (denoise == "wavelet") {
    max_dim <- max(dim(z_data))
    expand_to <- rep(2 ^ ceiling(log2(max_dim)), 3)
    z_data <- apply_wavelet_denoising_3d(
        z_data,
        expand_to = expand_to,
        data_mask = analysis_mask,
        wf = "d8",
        target_level = 1,
        J = 7,
        verbose = FALSE
    )
    z_data[!analysis_mask] <- 0
    z_data[analysis_mask & z_data == 0] <- 1e-12
} else if (denoise != "none") {
    stop(sprintf("Unsupported denoise mode: %s", denoise))
}

cat(sprintf("SEFT/FSL core: %d regions, %d voxels in analysis mask\n", length(region_ids), sum(analysis_mask)))

set.seed(seed)
z_til <- array(rnorm(length(z_data)), dim = dim(z_data))
z_til[!analysis_mask] <- 0
z_til[analysis_mask & z_til == 0] <- 1e-12

sample_values <- c(z_til, z_data)
if (length(sample_values) > 1000) {
    sample_values <- sample(sample_values, 1000)
}
h <- stats::density(sample_values, bw = "nrd0")$bw
if (!is.finite(h) || h <= 0) {
    h <- 1
}

score_results <- run_seft_working_model(
    z_data,
    z_til,
    mask = analysis_mask,
    working_model = working_model,
    seed = seed,
    density_bandwidth = h,
    spatial_bandwidth = bandwidth,
    lambda = lambda,
    neighbor_range = neighbor_range,
    score_clip_c = score_clip_c
)
saveRDS(
    list(
        working_model = working_model,
        model_identifier = score_results$model,
        fit = score_results$fit
    ),
    file.path(out_dir, paste0(prefix, "_working_model_fit.rds")),
    compress = "xz"
)

if (save_internals) {
    internal_name <- if (working_model == "co") {
        "claw_internal"
    } else {
        paste0(gsub("-", "_", working_model), "_internal")
    }
    internal_dir <- file.path(dirname(out_dir), "maps", internal_name)
    dir.create(internal_dir, recursive = TRUE, showWarnings = FALSE)
    write_internal <- function(values, name) {
        values <- array(as.numeric(values), dim = dim(z_data))
        values[!analysis_mask] <- 0
        output_img <- z_img
        output_img@.Data <- values
        output_img@datatype <- as.integer(16)
        output_img@bitpix <- as.integer(32)
        output_base <- file.path(internal_dir, paste0(prefix, "_", name))
        writeNIfTI(output_img, filename = output_base, gzipped = TRUE)
    }
    write_internal(z_data, "z_model")
    write_internal(z_til, "z_til")
    internal_fields <- intersect(
        c("background", "pi", "log_f", "log_f_til", "R", "R_til", "log_R", "log_R_til"),
        names(score_results)
    )
    for (name in internal_fields) {
        write_internal(score_results[[name]], name)
    }
}

labels_df <- read_labels(labels_path)
label_lookup <- setNames(labels_df$label, as.character(labels_df$region_id))
methods <- c("seft", if (include_simes) "simes" else character())

rows <- list()
row_idx <- 1L
for (pc_level in pc_levels) {
    method_pvalues <- matrix(NA_real_, nrow = length(methods), ncol = length(region_ids))
    method_evalues <- matrix(NA_real_, nrow = length(methods), ncol = length(region_ids))
    n_voxels <- integer(length(region_ids))
    u_values <- integer(length(region_ids))

    for (i_region in seq_along(region_ids)) {
        region_id <- region_ids[[i_region]]
        region_mask <- region_masks == region_id
        n_voxels[[i_region]] <- sum(region_mask)
        u <- ceiling(n_voxels[[i_region]] * pc_level)
        u <- max(1L, as.integer(u))
        u_values[[i_region]] <- u

        seft_e <- seft_pc_evalue(score_results, region_mask, u)
        method_evalues[methods == "seft", i_region] <- seft_e
        method_pvalues[methods == "seft", i_region] <- ifelse(is.na(seft_e), 1, min(1, 1 / seft_e))

        if (include_simes) {
            pvals <- 2 * (1 - pnorm(abs(z_data[region_mask])))
            simes_p <- Simes_PC_test_cpp(pvals, u)
            simes_p <- min(1, max(0, simes_p))
            method_pvalues[methods == "simes", i_region] <- simes_p
            method_evalues[methods == "simes", i_region] <- ifelse(simes_p > 0, 1 / simes_p, Inf)
        }
    }

    for (i_method in seq_along(methods)) {
        decisions <- bh_decision(method_pvalues[i_method, ], alpha)
        for (i_region in seq_along(region_ids)) {
            region_id <- region_ids[[i_region]]
            label <- unname(label_lookup[as.character(region_id)])
            if (!length(label) || is.na(label)) label <- ""
            rows[[row_idx]] <- data.frame(
                pc_level = pc_level,
                method = methods[[i_method]],
                working_model = if (methods[[i_method]] == "seft") working_model else "none",
                model_identifier = if (methods[[i_method]] == "seft") score_results$model else "simes",
                region_id = region_id,
                region_label = label,
                n_voxels = n_voxels[[i_region]],
                u = u_values[[i_region]],
                e_value = method_evalues[i_method, i_region],
                pc_p_value = method_pvalues[i_method, i_region],
                significant = decisions[[i_region]],
                stringsAsFactors = FALSE
            )
            row_idx <- row_idx + 1L
        }
    }
}

result_df <- data.table::rbindlist(rows)
sig_df <- result_df[significant == 1L]
summary_df <- result_df[, .(
    total_regions = .N,
    discovered_regions = sum(significant == 1L),
    alpha = alpha,
    seed = seed,
    bandwidth = bandwidth,
    lambda = lambda,
    neighbor_range = neighbor_range,
    score_clip_c = score_clip_c
), by = .(pc_level, method, working_model, model_identifier)]

data.table::fwrite(result_df, file.path(out_dir, paste0(prefix, "_region_results.tsv")), sep = "\t")
data.table::fwrite(sig_df, file.path(out_dir, paste0(prefix, "_significant_regions.tsv")), sep = "\t")
data.table::fwrite(summary_df, file.path(out_dir, paste0(prefix, "_summary.tsv")), sep = "\t")

cat(sprintf("Wrote SEFT/FSL tables for prefix %s to %s\n", prefix, out_dir))
