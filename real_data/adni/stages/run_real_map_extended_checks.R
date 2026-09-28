#!/usr/bin/env Rscript

parse_args <- function(args) {
    out <- list()
    i <- 1L
    while (i <= length(args)) {
        key <- args[[i]]
        if (!startsWith(key, "--") || i == length(args)) stop("Arguments must be --name value pairs")
        out[[substring(key, 3L)]] <- args[[i + 1L]]
        i <- i + 2L
    }
    out
}

opts <- parse_args(commandArgs(trailingOnly = TRUE))
file_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
if (!length(file_arg)) stop("Cannot locate script path")
seft_root <- normalizePath(file.path(
    dirname(sub("^--file=", "", file_arg[[1]])), "..", "..", ".."
), winslash = "/", mustWork = TRUE)
work_dir <- normalizePath(
    ifelse(is.null(opts[["work-dir"]]),
           file.path(seft_root, "outputs", "adni3"),
           opts[["work-dir"]]),
    winslash = "/", mustWork = TRUE
)
workers <- as.integer(ifelse(is.null(opts$workers), "64", opts$workers))
seed <- as.integer(ifelse(is.null(opts$seed), "20260717", opts$seed))

suppressPackageStartupMessages({
    library(oro.nifti)
    library(data.table)
    library(Rcpp)
    library(RcppArmadillo)
    library(parallel)
})

oldwd <- getwd()
setwd(seft_root)
source(file.path("R", "core", "CLAW_functions.R"))
setwd(oldwd)

methods_dir <- file.path(work_dir, "results", "application_methods")
tables_dir <- file.path(methods_dir, "qc", "tables")
dir.create(tables_dir, recursive = TRUE, showWarnings = FALSE)
atlas_path <- file.path(seft_root, "real_data", "input", "atlas", "AAL3v1.nii.gz")
atlas <- array(
    as.integer(round(drop(readNIfTI(atlas_path, reorient = FALSE)@.Data))),
    dim = dim(drop(readNIfTI(atlas_path, reorient = FALSE)@.Data))
)
contrasts <- c("CN_gt_Dementia", "MCI_gt_Dementia", "CN_gt_MCI")
caps <- c(0.95, 0.975, 0.99, 0.995, 0.999)
pc_levels <- c(0.10, 0.20, 0.30)

read_internal <- function(contrast) {
    prefix <- paste0("sigma3_", contrast, "_aal3")
    directory <- file.path(methods_dir, "seft_runs", prefix, "maps", "claw_internal")
    read_one <- function(name) {
        path <- file.path(directory, paste0(prefix, "_", name, ".nii.gz"))
        if (!file.exists(path)) stop(sprintf("Missing CLAW internal map: %s", path))
        drop(readNIfTI(path, reorient = FALSE)@.Data)
    }
    list(
        prefix = prefix,
        z = read_one("z_model"),
        z_til = read_one("z_til"),
        pi = read_one("pi"),
        log_f = read_one("log_f"),
        log_f_til = read_one("log_f_til"),
        R = read_one("R"),
        R_til = read_one("R_til")
    )
}

bh_decision <- function(p_values, alpha = 0.10) {
    p_values[!is.finite(p_values)] <- 1
    p_values <- pmin(pmax(p_values, 0), 1)
    order_idx <- order(p_values)
    m <- length(p_values)
    rejected <- rep(0L, m)
    for (i in seq(m, 1L)) {
        if (p_values[order_idx][i] <= alpha * i / m) {
            rejected[order_idx][seq_len(i)] <- 1L
            break
        }
    }
    rejected
}

seft_pc_evalue <- function(R, R_til, region_mask, u) {
    score_matrix <- matrix(c(R[region_mask], R_til[region_mask]), nrow = 2L, byrow = TRUE)
    score_max <- suppressWarnings(max(score_matrix))
    if (!is.finite(score_max) || score_max == 0) return(NA_real_)
    cal_ELIS_cpp(
        log(score_matrix / score_max),
        u,
        method = "Sym_miny",
        til_mth = "neglog",
        signmin_flag = FALSE
    )
}

cap_task <- function(task) {
    contrast <- task$contrast
    cap <- task$cap
    values <- read_internal(contrast)
    active <- values$z != 0 & atlas > 0
    region_ids <- sort(unique(as.integer(atlas[active])))
    log_q <- log1p(-values$pi) + dnorm(values$z, log = TRUE) - values$log_f
    log_q_til <- log1p(-values$pi) + dnorm(values$z_til, log = TRUE) - values$log_f_til
    q <- exp(pmin(log_q, log(cap)))
    q_til <- exp(pmin(log_q_til, log(cap)))
    R <- (0.5 - values$pi) / (1 - values$pi) * q / (1 - q)
    R_til <- (0.5 - values$pi) / (1 - values$pi) * q_til / (1 - q_til)
    rows <- list()
    row_index <- 1L
    for (pc_level in pc_levels) {
        e_values <- vapply(region_ids, function(region_id) {
            region_mask <- active & atlas == region_id
            u <- max(1L, as.integer(ceiling(sum(region_mask) * pc_level)))
            seft_pc_evalue(R, R_til, region_mask, u)
        }, numeric(1))
        decisions <- bh_decision(ifelse(is.na(e_values), 1, pmin(1, 1 / e_values)))
        for (i in seq_along(region_ids)) {
            rows[[row_index]] <- data.frame(
                contrast = contrast,
                score_cap = cap,
                pc_level = pc_level,
                region_id = region_ids[[i]],
                e_value = e_values[[i]],
                selected = decisions[[i]],
                observed_clipping_fraction = mean(log_q[active] >= log(cap)),
                mirror_clipping_fraction = mean(log_q_til[active] >= log(cap)),
                stringsAsFactors = FALSE
            )
            row_index <- row_index + 1L
        }
    }
    rbindlist(rows)
}

cap_tasks <- lapply(contrasts, function(contrast) {
    lapply(caps, function(cap) list(contrast = contrast, cap = cap))
})
cap_tasks <- unlist(cap_tasks, recursive = FALSE)
cap_results <- rbindlist(mclapply(
    cap_tasks, cap_task, mc.cores = min(workers, length(cap_tasks)), mc.preschedule = FALSE
))

default_sets <- cap_results[abs(score_cap - 0.99) < 1e-12 & selected == 1L,
                            .(default_ids = list(sort(region_id))),
                            by = .(contrast, pc_level)]
summary_rows <- list()
summary_index <- 1L
for (contrast_value in contrasts) {
    for (cap_value in caps) {
        for (pc_value in pc_levels) {
            block <- cap_results[
                contrast == contrast_value & abs(score_cap - cap_value) < 1e-12 &
                    abs(pc_level - pc_value) < 1e-12
            ]
            current <- sort(block[selected == 1L, region_id])
            reference_row <- default_sets[
                contrast == contrast_value & abs(pc_level - pc_value) < 1e-12
            ]
            reference <- if (nrow(reference_row)) reference_row$default_ids[[1L]] else integer()
            union_ids <- union(current, reference)
            jaccard <- if (length(union_ids) == 0L) 1 else length(intersect(current, reference)) / length(union_ids)
            summary_rows[[summary_index]] <- data.frame(
                contrast = contrast_value,
                score_cap = cap_value,
                pc_level = pc_value,
                discovered_regions = length(current),
                default_discovered_regions = length(reference),
                jaccard_vs_cap0p99 = jaccard,
                gained_regions = length(setdiff(current, reference)),
                lost_regions = length(setdiff(reference, current)),
                gained_region_ids = paste(setdiff(current, reference), collapse = ","),
                lost_region_ids = paste(setdiff(reference, current), collapse = ","),
                observed_clipping_fraction = unique(block$observed_clipping_fraction)[[1L]],
                mirror_clipping_fraction = unique(block$mirror_clipping_fraction)[[1L]],
                maximum_score_when_pi_zero = 0.5 * cap_value / (1 - cap_value),
                stringsAsFactors = FALSE
            )
            summary_index <- summary_index + 1L
        }
    }
}
cap_summary <- rbindlist(summary_rows)

archived <- fread(file.path(methods_dir, "tables", "seft_simes_region_results_long.tsv"))
archived <- archived[
    atlas == "aal3" & variant == "sigma3" & method == "seft" &
        abs(alpha - 0.10) < 1e-12 & contrast %in% contrasts,
    .(contrast, pc_level, region_id, archived_selected = significant_recomputed)
]
default_check <- merge(
    cap_results[abs(score_cap - 0.99) < 1e-12,
                .(contrast, pc_level, region_id, recomputed_selected = selected)],
    archived,
    by = c("contrast", "pc_level", "region_id"),
    all = TRUE
)
if (any(is.na(default_check)) || any(default_check$recomputed_selected != default_check$archived_selected)) {
    stop("score_cap=0.99 failed to reproduce archived region decisions")
}

fwrite(cap_results, file.path(tables_dir, "score_cap_sensitivity_by_region.tsv"), sep = "\t")
fwrite(cap_summary, file.path(tables_dir, "score_cap_sensitivity_summary.tsv"), sep = "\t")

audit_contrast <- function(contrast_index) {
    contrast <- contrasts[[contrast_index]]
    values <- read_internal(contrast)
    active <- values$z != 0 & atlas > 0
    p <- 2 * (1 - pnorm(abs(values$z)))
    p_til <- 2 * (1 - pnorm(abs(values$z_til)))
    p[!active] <- 1
    p_til[!active] <- 1
    set.seed(seed + 1000L + contrast_index)
    pooled <- c(values$z[active], values$z_til[active])
    bandwidth_h <- density(sample(pooled, min(1000L, length(pooled))), bw = "nrd0")$bw
    baseline <- cal_CLAW_scores_3d(
        values$z, values$z_til, p, p_til,
        h = bandwidth_h, bandwidth = 5, lambda = 0.5,
        neighbor_range = 10, c = 0.99
    )
    set.seed(seed + 2000L + contrast_index)
    random_swap <- array(FALSE, dim = dim(active))
    random_swap[active] <- runif(sum(active)) < 0.5
    patterns <- list(
        all_swap = active,
        random50 = random_swap
    )
    rows <- list()
    row_index <- 1L
    for (pattern_name in names(patterns)) {
        swap <- patterns[[pattern_name]]
        z_swapped <- values$z
        z_til_swapped <- values$z_til
        p_swapped <- p
        p_til_swapped <- p_til
        z_swapped[swap] <- values$z_til[swap]
        z_til_swapped[swap] <- values$z[swap]
        p_swapped[swap] <- p_til[swap]
        p_til_swapped[swap] <- p[swap]
        refit <- cal_CLAW_scores_3d(
            z_swapped, z_til_swapped, p_swapped, p_til_swapped,
            h = bandwidth_h, bandwidth = 5, lambda = 0.5,
            neighbor_range = 10, c = 0.99
        )
        expected <- list(
            pi = baseline$pi,
            log_f = baseline$log_f,
            log_f_til = baseline$log_f_til,
            R = baseline$R,
            R_til = baseline$R_til
        )
        for (pair in list(c("log_f", "log_f_til"), c("R", "R_til"))) {
            left <- pair[[1L]]
            right <- pair[[2L]]
            expected[[left]][swap] <- baseline[[right]][swap]
            expected[[right]][swap] <- baseline[[left]][swap]
        }
        for (quantity in names(expected)) {
            observed <- refit[[quantity]][active]
            target <- expected[[quantity]][active]
            if (quantity %in% c("R", "R_til")) {
                observed <- log(pmax(observed, .Machine$double.xmin))
                target <- log(pmax(target, .Machine$double.xmin))
                comparison_scale <- "log"
            } else {
                comparison_scale <- "native"
            }
            error <- abs(observed - target)
            rows[[row_index]] <- data.frame(
                contrast = contrast,
                swap_pattern = pattern_name,
                swapped_fraction = mean(swap[active]),
                quantity = quantity,
                comparison_scale = comparison_scale,
                median_absolute_error = median(error),
                q999_absolute_error = unname(quantile(error, 0.999)),
                maximum_absolute_error = max(error),
                pass_tolerance_1e8 = as.integer(max(error) < 1e-8),
                stringsAsFactors = FALSE
            )
            row_index <- row_index + 1L
        }
    }
    rbindlist(rows)
}

audit <- rbindlist(mclapply(
    seq_along(contrasts), audit_contrast,
    mc.cores = min(length(contrasts), workers), mc.preschedule = FALSE
))
fwrite(audit, file.path(tables_dir, "actual_map_swap_audit.tsv"), sep = "\t")
if (any(audit$pass_tolerance_1e8 != 1L)) {
    stop("Actual-map swap audit exceeded the prespecified numerical tolerance")
}

writeLines(
    format(Sys.time(), tz = "Asia/Shanghai", usetz = TRUE),
    file.path(methods_dir, "qc", "extended_model_checks_complete.ok")
)
cat(sprintf("Wrote score-cap sensitivity (%d rows) and swap audit (%d rows)\n",
            nrow(cap_summary), nrow(audit)))
