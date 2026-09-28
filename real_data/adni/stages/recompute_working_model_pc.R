#!/usr/bin/env Rscript

args <- commandArgs(TRUE)
file_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
if (!length(file_arg)) stop("Cannot locate script path")
repro_root <- normalizePath(file.path(dirname(sub("^--file=", "", file_arg[[1]])), "..", "..", ".."), mustWork = TRUE)
work_dir <- if (length(args) >= 3L) normalizePath(args[[3]], mustWork = TRUE) else file.path(repro_root, "outputs", "adni3")
output_dir <- if (length(args)) normalizePath(args[[1]], mustWork = TRUE) else file.path(work_dir, "results", "absmax_working_models_real_seed20260723")
atlas_name <- if (length(args) >= 2L) args[[2]] else "aal3"
if (!atlas_name %in% c("aal3", "harvard_oxford")) {
    stop("atlas_name must be 'aal3' or 'harvard_oxford'.")
}
application_dir <- file.path(work_dir, "results", "application_methods")
pc_levels <- c(0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5)
alpha <- 0.1
environment_bin <- normalizePath(
    file.path(R.home(), "..", "..", "bin"), mustWork = FALSE
)
Sys.setenv(PATH = paste(
    environment_bin, Sys.getenv("PATH"), sep = .Platform$path.sep
))

suppressPackageStartupMessages({
    library(data.table)
    library(RNifti)
    library(parallel)
})
oldwd <- getwd()
setwd(repro_root)
source(file.path("R", "core", "CLAW_functions.R"))
setwd(oldwd)

if (atlas_name == "aal3") {
    atlas_path <- file.path(
        repro_root, "real_data", "input",
        "atlas", "AAL3v1.nii.gz"
    )
    labels_path <- sub("\\.nii\\.gz$", ".nii.txt", atlas_path)
} else {
    atlas_path <- file.path(
        work_dir, "atlas", "harvard_oxford_thr25_2mm.nii.gz"
    )
    labels_path <- file.path(
        work_dir, "atlas", "harvard_oxford_thr25_2mm_labels.tsv"
    )
}
mask_path <- file.path(work_dir, "fslvbm", "stats", "GM_mask.nii.gz")
atlas <- round(as.array(readNifti(atlas_path)))
mask <- as.array(readNifti(mask_path)) > 0
labels <- fread(
    labels_path, header = atlas_name == "harvard_oxford", fill = TRUE
)
label_lookup <- setNames(as.character(labels[[2]]), as.character(labels[[1]]))
region_ids <- sort(unique(as.integer(atlas[mask & atlas > 0])))

method_specs <- data.table(
    method_key = c(
        "co", "fdrs_absmax", "deepfdr_absmax", "fchmrf_absmax",
        "ising_absmax"
    ),
    method = c(
        "SEFT-CO",
        "SEFT-FDRSmoothing-AbsMaxAdapt",
        "SEFT-DeepFDR-AbsMaxAdapt",
        "SEFT-fcHMRF-AbsMaxAdapt",
        "SEFT-Ising-AbsMaxAdapt"
    ),
    adaptation = c(
        "manuscript_logscore",
        "fdr_smoothing_absmax_grid_primal_dual_adaptation",
        "deepfdr_absmax_adaptation",
        "fchmrf_absmax_adaptation",
        "ising_absmax_plis_adaptation"
    )
)
contrasts <- c("CN_gt_Dementia", "MCI_gt_Dementia", "CN_gt_MCI")
tasks <- CJ(method_key = method_specs$method_key, contrast = contrasts)
tasks <- merge(tasks, method_specs, by = "method_key", sort = FALSE)

compute_evalue <- function(log_r, log_r_til, indices, u) {
    pair <- matrix(
        c(log_r[indices], log_r_til[indices]),
        nrow = 2L, byrow = TRUE
    )
    maximum <- max(pair)
    if (!is.finite(maximum)) return(NA_real_)
    cal_ELIS_cpp(
        pair - maximum, u,
        method = "Sym_miny",
        til_mth = "neglog",
        signmin_flag = FALSE
    )
}

evaluate_task <- function(index) {
    task <- tasks[index]
    if (task$method_key == "co") {
        # The saved CO score is voxelwise and atlas-independent.  The AAL3
        # run retained its internal score maps, so reuse those maps for both
        # regional partitions.
        prefix <- sprintf("sigma3_%s_aal3", task$contrast)
        internal_dir <- file.path(
            application_dir, "seft_runs", prefix, "maps", "claw_internal"
        )
        r <- as.array(readNifti(file.path(
            internal_dir, sprintf("%s_R.nii.gz", prefix)
        )))
        r_til <- as.array(readNifti(file.path(
            internal_dir, sprintf("%s_R_til.nii.gz", prefix)
        )))
        if (any(r[mask] < 0, na.rm = TRUE) ||
                any(r_til[mask] < 0, na.rm = TRUE)) {
            stop("SEFT-CO posterior scores must be nonnegative.")
        }
        log_r <- log(pmax(as.numeric(r[mask]), .Machine$double.xmin))
        log_r_til <- log(pmax(as.numeric(r_til[mask]), .Machine$double.xmin))
        atlas_score <- as.integer(atlas[mask])
    } else {
        payload_path <- file.path(
            output_dir, "scores",
            sprintf("%s__%s_scores.rds", task$method_key, task$contrast)
        )
        payload <- readRDS(payload_path)
        lower <- as.integer(payload$crop_lower_1based)
        upper <- as.integer(payload$crop_upper_1based)
        atlas_crop <- atlas[
            seq.int(lower[[1]], upper[[1]]),
            seq.int(lower[[2]], upper[[2]]),
            seq.int(lower[[3]], upper[[3]]),
            drop = FALSE
        ]
        atlas_score <- as.integer(atlas_crop[payload$gm_indices_in_crop])
        log_r <- payload$log_R_gm
        log_r_til <- payload$log_R_til_gm
    }
    if (length(atlas_score) != length(log_r) ||
            length(log_r) != length(log_r_til)) {
        stop("Score and atlas vectors differ for ", task$method_key)
    }

    result <- rbindlist(lapply(pc_levels, function(pc_level) {
        table <- rbindlist(lapply(region_ids, function(region_id) {
            indices <- which(atlas_score == region_id)
            u <- max(1L, ceiling(length(indices) * pc_level))
            e_value <- compute_evalue(log_r, log_r_til, indices, u)
            data.table(
                method = task$method,
                adaptation = task$adaptation,
                contrast = task$contrast,
                pc_level = pc_level,
                region_id = region_id,
                region_label = unname(
                    label_lookup[[as.character(region_id)]]
                ),
                n_voxels = length(indices),
                u = u,
                e_value = e_value,
                pc_p_value = if (
                    is.finite(e_value) && e_value > 0
                ) min(1, 1 / e_value) else 1
            )
        }))
        table[, significant := BH(pc_p_value, alpha)]
        table[, alpha := alpha]
        table
    }))
    result
}

workers <- min(12L, nrow(tasks))
results <- mclapply(
    seq_len(nrow(tasks)), evaluate_task,
    mc.cores = workers, mc.preschedule = FALSE
)
if (any(vapply(results, inherits, logical(1), "try-error"))) {
    stop("At least one all-PC score task failed.")
}
combined <- rbindlist(results)
setorder(combined, contrast, method, pc_level, region_id)
expected <- nrow(tasks) * length(pc_levels) * length(region_ids)
if (nrow(combined) != expected) stop("Unexpected all-PC table size.")

result_suffix <- if (atlas_name == "aal3") {
    ""
} else {
    "_harvard_oxford"
}
fwrite(
    combined,
    file.path(
        output_dir, "tables",
        paste0("combined_region_results_all_pc", result_suffix, ".tsv")
    ),
    sep = "\t"
)
summary <- combined[, .(
    discoveries = sum(significant),
    n_regions = .N
), by = .(method, adaptation, contrast, pc_level, alpha)]
fwrite(
    summary,
    file.path(
        output_dir, "tables",
        paste0("combined_region_summary_all_pc", result_suffix, ".tsv")
    ),
    sep = "\t"
)
writeLines(c(
    paste("updated_at:", format(Sys.time(), tz = "UTC")),
    paste("pc_levels:", paste(pc_levels, collapse = ",")),
    paste("alpha:", alpha),
    paste("workers:", workers),
    paste("rows:", nrow(combined)),
    paste("atlas:", atlas_name)
), file.path(
    output_dir,
    if (atlas_name == "aal3") {
        "ALL_PC_COMPLETE"
    } else {
        "ALL_PC_HARVARD_OXFORD_COMPLETE"
    }
))
print(summary)
