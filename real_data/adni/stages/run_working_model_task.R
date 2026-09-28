#!/usr/bin/env Rscript

parse_options <- function(args) {
    defaults <- list(
        method = "",
        contrast = "",
        seed = 20260723L,
        alpha = 0.1,
        pc_levels = "0.01,0.05,0.1,0.2,0.3,0.4,0.5",
        ising_iter_max = 20L,
        ising_sweep_b = 1L,
        ising_sweep_r = 2L,
        ising_burnin_lis = 20L,
        ising_sweep_lis = 50L,
        ising_n_chains = 2L,
        ising_fit_seed_offset = 53L,
        ising_fit_attempts = 1L,
        ising_basic_convergence_tol = Inf,
        ising_basic_convergence_max = Inf,
        ising_require_basic_convergence = 0L,
        work_dir = "",
        output_dir = ""
    )
    if (length(args) %% 2L != 0L) stop("Arguments must be --name value pairs.")
    for (i in seq(1L, length(args), by = 2L)) {
        key <- gsub("-", "_", sub("^--", "", args[[i]]))
        if (is.null(defaults[[key]])) stop("Unknown option: ", args[[i]])
        value <- args[[i + 1L]]
        if (is.integer(defaults[[key]])) value <- as.integer(value)
        if (is.numeric(defaults[[key]])) value <- as.numeric(value)
        defaults[[key]] <- value
    }
    methods <- c(
        "fdrs_absmax", "deepfdr_absmax", "fchmrf_absmax",
        "ising_absmax"
    )
    contrasts <- c("CN_gt_Dementia", "MCI_gt_Dementia", "CN_gt_MCI")
    if (!defaults$method %in% methods) {
        stop("--method must be one of: ", paste(methods, collapse = ", "))
    }
    if (!defaults$contrast %in% contrasts) {
        stop("--contrast must be one of: ", paste(contrasts, collapse = ", "))
    }
    defaults
}

find_repro_root <- function() {
    file_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
    candidates <- c(
        getwd(),
        if (length(file_arg)) file.path(
            dirname(sub("^--file=", "", file_arg[[1]])), "..", "..", ".."
        )
    )
    for (candidate in candidates) {
        candidate <- normalizePath(candidate, mustWork = FALSE)
        if (file.exists(file.path(candidate, "R", "working_models", "ml_working_model_scores.R"))) return(candidate)
    }
    stop("Cannot locate SEFT_repro root.")
}

opts <- parse_options(commandArgs(TRUE))
repro_root <- find_repro_root()
setwd(repro_root)
if (!nzchar(opts$work_dir)) opts$work_dir <- file.path(repro_root, "outputs", "adni3")
work_dir <- normalizePath(opts$work_dir, mustWork = TRUE)
if (!nzchar(opts$output_dir)) {
    opts$output_dir <- file.path(
        work_dir, "results",
        "absmax_working_models_real_seed20260723"
    )
}
output_dir <- normalizePath(opts$output_dir, mustWork = FALSE)
dir.create(file.path(output_dir, "tables"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(output_dir, "maps"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(output_dir, "scores"), recursive = TRUE, showWarnings = FALSE)

environment_bin <- normalizePath(
    file.path(R.home(), "..", "..", "bin"), mustWork = FALSE
)
Sys.setenv(
    PATH = paste(environment_bin, Sys.getenv("PATH"),
                 sep = .Platform$path.sep),
    OMP_NUM_THREADS = "1",
    MKL_NUM_THREADS = "1",
    OPENBLAS_NUM_THREADS = "1",
    NUMEXPR_NUM_THREADS = "1",
    TORCH_NUM_THREADS = "1",
    SEFT_FDRS_ABSMAX_SCRIPT = file.path(
        repro_root, "python", "working_models",
        "fdr_smoothing_absmax.py"
    ),
    SEFT_ML_WORKING_MODEL_SCRIPT = file.path(
        repro_root, "python", "working_models",
        "ml_working_model.py"
    ),
    SEFT_FDRS_PYTHON = Sys.getenv("SEFT_ML_PYTHON", unset = Sys.which("python3")),
    SEFT_ML_PYTHON = Sys.getenv("SEFT_ML_PYTHON", unset = Sys.which("python3"))
)

suppressPackageStartupMessages({
    library(data.table)
    library(RNifti)
})
oldwd <- getwd()
setwd(repro_root)
source(file.path("R", "core", "CLAW_functions.R"))
setwd(oldwd)
source(file.path(
    repro_root, "R", "working_models",
    "working_model_scores.R"
))
source(file.path(
    repro_root, "R", "working_models",
    "ml_working_model_scores.R"
))
source(file.path(
    repro_root, "R", "working_models",
    "ising_working_model_scores.R"
))

method_labels <- c(
    fdrs_absmax = "SEFT-FDRSmoothing-AbsMaxAdapt",
    deepfdr_absmax = "SEFT-DeepFDR-AbsMaxAdapt",
    fchmrf_absmax = "SEFT-fcHMRF-AbsMaxAdapt",
    ising_absmax = "SEFT-Ising-AbsMaxAdapt"
)
contrast_offsets <- c(
    CN_gt_Dementia = 101L,
    MCI_gt_Dementia = 211L,
    CN_gt_MCI = 307L
)
pc_levels <- as.numeric(strsplit(opts$pc_levels, ",", fixed = TRUE)[[1]])
task_seed <- as.integer(opts$seed + contrast_offsets[[opts$contrast]])
stem <- paste(opts$method, opts$contrast, sep = "__")

z_path <- file.path(
    work_dir, "results", "application_methods", "zmaps",
    sprintf("sigma3_%s_signed_z.nii.gz", opts$contrast)
)
mask_path <- file.path(work_dir, "fslvbm", "stats", "GM_mask.nii.gz")
atlas_path <- file.path(
    repro_root, "real_data", "input",
    "atlas", "AAL3v1.nii.gz"
)
label_path <- sub("\\.nii\\.gz$", ".nii.txt", atlas_path)
stopifnot(file.exists(z_path), file.exists(mask_path), file.exists(atlas_path))

z_full <- as.array(readNifti(z_path))
mask_full <- as.array(readNifti(mask_path)) > 0
atlas_full <- round(as.array(readNifti(atlas_path)))
if (!identical(dim(z_full), dim(mask_full)) ||
        !identical(dim(z_full), dim(atlas_full))) {
    stop("Z map, GM mask, and AAL3 atlas geometries differ.")
}

coordinates <- which(mask_full, arr.ind = TRUE)
lower <- apply(coordinates, 2L, min)
upper <- apply(coordinates, 2L, max)
crop_dim <- as.integer(ceiling((upper - lower + 1L) / 4L) * 4L)
crop_upper <- lower + crop_dim - 1L
if (any(crop_upper > dim(z_full))) stop("Padded crop exceeds image geometry.")
crop <- list(
    seq.int(lower[[1]], crop_upper[[1]]),
    seq.int(lower[[2]], crop_upper[[2]]),
    seq.int(lower[[3]], crop_upper[[3]])
)
mask_crop <- mask_full[crop[[1]], crop[[2]], crop[[3]], drop = FALSE]
atlas_crop <- atlas_full[crop[[1]], crop[[2]], crop[[3]], drop = FALSE]
z_crop <- z_full[crop[[1]], crop[[2]], crop[[3]], drop = FALSE]

# The padded voxels are generated from the same exchangeable N(0,1) pair for
# every method. They prevent zero padding from becoming an artificial feature.
set.seed(task_seed)
x <- array(rnorm(prod(crop_dim)), dim = crop_dim)
x_til <- array(rnorm(prod(crop_dim)), dim = crop_dim)
x[mask_crop] <- z_crop[mask_crop]

cat(sprintf(
    "task=%s seed=%d full=%s crop=%s gm_voxels=%d padding_voxels=%d\n",
    stem, task_seed, paste(dim(z_full), collapse = "x"),
    paste(crop_dim, collapse = "x"), sum(mask_crop),
    length(mask_crop) - sum(mask_crop)
))

tic <- proc.time()[["elapsed"]]
scores <- switch(
    opts$method,
    fdrs_absmax = cal_fdr_smoothing_scores_3d_absmax(
        x, x_til, mask = array(TRUE, dim = crop_dim),
        seed = task_seed + 1L,
        solver_mode = "grid_primal_dual",
        gfl_threads = 1L,
        pr_table_size = 65536L
    ),
    deepfdr_absmax = cal_deepfdr_scores_3d_absmax(
        x, x_til, mask = array(TRUE, dim = crop_dim),
        seed = 0L,
        device = Sys.getenv("SEFT_DEEPFDR_DEVICE", unset = "auto")
    ),
    fchmrf_absmax = cal_fchmrf_scores_3d_absmax(
        x, x_til, mask = array(TRUE, dim = crop_dim),
        appearance_mode = "background",
        seed = task_seed + 37L
    ),
    ising_absmax = cal_ising_scores_3d_absmax(
        x, x_til, mask = array(TRUE, dim = crop_dim),
        # The weak CN--MCI contrast can enter a near-all-null Ising phase.
        # Preserve the free-beta result and diagnostics.  Longer MCMC can
        # diagnose mixing but must not be used to select a richer map.
        seed = task_seed + opts$ising_fit_seed_offset,
        iter_max = opts$ising_iter_max,
        sweep_b = opts$ising_sweep_b,
        sweep_r = opts$ising_sweep_r,
        burnin_lis = opts$ising_burnin_lis,
        sweep_lis = opts$ising_sweep_lis,
        n_chains = opts$ising_n_chains,
        fit_attempts = opts$ising_fit_attempts,
        basic_convergence_tol = opts$ising_basic_convergence_tol,
        basic_convergence_max = opts$ising_basic_convergence_max,
        require_basic_convergence =
            opts$ising_require_basic_convergence == 1L
    )
)
elapsed <- proc.time()[["elapsed"]] - tic
if (any(!is.finite(scores$log_R)) ||
        any(!is.finite(scores$log_R_til))) {
    stop("Non-finite score produced.")
}

labels <- fread(label_path, header = FALSE, fill = TRUE)
label_lookup <- setNames(as.character(labels[[2]]), as.character(labels[[1]]))
region_ids <- sort(unique(as.integer(atlas_crop[mask_crop & atlas_crop > 0])))

seft_evalue <- function(indices, u) {
    log_pair <- matrix(
        c(scores$log_R[indices], scores$log_R_til[indices]),
        nrow = 2L, byrow = TRUE
    )
    maximum <- max(log_pair)
    if (!is.finite(maximum)) return(NA_real_)
    cal_ELIS_cpp(
        log_pair - maximum,
        u,
        method = "Sym_miny",
        til_mth = "neglog",
        signmin_flag = FALSE
    )
}

region_tables <- lapply(pc_levels, function(pc_level) {
    table <- rbindlist(lapply(region_ids, function(region_id) {
        indices <- which(mask_crop & atlas_crop == region_id)
        u <- max(1L, ceiling(length(indices) * pc_level))
        e_value <- seft_evalue(indices, u)
        data.table(
            method = unname(method_labels[[opts$method]]),
            adaptation = scores$model,
            contrast = opts$contrast,
            pc_level = pc_level,
            region_id = region_id,
            region_label = unname(label_lookup[[as.character(region_id)]]),
            n_voxels = length(indices),
            u = u,
            e_value = e_value,
            pc_p_value = if (is.finite(e_value) && e_value > 0) {
                min(1, 1 / e_value)
            } else {
                1
            }
        )
    }))
    table[, significant := BH(pc_p_value, opts$alpha)]
    table[, alpha := opts$alpha]
    table
})
region_table <- rbindlist(region_tables)

table_path <- file.path(output_dir, "tables", paste0(stem, "_regions.tsv"))
tmp_table <- paste0(table_path, ".tmp.", Sys.getpid())
fwrite(region_table, tmp_table, sep = "\t")
if (!file.rename(tmp_table, table_path)) stop("Could not publish region table.")

score_payload <- list(
    method = unname(method_labels[[opts$method]]),
    adaptation = scores$model,
    contrast = opts$contrast,
    seed = task_seed,
    alpha = opts$alpha,
    pc_levels = pc_levels,
    crop_lower_1based = lower,
    crop_upper_1based = crop_upper,
    crop_dim = crop_dim,
    gm_indices_in_crop = which(mask_crop),
    log_R_gm = as.numeric(scores$log_R[mask_crop]),
    log_R_til_gm = as.numeric(scores$log_R_til[mask_crop]),
    background_gm = as.numeric(scores$background[mask_crop]),
    elapsed_seconds = elapsed,
    fit = scores$fit,
    fit_structure = capture.output(str(
        scores$fit, max.level = 2L, give.attr = FALSE
    ))
)
saveRDS(
    score_payload,
    file.path(output_dir, "scores", paste0(stem, "_scores.rds")),
    compress = "xz"
)

for (pc_level_value in pc_levels) {
    selected <- region_table[
        abs(pc_level - pc_level_value) < 1e-12 & significant == 1L,
        region_id
    ]
    selected_map <- array(0L, dim = dim(z_full))
    selected_map[mask_full & atlas_full %in% selected] <- 1L
    pc_tag <- gsub("\\.", "p", sprintf("%.1f", pc_level_value))
    writeNifti(
        selected_map,
        file.path(
            output_dir, "maps",
            sprintf("%s__pc%s_selected.nii.gz", stem, pc_tag)
        ),
        template = z_path, datatype = "uint8"
    )
}

score_difference <- array(0, dim = dim(z_full))
score_difference_crop <- scores$log_R_til - scores$log_R
score_difference[
    crop[[1]], crop[[2]], crop[[3]]
] <- ifelse(mask_crop, score_difference_crop, 0)
writeNifti(
    score_difference,
    file.path(output_dir, "maps", paste0(stem, "__logRtil_minus_logR.nii.gz")),
    template = z_path, datatype = "float32"
)

summary_table <- data.table(
    method = unname(method_labels[[opts$method]]),
    adaptation = scores$model,
    contrast = opts$contrast,
    pc_level = pc_levels,
    alpha = opts$alpha,
    discoveries = vapply(
        pc_levels,
        function(level) sum(
            region_table$significant[abs(region_table$pc_level - level) < 1e-12]
        ),
        numeric(1L)
    ),
    n_regions = length(region_ids),
    elapsed_seconds = elapsed,
    seed = task_seed,
    crop_dim = paste(crop_dim, collapse = "x")
)
fwrite(
    summary_table,
    file.path(output_dir, "tables", paste0(stem, "_summary.tsv")),
    sep = "\t"
)
cat(sprintf(
    "COMPLETE task=%s elapsed=%.3f discoveries=%s\n",
    stem, elapsed,
    paste(summary_table$discoveries, collapse = ",")
))
