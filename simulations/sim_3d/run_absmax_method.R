#!/usr/bin/env Rscript

parse_options <- function(args) {
    defaults <- list(
        method = "",
        workers = 64L,
        gpu_workers = 1L,
        task_limit = 0L,
        task_ids = "",
        seed = 20260723L,
        L = 64L,
        radius = 12,
        signal_fwhm = 24,
        noise_fwhm = 4,
        denoise_level = 1L,
        region_size = 8L,
        alpha = 0.1,
        pc_levels = "0.01,0.05,0.1,0.2,0.3,0.4,0.5",
        ising_iter_max = 20L,
        ising_sweep_b = 1L,
        ising_sweep_r = 2L,
        ising_burnin_lis = 20L,
        ising_sweep_lis = 50L,
        ising_n_chains = 2L,
        ising_fixing_beta = 0L,
        ising_fit_attempts = 1L,
        ising_basic_convergence_tol = Inf,
        ising_basic_convergence_max = Inf,
        ising_require_basic_convergence = 0L,
        paired_mu_seed = 0L,
        mirror_mode = "iid",
        original_dir = file.path("outputs", "sim_3d", "common"),
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
    allowed <- c(
        "fdrs_absmax", "deepfdr_absmax", "fchmrf_absmax",
        "ising_absmax"
    )
    if (!defaults$method %in% allowed) {
        stop("--method must be one of: ", paste(allowed, collapse = ", "))
    }
    if (!nzchar(defaults$output_dir)) {
        defaults$output_dir <- file.path(
            "outputs", "sim_3d",
            defaults$method
        )
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
if (!opts$mirror_mode %in% c("iid", "matched")) {
    stop("--mirror-mode must be iid or matched.")
}
repro_root <- find_repro_root()
setwd(repro_root)
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
    library(neuRosim)
})
source(file.path("R", "core", "CLAW_functions.R"))
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

original_dir <- normalizePath(opts$original_dir, mustWork = TRUE)
canonical_grid <- fread(file.path(original_dir, "task_grid.csv"))
setorder(canonical_grid, task_id)
canonical_repetitions <- sort(unique(canonical_grid$repetition))
if (!length(canonical_repetitions) ||
        anyNA(canonical_repetitions) ||
        anyDuplicated(canonical_repetitions)) {
    stop("Canonical task grid has invalid repetition labels.")
}
expected <- CJ(
    noise = c("iid", "grf"),
    mu = c(2, 2.5, 3, 3.5, 4, 4.5, 5),
    num_points = c(10L, 20L, 30L),
    repetition = canonical_repetitions,
    sorted = TRUE
)
setcolorder(expected, c("noise", "mu", "num_points", "repetition"))
observed <- copy(canonical_grid)[, .(noise, mu, num_points, repetition)]
if (opts$task_limit == 0L && (nrow(observed) != nrow(expected) ||
        !data.table::fsetequal(observed, expected))) {
    stop(
        "Canonical task grid is not a complete original-paper factorial grid."
    )
}
if (!identical(canonical_grid$task_id, seq_len(nrow(canonical_grid)))) {
    stop("Canonical task IDs must be consecutive and sorted.")
}

task_grid <- copy(canonical_grid)
if (nzchar(opts$task_ids)) {
    requested <- as.integer(strsplit(opts$task_ids, ",", fixed = TRUE)[[1]])
    if (anyNA(requested) || length(setdiff(requested, task_grid$task_id))) {
        stop("--task-ids contains invalid IDs.")
    }
    task_grid <- task_grid[task_id %in% requested]
}
# IID is deliberately scheduled first. Task IDs themselves remain canonical.
task_grid[, schedule_noise := match(noise, c("iid", "grf"))]
setorder(task_grid, schedule_noise, num_points, mu, repetition)
task_grid[, schedule_noise := NULL]
if (opts$task_limit > 0L) {
    task_grid <- task_grid[seq_len(min(opts$task_limit, .N))]
}

output_dir <- normalizePath(opts$output_dir, mustWork = FALSE)
result_dir <- file.path(output_dir, "results")
log_dir <- file.path(output_dir, "logs")
status_dir <- file.path(output_dir, "status")
dir.create(result_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(log_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(status_dir, recursive = TRUE, showWarnings = FALSE)
fwrite(canonical_grid, file.path(output_dir, "task_grid.csv"))
pc_levels <- as.numeric(strsplit(opts$pc_levels, ",", fixed = TRUE)[[1]])

model_label <- switch(
    opts$method,
    fdrs_absmax = "SEFT-FDRSmoothing-AbsMaxAdapt",
    deepfdr_absmax = "SEFT-DeepFDR-AbsMaxAdapt",
    fchmrf_absmax = "SEFT-fcHMRF-AbsMaxAdapt",
    ising_absmax = "SEFT-Ising-AbsMaxAdapt"
)
writeLines("RUNNING", file.path(output_dir, "RUN_STATE"))
writeLines(c(
    capture.output(str(opts)),
    paste("model_label:", model_label),
    paste("canonical_tasks:", nrow(canonical_grid)),
    paste("selected_tasks:", nrow(task_grid)),
    paste("requested_workers:", opts$workers),
    paste("iid_scheduled_first:", TRUE),
    paste("started_at:", format(Sys.time(), tz = "UTC"))
), file.path(output_dir, "run_config.txt"))

build_region_mask <- function(L, region_size) {
    grid <- array(0L, dim = rep(L, 3L))
    n_axis <- ceiling(L / region_size)
    for (i in seq_len(n_axis)) for (j in seq_len(n_axis)) {
        for (k in seq_len(n_axis)) {
            grid[
                ((i - 1L) * region_size + 1L):min(i * region_size, L),
                ((j - 1L) * region_size + 1L):min(j * region_size, L),
                ((k - 1L) * region_size + 1L):min(k * region_size, L)
            ] <- i + (j - 1L) * n_axis + (k - 1L) * n_axis^2
        }
    }
    grid
}
region_mask <- build_region_mask(opts$L, opts$region_size)
region_ids <- sort(unique(as.integer(region_mask)))
region_indices <- lapply(region_ids, function(id) which(region_mask == id))

seft_evalue <- function(scores, indices, u) {
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

evaluate_scores <- function(scores, theta, elapsed_seconds) {
    rbindlist(lapply(pc_levels, function(pc_level) {
        set_truth <- e_values <- numeric(length(region_indices))
        signal_counts <- integer(length(region_indices))
        for (i in seq_along(region_indices)) {
            indices <- region_indices[[i]]
            u <- max(1L, ceiling(length(indices) * pc_level))
            signal_counts[[i]] <- sum(theta[indices])
            set_truth[[i]] <- as.integer(signal_counts[[i]] >= u)
            e_values[[i]] <- seft_evalue(scores, indices, u)
        }
        p_values <- pmin(1, 1 / e_values)
        p_values[!is.finite(p_values)] <- 1
        decision <- BH(p_values, opts$alpha)
        null <- set_truth == 0
        false_discovery <- decision == 1 & null
        data.table(
            method = model_label,
            pc_level = pc_level,
            fdp = cal_FDP(set_truth, decision),
            msr = cal_MSR(set_truth, decision),
            discoveries = sum(decision),
            false_discoveries = sum(false_discovery),
            false_discoveries_boundary = sum(
                false_discovery & signal_counts > 0
            ),
            false_discoveries_pure_null = sum(
                false_discovery & signal_counts == 0
            ),
            max_signal_count_false_discovery = if (any(false_discovery)) {
                max(signal_counts[false_discovery])
            } else {
                0L
            },
            null_evalue_q99 = as.numeric(quantile(
                e_values[null], 0.99, na.rm = TRUE, names = FALSE
            )),
            null_evalue_max = max(e_values[null], na.rm = TRUE),
            true_active_regions = sum(set_truth),
            elapsed_seconds = elapsed_seconds
        )
    }))
}

simulate_task <- function(task_row) {
    task_id <- task_row$task_id[[1]]
    task_seed <- if (opts$paired_mu_seed == 1L) {
        opts$seed +
            match(task_row$noise[[1]], c("grf", "iid")) * 100000L +
            task_row$num_points[[1]] * 100L +
            task_row$repetition[[1]]
    } else {
        opts$seed + task_id * 1009L
    }
    set.seed(task_seed)
    centers <- cbind(
        sample.int(opts$L, task_row$num_points[[1]]),
        sample.int(opts$L, task_row$num_points[[1]]),
        sample.int(opts$L, task_row$num_points[[1]])
    )
    signal <- generate_3d_signal(
        dim = rep(opts$L, 3L),
        mu = task_row$mu[[1]],
        shape = "sphere",
        radius = opts$radius,
        coord = centers,
        decay_method = "gaussian",
        mix_method = "max",
        FWHM = opts$signal_fwhm
    )
    if (task_row$noise[[1]] == "iid") {
        noise <- array(rnorm(opts$L^3), dim = rep(opts$L, 3L))
    } else {
        noise <- array(drop(spatialnoise(
            dim = rep(opts$L, 3L),
            sigma = 1,
            nscan = 1L,
            method = "gaussRF",
            FWHM = opts$noise_fwhm,
            verbose = FALSE
        )), dim = rep(opts$L, 3L))
    }
    x <- signal$signal + noise
    if (task_row$noise[[1]] == "grf" && opts$denoise_level > 0L) {
        x <- apply_wavelet_denoising_3d(
            x, wf = "d8", target_level = opts$denoise_level,
            J = 6, verbose = FALSE
        )
    }
    if (opts$mirror_mode == "matched" &&
            task_row$noise[[1]] == "grf") {
        mirror <- array(drop(spatialnoise(
            dim = rep(opts$L, 3L),
            sigma = 1,
            nscan = 1L,
            method = "gaussRF",
            FWHM = opts$noise_fwhm,
            verbose = FALSE
        )), dim = rep(opts$L, 3L))
        if (opts$denoise_level > 0L) {
            mirror <- apply_wavelet_denoising_3d(
                mirror, wf = "d8", target_level = opts$denoise_level,
                J = 6, verbose = FALSE
            )
        }
    } else {
        mirror <- array(rnorm(opts$L^3), dim = rep(opts$L, 3L))
    }
    x[x == 0] <- 1e-10
    mirror[mirror == 0] <- 1e-10
    list(x = x, mirror = mirror, theta = signal$mask, seed = task_seed)
}

fit_value <- function(fit, name, default = NA_real_) {
    value <- fit[[name]]
    if (is.null(value) || !length(value)) default else value
}

run_task <- function(task_row) {
    task_id <- task_row$task_id[[1]]
    stem <- sprintf("task_%04d", task_id)
    result_file <- file.path(result_dir, paste0(stem, ".rds"))
    status_file <- file.path(status_dir, paste0(stem, ".status"))
    log_file <- file.path(log_dir, paste0(stem, ".log"))
    if (file.exists(result_file) && file.exists(status_file) &&
            identical(readLines(status_file, warn = FALSE)[[1]], "COMPLETE")) {
        return(readRDS(result_file))
    }

    log_connection <- file(log_file, open = "wt")
    sink(log_connection, split = TRUE)
    sink(log_connection, type = "message")
    on.exit({
        sink(type = "message")
        sink()
        close(log_connection)
    }, add = TRUE)
    writeLines("RUNNING", status_file)
    started <- Sys.time()
    cat("Started", stem, format(started), "\n")
    print(task_row)
    case <- simulate_task(task_row)
    if (opts$method == "deepfdr_absmax") {
        gpu_ids <- strsplit(Sys.getenv("SEFT_GPU_IDS", unset = ""), ",", fixed = TRUE)[[1]]
        gpu_ids <- gpu_ids[nzchar(gpu_ids)]
        if (length(gpu_ids)) {
            Sys.setenv(CUDA_VISIBLE_DEVICES = gpu_ids[[(task_id - 1L) %% length(gpu_ids) + 1L]])
        }
    }

    tic <- proc.time()[["elapsed"]]
    scores <- switch(
        opts$method,
        fdrs_absmax = cal_fdr_smoothing_scores_3d_absmax(
            case$x, case$mirror, seed = case$seed + 1L,
            solver_mode = "grid_primal_dual", gfl_threads = 1L,
            pr_table_size = 65536L
        ),
        deepfdr_absmax = cal_deepfdr_scores_3d_absmax(
            case$x, case$mirror, seed = 0L,
            device = Sys.getenv("SEFT_DEEPFDR_DEVICE", unset = "auto")
        ),
        fchmrf_absmax = cal_fchmrf_scores_3d_absmax(
            case$x, case$mirror,
            appearance_mode = "background",
            seed = case$seed + 37L
        ),
        ising_absmax = cal_ising_scores_3d_absmax(
            case$x, case$mirror,
            # Weak, sparse signals can place the fitted finite Ising model
            # near competing occupancy phases.  We retain that free-beta fit
            # and its convergence/chain diagnostics; it is working-model
            # instability, not a reason to alter beta or the target.
            seed = case$seed + 53L,
            iter_max = opts$ising_iter_max,
            sweep_b = opts$ising_sweep_b,
            sweep_r = opts$ising_sweep_r,
            burnin_lis = opts$ising_burnin_lis,
            sweep_lis = opts$ising_sweep_lis,
            n_chains = opts$ising_n_chains,
            fixing_beta = opts$ising_fixing_beta == 1L,
            fit_attempts = opts$ising_fit_attempts,
            basic_convergence_tol = opts$ising_basic_convergence_tol,
            basic_convergence_max = opts$ising_basic_convergence_max,
            require_basic_convergence =
                opts$ising_require_basic_convergence == 1L
        )
    )
    elapsed_seconds <- proc.time()[["elapsed"]] - tic
    if (!identical(scores$model, switch(
        opts$method,
        fdrs_absmax = "fdr_smoothing_absmax_grid_primal_dual_adaptation",
        deepfdr_absmax = "deepfdr_absmax_adaptation",
        fchmrf_absmax = "fchmrf_absmax_adaptation",
        ising_absmax = "ising_absmax_plis_adaptation"
    ))) stop("Unexpected model identity from score interface.")
    if (any(!is.finite(scores$log_R)) ||
            any(!is.finite(scores$log_R_til))) {
        stop("Non-finite log scores.")
    }

    metrics <- evaluate_scores(scores, case$theta, elapsed_seconds)
    metrics[, `:=`(
        task_id = task_id,
        noise = task_row$noise[[1]],
        mu = task_row$mu[[1]],
        num_points = task_row$num_points[[1]],
        repetition = task_row$repetition[[1]]
    )]
    active <- case$theta == 1
    log_difference <- abs(scores$log_R - scores$log_R_til)
    diagnostics <- data.table(
        method = model_label,
        finite_fraction = mean(
            is.finite(scores$log_R) & is.finite(scores$log_R_til)
        ),
        mean_log_real_score_signal = mean(scores$log_R[active]),
        mean_log_real_score_null = mean(scores$log_R[!active]),
        median_log_real_score_signal = median(scores$log_R[active]),
        median_log_real_score_null = median(scores$log_R[!active]),
        mean_pair_abs_log_difference = mean(log_difference),
        exact_log_tie_fraction_signal = mean(log_difference[active] == 0),
        background_unchanged = identical(
            as.double(scores$background),
            as.double(make_plis_background(case$x, case$mirror))
        ),
        backend_elapsed_seconds = as.numeric(fit_value(
            scores$fit, "elapsed_seconds"
        )),
        kde_grid_size = as.integer(fit_value(
            scores$fit, "kde_grid_size", NA_integer_
        )),
        kde_max_log_error_f1 = as.numeric(fit_value(
            scores$fit, "kde_max_log_error_f1"
        )),
        kde_max_log_error_g1 = as.numeric(fit_value(
            scores$fit, "kde_max_log_error_g1"
        )),
        ising_beta = as.numeric(fit_value(scores$fit, "beta")),
        ising_h = as.numeric(fit_value(scores$fit, "h")),
        ising_mu = paste(
            as.numeric(fit_value(scores$fit, "mu")), collapse = ";"
        ),
        ising_sig2 = paste(
            as.numeric(fit_value(scores$fit, "sig2")), collapse = ";"
        ),
        ising_gamma_min = as.numeric(fit_value(
            scores$fit, "gamma_min"
        )),
        ising_gamma_max = as.numeric(fit_value(
            scores$fit, "gamma_max"
        )),
        ising_gamma_mean = as.numeric(fit_value(
            scores$fit, "gamma_mean"
        )),
        ising_gamma_chain_range = as.numeric(fit_value(
            scores$fit, "gamma_chain_range"
        )),
        ising_gem_iterations = as.integer(fit_value(
            scores$fit, "gem_iterations", NA_integer_
        )),
        ising_gem_last_rel = as.numeric(fit_value(
            scores$fit, "gem_last_rel"
        )),
        ising_gem_converged = as.logical(fit_value(
            scores$fit, "gem_converged", NA
        )),
        ising_gem_basic_converged = as.logical(fit_value(
            scores$fit, "gem_basic_converged", NA
        )),
        ising_gem_selected_attempt = as.integer(fit_value(
            scores$fit, "gem_selected_attempt", NA_integer_
        )),
        ising_gem_fit_attempts_used = as.integer(fit_value(
            scores$fit, "gem_fit_attempts_used", NA_integer_
        )),
        ising_gem_tail5_median_rel = as.numeric(fit_value(
            scores$fit, "gem_tail5_median_rel"
        )),
        ising_gem_tail5_max_rel = as.numeric(fit_value(
            scores$fit, "gem_tail5_max_rel"
        )),
        task_id = task_id,
        noise = task_row$noise[[1]],
        mu = task_row$mu[[1]],
        num_points = task_row$num_points[[1]],
        repetition = task_row$repetition[[1]]
    )
    out <- list(metrics = metrics, diagnostics = diagnostics)
    saveRDS(out, result_file, compress = FALSE)
    writeLines("COMPLETE", status_file)
    cat(
        "Completed", stem, "in",
        round(difftime(Sys.time(), started, units = "secs"), 2), "seconds\n"
    )
    out
}

run_task_safe <- function(requested_task_id) {
    task_row <- task_grid[task_id == requested_task_id]
    tryCatch(
        run_task(task_row),
        error = function(error) {
            stem <- sprintf("task_%04d", requested_task_id)
            writeLines(
                paste("FAILED", conditionMessage(error)),
                file.path(status_dir, paste0(stem, ".status"))
            )
            structure(
                list(
                    task_id = requested_task_id,
                    message = conditionMessage(error)
                ),
                class = "simulation_task_error"
            )
        }
    )
}

workers_used <- min(opts$workers, nrow(task_grid))
cat(
    "Starting", opts$method, "for", nrow(task_grid), "tasks with",
    workers_used, "workers at", format(Sys.time()), "\n"
)
results <- parallel::mclapply(
    task_grid$task_id,
    run_task_safe,
    mc.cores = workers_used,
    mc.preschedule = FALSE
)
failed <- vapply(results, inherits, logical(1), "simulation_task_error")
if (any(failed)) {
    failed_ids <- vapply(results[failed], `[[`, integer(1), "task_id")
    writeLines(as.character(failed_ids), file.path(output_dir, "failed_tasks.txt"))
    writeLines("FAILED", file.path(output_dir, "RUN_STATE"))
    stop(
        "Failed tasks: ", paste(failed_ids, collapse = ", "),
        ". Re-run the same command to resume after inspection."
    )
}

all_metrics <- rbindlist(lapply(results, `[[`, "metrics"))
all_diagnostics <- rbindlist(lapply(results, `[[`, "diagnostics"))
setorder(all_metrics, task_id, pc_level)
setorder(all_diagnostics, task_id)
summary_table <- all_metrics[, .(
    mean_fdp = mean(fdp),
    se_fdp = sd(fdp) / sqrt(.N),
    mean_msr = mean(msr),
    se_msr = sd(msr) / sqrt(.N),
    mean_discoveries = mean(discoveries),
    mean_elapsed_seconds = mean(elapsed_seconds),
    repetitions = .N
), by = .(noise, mu, num_points, method, pc_level)]
fwrite(all_metrics, file.path(output_dir, "per_replication_metrics.csv"))
fwrite(all_diagnostics, file.path(output_dir, "score_diagnostics.csv"))
fwrite(summary_table, file.path(output_dir, "summary.csv"))
writeLines(c(
    capture.output(str(opts)),
    paste("model_label:", model_label),
    paste("canonical_tasks:", nrow(canonical_grid)),
    paste("selected_tasks:", nrow(task_grid)),
    paste("workers_used:", workers_used),
    paste("iid_scheduled_first:", TRUE),
    paste("completed_at:", format(Sys.time(), tz = "UTC"))
), file.path(output_dir, "run_config.txt"))
writeLines("COMPLETE", file.path(output_dir, "RUN_COMPLETE"))
writeLines("COMPLETE", file.path(output_dir, "RUN_STATE"))
cat("All", opts$method, "tasks completed at", format(Sys.time()), "\n")
