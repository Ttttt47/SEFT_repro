#!/usr/bin/env Rscript

parse_options <- function(args) {
    defaults <- list(
        method = "",
        workers = 1L,
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
        deepfdr_epochs = 17L,
        deepfdr_channels = 64L,
        deepfdr_device = "cuda",
        fchmrf_em_steps = 5L,
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
    if (!defaults$method %in% c("deepfdr", "fchmrf")) {
        stop("--method must be deepfdr or fchmrf.")
    }
    if (!nzchar(defaults$output_dir)) {
        defaults$output_dir <- file.path(
            "outputs", "sim_3d", defaults$method
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
if (opts$method == "fchmrf") {
    stop(
        "fcHMRF cannot be run faithfully on this z-only grid: the required ",
        "beta/delta-mu map and the authors' scalable paper runner are absent."
    )
}
repro_root <- find_repro_root()
setwd(repro_root)
environment_bin <- normalizePath(
    file.path(R.home(), "..", "..", "bin"), mustWork = FALSE
)
Sys.setenv(
    PATH = paste(environment_bin, Sys.getenv("PATH"),
                 sep = .Platform$path.sep),
    OMP_NUM_THREADS = if (opts$method == "fchmrf") "2" else "4",
    MKL_NUM_THREADS = if (opts$method == "fchmrf") "2" else "4",
    SEFT_WORKING_MODEL_CPP = file.path(
        repro_root, "src", "working_models",
        "working_model_scores.cpp"
    ),
    SEFT_ML_WORKING_MODEL_SCRIPT = file.path(
        repro_root, "python", "working_models",
        "ml_working_model.py"
    ),
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

original_dir <- normalizePath(opts$original_dir, mustWork = TRUE)
task_grid <- fread(file.path(original_dir, "task_grid.csv"))
setorder(task_grid, task_id)
if (nzchar(opts$task_ids)) {
    requested_task_ids <- as.integer(strsplit(
        opts$task_ids, ",", fixed = TRUE
    )[[1]])
    if (anyNA(requested_task_ids)) {
        stop("--task-ids must be a comma-separated list of integers.")
    }
    missing_task_ids <- setdiff(requested_task_ids, task_grid$task_id)
    if (length(missing_task_ids)) {
        stop(
            "Unknown task IDs: ",
            paste(missing_task_ids, collapse = ", ")
        )
    }
    task_grid <- task_grid[task_id %in% requested_task_ids]
    setorder(task_grid, task_id)
}
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
fwrite(task_grid, file.path(output_dir, "task_grid.csv"))
pc_levels <- as.numeric(strsplit(opts$pc_levels, ",", fixed = TRUE)[[1]])

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
    score_pair <- matrix(
        c(scores$R[indices], scores$R_til[indices]),
        nrow = 2L, byrow = TRUE
    )
    maximum <- max(score_pair)
    if (!is.finite(maximum) || maximum <= 0) return(NA_real_)
    cal_ELIS_cpp(
        log(score_pair / maximum),
        u,
        method = "Sym_miny",
        til_mth = "neglog",
        signmin_flag = FALSE
    )
}

evaluate_scores <- function(scores, theta, elapsed_seconds) {
    rbindlist(lapply(pc_levels, function(pc_level) {
        set_truth <- e_values <- numeric(length(region_indices))
        for (i in seq_along(region_indices)) {
            indices <- region_indices[[i]]
            u <- max(1L, ceiling(length(indices) * pc_level))
            set_truth[[i]] <- as.integer(sum(theta[indices]) >= u)
            e_values[[i]] <- seft_evalue(scores, indices, u)
        }
        p_values <- pmin(1, 1 / e_values)
        p_values[!is.finite(p_values)] <- 1
        decision <- BH(p_values, opts$alpha)
        data.table(
            method = if (opts$method == "deepfdr") {
                "SEFT-DeepFDR-PaperScale"
            } else {
                "SEFT-fcHMRF"
            },
            pc_level = pc_level,
            fdp = cal_FDP(set_truth, decision),
            msr = cal_MSR(set_truth, decision),
            discoveries = sum(decision),
            true_active_regions = sum(set_truth),
            elapsed_seconds = elapsed_seconds
        )
    }))
}

simulate_task <- function(task_row) {
    task_id <- task_row$task_id[[1]]
    task_seed <- opts$seed + task_id * 1009L
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
    mirror <- array(rnorm(opts$L^3), dim = rep(opts$L, 3L))
    x[x == 0] <- 1e-10
    mirror[mirror == 0] <- 1e-10
    list(
        x = x,
        mirror = mirror,
        theta = signal$mask,
        seed = task_seed
    )
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

    tic <- proc.time()[["elapsed"]]
    scores <- if (opts$method == "deepfdr") {
        cal_deepfdr_scores_3d(
            case$x,
            case$mirror,
            seed = 0L,
            epochs = opts$deepfdr_epochs,
            channels = opts$deepfdr_channels,
            device = opts$deepfdr_device
        )
    } else {
        cal_fchmrf_scores_3d(
            case$x,
            case$mirror,
            seed = case$seed + 37L,
            em_steps = opts$fchmrf_em_steps
        )
    }
    elapsed_seconds <- proc.time()[["elapsed"]] - tic
    metrics <- evaluate_scores(scores, case$theta, elapsed_seconds)
    metrics[, `:=`(
        task_id = task_id,
        noise = task_row$noise[[1]],
        mu = task_row$mu[[1]],
        num_points = task_row$num_points[[1]],
        repetition = task_row$repetition[[1]]
    )]
    active <- case$theta == 1
    fit_value <- function(name, default = NA) {
        value <- scores$fit[[name]]
        if (is.null(value) || !length(value)) default else value
    }
    loss_trace <- scores$fit$loss_trace
    final_encoder_loss <- final_decoder_loss <- NA_real_
    if (is.data.frame(loss_trace) && nrow(loss_trace)) {
        final_encoder_loss <- tail(loss_trace$encoder, 1L)
        final_decoder_loss <- tail(loss_trace$decoder, 1L)
    }
    context_range <- fit_value("context_signal_range", c(NA_real_, NA_real_))
    diagnostics <- data.table(
        method = metrics$method[[1]],
        finite_fraction = mean(
            is.finite(scores$R) & is.finite(scores$R_til)
        ),
        mean_real_score_signal = mean(scores$R[active]),
        mean_real_score_null = mean(scores$R[!active]),
        mean_pair_abs_difference = mean(abs(scores$R - scores$R_til)),
        backend_elapsed_seconds = scores$fit$elapsed_seconds,
        orientation_flipped = as.logical(fit_value(
            "orientation_flipped", NA
        )),
        dice_raw_as_lis = as.numeric(fit_value(
            "dice_raw_as_lis", NA_real_
        )),
        dice_flipped_as_lis = as.numeric(fit_value(
            "dice_flipped_as_lis", NA_real_
        )),
        qvalue_reference_discoveries = as.integer(fit_value(
            "qvalue_reference_discoveries", NA_integer_
        )),
        context_signal_min = as.numeric(context_range[[1]]),
        context_signal_max = as.numeric(context_range[[2]]),
        final_encoder_loss = final_encoder_loss,
        final_decoder_loss = final_decoder_loss,
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
    stop("Failed tasks: ", paste(failed_ids, collapse = ", "))
}

all_metrics <- rbindlist(lapply(results, `[[`, "metrics"))
all_diagnostics <- rbindlist(lapply(results, `[[`, "diagnostics"))
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
    paste("tasks:", nrow(task_grid)),
    paste("workers_used:", workers_used),
    paste("completed_at:", format(Sys.time(), tz = "UTC"))
), file.path(output_dir, "run_config.txt"))
writeLines("COMPLETE", file.path(output_dir, "RUN_COMPLETE"))
cat("All", opts$method, "tasks completed at", format(Sys.time()), "\n")
