#!/usr/bin/env Rscript

parse_options <- function(args) {
    defaults <- list(
        L = 64L,
        repetitions = 100L,
        repetition_start = 1L,
        workers = 64L,
        mu_values = "2,2.5,3,3.5,4,4.5,5",
        point_values = "10,20,30",
        noise_types = "iid,grf",
        radius = 12,
        signal_fwhm = 24,
        noise_fwhm = 4,
        denoise_level = 1L,
        region_size = 8L,
        alpha = 0.1,
        pc_levels = "0.01,0.05,0.1,0.2,0.3,0.4,0.5",
        seed = 20260723L,
        claw_neighbor_range = 10L,
        claw_bandwidth = 5,
        fdrs_lambda = 0.1,
        methods = "co",
        task_limit = 0L,
        output_dir = file.path("outputs", "sim_3d", "common")
    )
    if (length(args) %% 2L != 0L) {
        stop("Arguments must be --name value pairs.")
    }
    for (i in seq(1L, length(args), by = 2L)) {
        key <- gsub("-", "_", sub("^--", "", args[[i]]))
        if (is.null(defaults[[key]])) stop("Unknown option: ", args[[i]])
        value <- args[[i + 1L]]
        if (is.integer(defaults[[key]])) value <- as.integer(value)
        if (is.numeric(defaults[[key]])) value <- as.numeric(value)
        defaults[[key]] <- value
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
        if (file.exists(file.path(candidate, "R", "working_models", "working_model_scores.R"))) return(candidate)
    }
    stop("Cannot locate SEFT_repro root.")
}

csv_numeric <- function(x) as.numeric(strsplit(x, ",", fixed = TRUE)[[1]])
csv_character <- function(x) strsplit(x, ",", fixed = TRUE)[[1]]

opts <- parse_options(commandArgs(TRUE))
repro_root <- find_repro_root()
setwd(repro_root)
environment_bin <- normalizePath(
    file.path(R.home(), "..", "..", "bin"), mustWork = FALSE
)
if (dir.exists(environment_bin)) {
    Sys.setenv(PATH = paste(
        environment_bin, Sys.getenv("PATH"), sep = .Platform$path.sep
    ))
}
Sys.setenv(SEFT_WORKING_MODEL_CPP = file.path(
    repro_root, "src", "working_models",
    "working_model_scores.cpp"
))

suppressPackageStartupMessages({
    library(data.table)
    library(ggplot2)
    library(neuRosim)
})
source(file.path("R", "core", "CLAW_functions.R"))
source(file.path(
    repro_root, "R", "working_models",
    "working_model_scores.R"
))

output_dir <- normalizePath(opts$output_dir, mustWork = FALSE)
result_dir <- file.path(output_dir, "results")
log_dir <- file.path(output_dir, "logs")
status_dir <- file.path(output_dir, "status")
dir.create(result_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(log_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(status_dir, recursive = TRUE, showWarnings = FALSE)

mu_values <- csv_numeric(opts$mu_values)
point_values <- as.integer(csv_numeric(opts$point_values))
noise_types <- csv_character(opts$noise_types)
pc_levels <- csv_numeric(opts$pc_levels)
methods <- csv_character(opts$methods)
if (!all(noise_types %in% c("iid", "grf"))) {
    stop("noise_types must contain only iid and/or grf.")
}
if (!identical(methods, "co")) {
    stop("The common runner uses methods=co; adapted models run separately.")
}

task_grid <- CJ(
    noise = noise_types,
    mu = mu_values,
    num_points = point_values,
    repetition = seq.int(
        opts$repetition_start,
        length.out = opts$repetitions
    ),
    sorted = TRUE
)
task_grid[, task_id := .I]
if (opts$task_limit > 0L) {
    task_grid <- task_grid[seq_len(min(opts$task_limit, .N))]
}
fwrite(task_grid, file.path(output_dir, "task_grid.csv"))

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
    if (!is.null(scores$log_R) && !is.null(scores$log_R_til)) {
        log_pair <- matrix(
            c(scores$log_R[indices], scores$log_R_til[indices]),
            nrow = 2L, byrow = TRUE
        )
    } else {
        score_pair <- matrix(
            c(scores$R[indices], scores$R_til[indices]),
            nrow = 2L, byrow = TRUE
        )
        if (any(score_pair < 0) || any(!is.finite(score_pair))) {
            return(NA_real_)
        }
        log_pair <- log(score_pair)
    }
    maximum <- max(log_pair)
    if (!is.finite(maximum)) return(NA_real_)
    log_pair <- log_pair - maximum
    cal_ELIS_cpp(
        log_pair, u, method = "Sym_miny",
        til_mth = "neglog", signmin_flag = FALSE
    )
}

score_log_arrays <- function(scores) {
    if (!is.null(scores$log_R) && !is.null(scores$log_R_til)) {
        return(list(R = scores$log_R, R_til = scores$log_R_til))
    }
    list(
        R = log(scores$R),
        R_til = log(scores$R_til)
    )
}

score_raw_arrays <- function(scores) {
    if (!is.null(scores$log_R) && !is.null(scores$log_R_til)) {
        return(list(
            R = exp(scores$log_R),
            R_til = exp(scores$log_R_til)
        ))
    }
    list(R = scores$R, R_til = scores$R_til)
}

evaluate_method <- function(scores, theta, method, elapsed_seconds) {
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
            method = method,
            pc_level = pc_level,
            fdp = cal_FDP(set_truth, decision),
            msr = cal_MSR(set_truth, decision),
            discoveries = sum(decision),
            true_active_regions = sum(set_truth),
            elapsed_seconds = elapsed_seconds
        )
    }))
}

evaluate_classical <- function(x, theta) {
    voxel_p <- pmax(2 * (1 - pnorm(abs(x))), .Machine$double.xmin)
    rbindlist(lapply(pc_levels, function(pc_level) {
        truth <- p_simes <- p_fisher <- numeric(length(region_indices))
        for (i in seq_along(region_indices)) {
            indices <- region_indices[[i]]
            u <- max(1L, ceiling(length(indices) * pc_level))
            truth[[i]] <- as.integer(sum(theta[indices]) >= u)
            p_simes[[i]] <- min(1, Simes_PC_test_cpp(voxel_p[indices], u))
            p_fisher[[i]] <- min(1, Fisher_PC_test_cpp(voxel_p[indices], u))
        }
        rbindlist(lapply(list(`BH+Simes` = p_simes, `BH+Fisher` = p_fisher), function(p_values) {
            decision <- BH(p_values, opts$alpha)
            data.table(
                pc_level = pc_level,
                fdp = cal_FDP(truth, decision), msr = cal_MSR(truth, decision),
                discoveries = sum(decision), true_active_regions = sum(truth),
                elapsed_seconds = 0
            )
        }), idcol = "method")
    }))
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
    task_seed <- opts$seed + task_id * 1009L
    set.seed(task_seed)
    cat("Started", stem, format(started), "\n")
    print(task_row)

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
        noise <- spatialnoise(
            dim = rep(opts$L, 3L), sigma = 1, nscan = 1L,
            method = "gaussRF", FWHM = opts$noise_fwhm,
            verbose = FALSE
        )
        noise <- array(drop(noise), dim = rep(opts$L, 3L))
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
    h <- density(sample(c(x, mirror), 1000L), bw = "nrd0")$bw

    method_scores <- list()
    method_times <- numeric()
    if ("co" %in% methods) {
        tic <- proc.time()[["elapsed"]]
        method_scores[["SEFT-CO"]] <- cal_CLAW_scores_3d(
            x, mirror,
            2 * (1 - pnorm(abs(x))),
            2 * (1 - pnorm(abs(mirror))),
            h = h, bandwidth = opts$claw_bandwidth,
            lambda = 0.5, neighbor_range = opts$claw_neighbor_range
        )
        method_times[["SEFT-CO"]] <- proc.time()[["elapsed"]] - tic
    }

    if ("spatial_approx" %in% methods) {
        tic <- proc.time()[["elapsed"]]
        method_scores[["SEFT-SpatialFDR-WhittleApprox"]] <-
            cal_spatial_fdr_scores_3d_whittle_approx(
                x, mirror,
                signal_fwhm_grid = unique(c(
                    opts$signal_fwhm / 2, opts$signal_fwhm,
                    opts$signal_fwhm * 1.5
                )),
                error_fwhm = opts$noise_fwhm
            )
        method_times[["SEFT-SpatialFDR-WhittleApprox"]] <-
            proc.time()[["elapsed"]] - tic
    }

    if ("fdrs_official" %in% methods) {
        tic <- proc.time()[["elapsed"]]
        method_scores[["SEFT-FDRSmoothing-Official"]] <-
            cal_fdr_smoothing_scores_3d(
                x, mirror, seed = task_seed + 1L
            )
        method_times[["SEFT-FDRSmoothing-Official"]] <-
            proc.time()[["elapsed"]] - tic
    }

    metrics <- rbindlist(lapply(names(method_scores), function(method) {
        evaluate_method(
            method_scores[[method]], signal$mask, method,
            method_times[[method]]
        )
    }))
    metrics <- rbindlist(list(metrics, evaluate_classical(x, signal$mask)), fill = TRUE)
    metrics[, `:=`(
        task_id = task_id,
        noise = task_row$noise[[1]],
        mu = task_row$mu[[1]],
        num_points = task_row$num_points[[1]],
        repetition = task_row$repetition[[1]]
    )]
    diagnostics <- rbindlist(lapply(names(method_scores), function(method) {
        scores <- method_scores[[method]]
        raw_scores <- score_raw_arrays(scores)
        log_scores <- score_log_arrays(scores)
        active <- signal$mask == 1
        data.table(
            method = method,
            finite_fraction = mean(
                is.finite(log_scores$R) & is.finite(log_scores$R_til)
            ),
            mean_real_score_signal = mean(raw_scores$R[active]),
            mean_real_score_null = mean(raw_scores$R[!active]),
            mean_pair_abs_difference = mean(abs(
                raw_scores$R - raw_scores$R_til
            )),
            mean_log_real_score_signal = mean(log_scores$R[active]),
            mean_log_real_score_null = mean(log_scores$R[!active]),
            mean_pair_abs_log_difference = mean(abs(
                log_scores$R - log_scores$R_til
            )),
            exact_log_tie_fraction_signal = mean(
                log_scores$R[active] == log_scores$R_til[active]
            ),
            both_below_1e10_fraction_signal = mean(
                log_scores$R[active] <= log(1e-10) &
                    log_scores$R_til[active] <= log(1e-10)
            ),
            task_id = task_id,
            noise = task_row$noise[[1]],
            mu = task_row$mu[[1]],
            num_points = task_row$num_points[[1]],
            repetition = task_row$repetition[[1]]
        )
    }))
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
    "Starting", nrow(task_grid), "tasks with", workers_used,
    "workers at", format(Sys.time()), "\n"
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
    stop(
        "Failed tasks: ", paste(failed_ids, collapse = ", "),
        ". Re-run the same command after inspecting per-task logs."
    )
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

plot_data <- melt(
    summary_table,
    id.vars = c("noise", "mu", "num_points", "method", "pc_level"),
    measure.vars = c("mean_fdp", "mean_msr"),
    variable.name = "metric", value.name = "value"
)
plot_data[, metric := fifelse(
    metric == "mean_fdp", "Mean FDP", "Mean missed-set rate"
)]
comparison_plot <- ggplot(
    plot_data,
    aes(mu, value, colour = method, group = method)
) +
    geom_hline(
        data = data.frame(metric = "Mean FDP", y = opts$alpha),
        aes(yintercept = y), colour = "grey45", linetype = 2
    ) +
    geom_line() +
    geom_point(size = 1) +
    facet_grid(
        metric + noise ~ num_points + pc_level,
        scales = "free_y", labeller = label_both
    ) +
    scale_y_continuous(limits = c(0, NA)) +
    labs(x = expression(mu), y = NULL, colour = NULL) +
    theme_bw(base_size = 7) +
    theme(legend.position = "bottom")
ggsave(
    file.path(output_dir, "original_setting_comparison.png"),
    comparison_plot, width = 18, height = 8, dpi = 180
)

writeLines(c(
    capture.output(str(opts)),
    paste("tasks:", nrow(task_grid)),
    paste("workers_used:", workers_used),
    paste("R_version:", R.version.string),
    paste("completed_at:", format(Sys.time(), tz = "UTC"))
), file.path(output_dir, "run_config.txt"))
writeLines("COMPLETE", file.path(output_dir, "RUN_COMPLETE"))
cat("All tasks completed at", format(Sys.time()), "\n")
