#!/usr/bin/env Rscript

# Resumable paper-scale Ising simulation for SEFT-Ising. Run from any directory.

parse_options <- function(args) {
  defaults <- list(
    workers = 64L,
    output_dir = "",
    task_limit = 0L,
    task_ids = "",
    seed_base = 20260731L,
    L = 64L,
    beta = 0.8,
    B = 100L,
    iter_max = 1000L,
    sweep_b = 20L,
    sweep_r = 40L,
    burnin_lis = 1000L,
    sweep_lis = 2000L,
    n_chains = 2L,
    data_burnin = 1000L,
    data_sweeps = 500L,
    region_size = 8L,
    alpha = 0.10,
    pc_levels = "0.01,0.05,0.1,0.2,0.3,0.4,0.5"
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
  defaults
}

find_repro_root <- function() {
  file_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  candidates <- c(
    getwd(),
    if (length(file_arg)) file.path(
      dirname(sub("^--file=", "", file_arg[[1]])), "..", ".."
    )
  )
  for (candidate in candidates) {
    candidate <- normalizePath(candidate, mustWork = FALSE)
    if (file.exists(file.path(candidate, "R", "core", "Ising_functions.R"))) {
      return(candidate)
    }
  }
  stop("Cannot locate SEFT_repro.")
}

opts <- parse_options(commandArgs(TRUE))
if (opts$workers < 1L || opts$L < 8L || opts$region_size < 1L || opts$L %% opts$region_size != 0L) {
  stop("Require workers >= 1, L >= 8, and region_size to divide L.")
}
repro_root <- find_repro_root()
setwd(repro_root)
if (!nzchar(opts$output_dir)) {
  opts$output_dir <- file.path(
    repro_root, "outputs", "sim_ising"
  )
}
output_dir <- normalizePath(opts$output_dir, mustWork = FALSE)
result_dir <- file.path(output_dir, "results")
log_dir <- file.path(output_dir, "logs")
status_dir <- file.path(output_dir, "status")
legacy_is_dir <- file.path(output_dir, "legacy", "seft_is")
legacy_co_dir <- file.path(output_dir, "legacy", "seft_co")
for (path in c(result_dir, log_dir, status_dir, legacy_is_dir, legacy_co_dir)) {
  dir.create(path, recursive = TRUE, showWarnings = FALSE)
}

environment_bin <- normalizePath(
  file.path(R.home(), "..", "..", "bin"), mustWork = FALSE
)
Sys.setenv(
  PATH = paste(environment_bin, Sys.getenv("PATH"), sep = .Platform$path.sep),
  OMP_NUM_THREADS = "1",
  MKL_NUM_THREADS = "1",
  OPENBLAS_NUM_THREADS = "1",
  NUMEXPR_NUM_THREADS = "1",
  VECLIB_MAXIMUM_THREADS = "1",
  TORCH_NUM_THREADS = "1"
)
suppressPackageStartupMessages({
  library(data.table)
  library(digest)
})
source(file.path("R", "core", "Ising_functions.R"))

h_values <- c(-2.5, -2.4, -2.3)
mu_values <- seq(1.5, 4, by = 0.5)
pc_levels <- as.numeric(strsplit(opts$pc_levels, ",", fixed = TRUE)[[1]])
if (length(pc_levels) != 7L || any(!is.finite(pc_levels))) {
  stop("The paper run requires seven finite PC levels.")
}

task_grid <- CJ(
  h0 = h_values,
  mu = mu_values,
  repetition = seq_len(opts$B),
  sorted = TRUE
)
task_grid[, cell_id :=
  (match(h0, h_values) - 1L) * length(mu_values) + match(mu, mu_values)]
task_grid[, task_id := .I]
task_grid[, data_seed := repetition]
task_grid[, mirror_seed := as.integer(opts$seed_base + cell_id * 1000L + repetition)]
task_grid[, bandwidth_seed := as.integer(mirror_seed + 100000000L)]
if (anyDuplicated(task_grid$mirror_seed)) stop("Mirror seeds are not unique.")
fwrite(task_grid, file.path(output_dir, "task_grid.csv"))
fwrite(
  task_grid[, .(task_id, h0, mu, repetition, data_seed, mirror_seed, bandwidth_seed)],
  file.path(output_dir, "seed_manifest.csv")
)

selected_grid <- copy(task_grid)
if (nzchar(opts$task_ids)) {
  requested <- as.integer(strsplit(opts$task_ids, ",", fixed = TRUE)[[1]])
  if (anyNA(requested) || length(setdiff(requested, task_grid$task_id))) {
    stop("--task-ids contains invalid task IDs.")
  }
  selected_grid <- selected_grid[task_id %in% requested]
}
if (opts$task_limit > 0L) {
  selected_grid <- selected_grid[seq_len(min(opts$task_limit, .N))]
}
if (!nrow(selected_grid)) stop("No tasks selected.")

writeLines("RUNNING", file.path(output_dir, "RUN_STATE"))
writeLines(c(
  capture.output(str(opts)),
  paste("canonical_tasks:", nrow(task_grid)),
  paste("selected_tasks:", nrow(selected_grid)),
  paste("workers_used:", min(opts$workers, nrow(selected_grid))),
  paste("candidate_emission: ordinary f0/f1"),
  paste("baseline_emission: abs-max g0/g1"),
  paste("started_at_utc:", format(Sys.time(), tz = "UTC"))
), file.path(output_dir, "run_config.txt"))

build_region_indices <- function(L, region_size) {
  mask <- array(0L, dim = rep(L, 3L))
  n_axis <- ceiling(L / region_size)
  for (i in seq_len(n_axis)) for (j in seq_len(n_axis)) for (k in seq_len(n_axis)) {
    mask[
      ((i - 1L) * region_size + 1L):min(i * region_size, L),
      ((j - 1L) * region_size + 1L):min(j * region_size, L),
      ((k - 1L) * region_size + 1L):min(k * region_size, L)
    ] <- i + (j - 1L) * n_axis + (k - 1L) * n_axis^2
  }
  lapply(sort(unique(as.integer(mask))), function(id) which(mask == id))
}

region_indices <- build_region_indices(opts$L, opts$region_size)
mask_full <- array(TRUE, dim = rep(opts$L, 3L))
nb <- hmrf_build_neighbors_6n(mask_full)
eps <- 1e-32

result_path <- function(task_id) file.path(result_dir, sprintf("task_%04d.rds", task_id))
status_path <- function(task_id) file.path(status_dir, sprintf("task_%04d.status", task_id))
log_path <- function(task_id) file.path(log_dir, sprintf("task_%04d.log", task_id))

result_valid <- function(path, expected_task_id) {
  if (!file.exists(path)) return(FALSE)
  object <- tryCatch(readRDS(path), error = function(e) NULL)
  is.list(object) && identical(as.integer(object$task$task_id), as.integer(expected_task_id)) &&
    is.data.frame(object$metrics) && nrow(object$metrics) == 4L * length(pc_levels) &&
    all(is.finite(object$metrics$fdr)) && all(is.finite(object$metrics$fnr)) &&
    isTRUE(object$diagnostics$scores_finite_nonzero)
}

pc_evalue <- function(score, mirror_score, indices, u) {
  log_pair <- matrix(
    c(log(pmax(score[indices], eps)), log(pmax(mirror_score[indices], eps))),
    nrow = 2L, byrow = TRUE
  )
  maximum <- max(log_pair)
  cal_ELIS_cpp(
    log_pair - maximum, u,
    method = "Sym_miny", til_mth = "neglog", signmin_flag = FALSE
  )
}

evaluate_methods <- function(theta, x, is_scores, co_scores) {
  p_voxel <- pmax(2 * (1 - pnorm(abs(x))), .Machine$double.xmin)
  rbindlist(lapply(pc_levels, function(pc_level) {
    truth <- integer(length(region_indices))
    p_is <- p_co <- p_simes <- p_fisher <- numeric(length(region_indices))
    for (j in seq_along(region_indices)) {
      indices <- region_indices[[j]]
      u <- max(1L, ceiling(length(indices) * pc_level))
      truth[[j]] <- as.integer(sum(theta[indices]) >= u)
      p_is[[j]] <- min(1, 1 / pc_evalue(
        is_scores$R, is_scores$R_til, indices, u
      ))
      p_co[[j]] <- min(1, 1 / pc_evalue(
        co_scores$R, co_scores$R_til, indices, u
      ))
      p_simes[[j]] <- min(1, Simes_PC_test_cpp(p_voxel[indices], u))
      p_fisher[[j]] <- min(1, Fisher_PC_test_cpp(p_voxel[indices], u))
    }
    p_by_method <- list(
      `SEFT-Ising` = p_is,
      `SEFT-CO` = p_co,
      `BH+Simes` = p_simes,
      `BH+Fisher` = p_fisher
    )
    rbindlist(lapply(names(p_by_method), function(method) {
      decision <- BH(p_by_method[[method]], opts$alpha)
      data.table(
        method = method,
        pc_level = pc_level,
        discoveries = sum(decision),
        false_discoveries = sum(decision == 1L & truth == 0L),
        true_sets = sum(truth),
        fdr = cal_FDP(truth, decision),
        fnr = cal_MSR(truth, decision)
      )
    }))
  }))
}

run_task <- function(task_id) {
  requested_task_id <- task_id
  path <- result_path(task_id)
  if (result_valid(path, task_id)) {
    writeLines("COMPLETED (existing)", status_path(task_id))
    return(list(task_id = task_id, status = "existing"))
  }
  task <- task_grid[task_grid$task_id == requested_task_id]
  if (nrow(task) != 1L) stop("Cannot resolve task ", task_id)
  connection <- file(log_path(task_id), open = "wt")
  sink(connection, type = "output")
  sink(connection, type = "message")
  on.exit({
    try(sink(type = "message"), silent = TRUE)
    try(sink(type = "output"), silent = TRUE)
    try(close(connection), silent = TRUE)
  }, add = TRUE)
  started <- Sys.time()
  writeLines(
    sprintf("RUNNING pid=%d started=%s", Sys.getpid(), format(started, tz = "UTC")),
    status_path(task_id)
  )
  cat(sprintf(
    "task=%d h0=%.1f mu=%.1f repetition=%d data_seed=%d mirror_seed=%d\n",
    task$task_id, task$h0, task$mu, task$repetition,
    task$data_seed, task$mirror_seed
  ))

  simulated <- simulate_hmrf_data(
    L = opts$L,
    beta = opts$beta,
    h = task$h0,
    burnin_theta = opts$data_burnin,
    sweeps_theta = opts$data_sweeps,
    f1_weights = 1,
    f1_means = task$mu,
    f1_vars = 1,
    seed = task$data_seed
  )
  x <- simulated$x3d
  theta <- simulated$theta
  set.seed(task$mirror_seed)
  x_til <- array(rnorm(opts$L^3), dim = dim(x))
  baseline <- array(ifelse(abs(x) >= abs(x_til), x, x_til), dim = dim(x))

  fit <- hmrf_gem_fit(
    nb = nb, x3d = baseline, L = 1L,
    iter_max = opts$iter_max,
    sweep_b = opts$sweep_b,
    sweep_r = opts$sweep_r,
    a = 1, b = 2, alpha = 1e-3,
    stpmax = 1, max_backtrack = 10L, tol = 1e-4,
    beta_init = 0.5, h_init = -2,
    p_init = 1, mu_init = 1, sig2_init = 1,
    init_prob_theta = 0.8,
    seed = task$repetition,
    verbose = FALSE,
    f0_absmax2 = TRUE,
    f1_absmax01 = TRUE
  )
  gamma_fit <- hmrf_estimate_gamma_base(
    nb = nb, baseline3d = baseline,
    beta = fit$beta, h = fit$h,
    p = fit$p, mu = fit$mu, sig2 = fit$sig2,
    burnin = opts$burnin_lis, sweeps = opts$sweep_lis, thin = 1L,
    init_prob_theta = 0.5,
    seed = 10000L + task$repetition,
    n_chains = opts$n_chains,
    random_scan = TRUE, init_mode = 3L,
    clamp_eps = eps,
    f0_absmax2 = TRUE, f1_absmax01 = TRUE
  )
  is_R <- hmrf_plis_reweight_absmax(
    nb, x, baseline, gamma_fit$gamma_full,
    fit$p, fit$mu, fit$sig2, eps
  )
  is_R_til <- hmrf_plis_reweight_absmax(
    nb, x_til, baseline, gamma_fit$gamma_full,
    fit$p, fit$mu, fit$sig2, eps
  )

  set.seed(task$bandwidth_seed)
  bandwidth_h <- density(sample(c(as.vector(x_til), as.vector(x)), 1000L), bw = "nrd0")$bw
  co_fit <- cal_CLAW_scores_3d(
    x, x_til,
    2 * (1 - pnorm(abs(x))),
    2 * (1 - pnorm(abs(x_til))),
    h = bandwidth_h, bandwidth = 5,
    lambda = 0.5, neighbor_range = 10
  )
  co_R <- pmax(co_fit$R, eps)
  co_R_til <- pmax(co_fit$R_til, eps)
  is_R <- pmax(is_R, eps)
  is_R_til <- pmax(is_R_til, eps)
  all_scores <- c(is_R, is_R_til, co_R, co_R_til)
  scores_ok <- all(is.finite(all_scores)) && all(all_scores > 0)
  if (!scores_ok) stop("Non-finite or non-positive conformity score detected.")

  metrics <- evaluate_methods(
    theta, x,
    list(R = is_R, R_til = is_R_til),
    list(R = co_R, R_til = co_R_til)
  )
  trace_rel <- vapply(fit$trace, function(entry) as.numeric(entry$rel), numeric(1))
  elapsed <- as.numeric(difftime(Sys.time(), started, units = "secs"))
  result <- list(
    task = as.list(task),
    metrics = metrics,
    diagnostics = list(
      scores_finite_nonzero = scores_ok,
      is_score_range = range(c(is_R, is_R_til)),
      co_score_range = range(c(co_R, co_R_til)),
      is_observed_mean = mean(is_R),
      is_mirror_mean = mean(is_R_til),
      co_observed_mean = mean(co_R),
      co_mirror_mean = mean(co_R_til),
      fitted_beta = fit$beta,
      fitted_h = fit$h,
      fitted_p = fit$p,
      fitted_mu = fit$mu,
      fitted_sig2 = fit$sig2,
      gem_iterations = length(trace_rel),
      gem_last_rel = if (length(trace_rel)) tail(trace_rel, 1L) else NA_real_,
      gamma_chain_means = as.numeric(gamma_fit$chain_means),
      bandwidth_h = bandwidth_h,
      theta_count = sum(theta),
      x_sum = sum(x),
      mirror_sum = sum(x_til),
      x_digest = digest(x, algo = "xxhash64"),
      theta_digest = digest(theta, algo = "xxhash64"),
      mirror_digest = digest(x_til, algo = "xxhash64"),
      elapsed_seconds = elapsed
    )
  )
  temporary <- paste0(path, ".tmp.", Sys.getpid())
  saveRDS(result, temporary, compress = FALSE)
  if (!file.rename(temporary, path)) stop("Could not atomically install task result.")
  if (!result_valid(path, task_id)) stop("Saved task result failed validation.")
  writeLines(
    sprintf("COMPLETED elapsed_seconds=%.3f", elapsed),
    status_path(task_id)
  )
  cat(sprintf("completed elapsed_seconds=%.3f\n", elapsed))
  list(task_id = task_id, status = "completed")
}

run_safe <- function(task_id) {
  tryCatch(
    run_task(task_id),
    error = function(error) {
      writeLines(
        paste("FAILED", conditionMessage(error)), status_path(task_id)
      )
      list(task_id = task_id, status = "failed", error = conditionMessage(error))
    }
  )
}

workers_used <- min(opts$workers, nrow(selected_grid))
message("Starting ", nrow(selected_grid), " tasks with ", workers_used, " workers.")
run_results <- parallel::mclapply(
  selected_grid$task_id,
  run_safe,
  mc.cores = workers_used,
  mc.preschedule = FALSE,
  mc.set.seed = FALSE
)
failed <- vapply(run_results, function(item) identical(item$status, "failed"), logical(1))
if (any(failed)) {
  failed_ids <- vapply(run_results[failed], `[[`, integer(1), "task_id")
  writeLines(as.character(failed_ids), file.path(output_dir, "failed_tasks.txt"))
  writeLines("FAILED", file.path(output_dir, "RUN_STATE"))
  stop(length(failed_ids), " selected tasks failed.")
}

selected_results <- lapply(selected_grid$task_id, function(id) readRDS(result_path(id)))
metrics <- rbindlist(lapply(selected_results, function(result) {
  table <- copy(as.data.table(result$metrics))
  table[, `:=`(
    task_id = result$task$task_id,
    h0 = result$task$h0,
    mu = result$task$mu,
    repetition = result$task$repetition
  )]
  setcolorder(table, c("task_id", "h0", "mu", "repetition", setdiff(
    names(table), c("task_id", "h0", "mu", "repetition")
  )))
  table
}))
fwrite(metrics, file.path(output_dir, "per_replication_metrics.csv"))
summary_table <- metrics[, .(
  mean_fdr = mean(fdr),
  se_fdr = sd(fdr) / sqrt(.N),
  mean_fnr = mean(fnr),
  se_fnr = sd(fnr) / sqrt(.N),
  mean_discoveries = mean(discoveries)
), by = .(h0, mu, method, pc_level)]
fwrite(summary_table, file.path(output_dir, "summary.csv"))

diagnostics <- rbindlist(lapply(selected_results, function(result) {
  data.table(
    task_id = result$task$task_id,
    h0 = result$task$h0,
    mu = result$task$mu,
    repetition = result$task$repetition,
    scores_finite_nonzero = result$diagnostics$scores_finite_nonzero,
    fitted_beta = result$diagnostics$fitted_beta,
    fitted_h = result$diagnostics$fitted_h,
    fitted_mu = paste(result$diagnostics$fitted_mu, collapse = ";"),
    fitted_sig2 = paste(result$diagnostics$fitted_sig2, collapse = ";"),
    gem_iterations = result$diagnostics$gem_iterations,
    gem_last_rel = result$diagnostics$gem_last_rel,
    gamma_chain_range = diff(range(result$diagnostics$gamma_chain_means)),
    elapsed_seconds = result$diagnostics$elapsed_seconds
  )
}))
fwrite(diagnostics, file.path(output_dir, "score_and_fit_diagnostics.csv"))

is_full_run <- nrow(selected_grid) == nrow(task_grid) &&
  setequal(selected_grid$task_id, task_grid$task_id)
if (is_full_run) {
  if (nrow(metrics) != 1800L * 4L * length(pc_levels)) {
    stop("Unexpected number of full-run metric rows.")
  }
  cell_counts <- unique(metrics[, .(h0, mu, repetition)])[, .N, by = .(h0, mu)]
  if (nrow(cell_counts) != 18L || any(cell_counts$N != 100L)) {
    stop("Each Ising parameter cell must contain exactly 100 repetitions.")
  }
  for (h0_value in h_values) for (mu_value in mu_values) {
    cell <- metrics[h0 == h0_value & mu == mu_value]
    msr_is <- fsr_is <- msr_co <- fsr_co <- array(
      NA_real_, dim = c(opts$B, 5L, length(pc_levels))
    )
    for (repetition_value in seq_len(opts$B)) {
      one <- cell[repetition == repetition_value]
      for (pc_index in seq_along(pc_levels)) {
        level <- pc_levels[[pc_index]]
        get_value <- function(method_name, column_name) one[
          one$method == method_name & abs(one$pc_level - level) < 1e-12,
          get(column_name)
        ]
        msr_is[repetition_value, 3L, pc_index] <- get_value("SEFT-Ising", "fnr")
        fsr_is[repetition_value, 3L, pc_index] <- get_value("SEFT-Ising", "fdr")
        msr_co[repetition_value, 3L, pc_index] <- get_value("SEFT-CO", "fnr")
        fsr_co[repetition_value, 3L, pc_index] <- get_value("SEFT-CO", "fdr")
        for (method_index in 4:5) {
          method_name <- if (method_index == 4L) "BH+Simes" else "BH+Fisher"
          msr_is[repetition_value, method_index, pc_index] <- get_value(method_name, "fnr")
          fsr_is[repetition_value, method_index, pc_index] <- get_value(method_name, "fdr")
          msr_co[repetition_value, method_index, pc_index] <- get_value(method_name, "fnr")
          fsr_co[repetition_value, method_index, pc_index] <- get_value(method_name, "fdr")
        }
      }
    }
    file_name <- sprintf(
      "MSR_FDP_L%d_beta%.2f_h%.2f_mu%.2f.rds",
      opts$L, opts$beta, h0_value, mu_value
    )
    saveRDS(list(MSR = msr_is, FSR = fsr_is), file.path(legacy_is_dir, file_name))
    saveRDS(list(MSR = msr_co, FSR = fsr_co), file.path(legacy_co_dir, file_name))
  }
  writeLines("COMPLETE", file.path(output_dir, "RUN_STATE"))
  writeLines(format(Sys.time(), tz = "UTC"), file.path(output_dir, "RUN_COMPLETE"))
} else {
  writeLines("COMPLETE_SELECTED_TASKS", file.path(output_dir, "RUN_STATE"))
}
