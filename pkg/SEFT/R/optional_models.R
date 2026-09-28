.optional_root <- function() {
    configured <- Sys.getenv("SEFT_OPTIONAL_ROOT", unset = "")
    if (nzchar(configured)) return(normalizePath(configured, winslash = "/", mustWork = FALSE))
    file.path(tools::R_user_dir("SEFT", which = "data"), "working-models")
}

.optional_setup_command <- function() {
    script <- system.file("optional", "setup_working_models.sh", package = "SEFT")
    paste("bash", shQuote(script))
}

.optional_backend <- function(script_name) {
    root <- .optional_root()
    python <- Sys.getenv("SEFT_ML_PYTHON", unset = file.path(root, "env", "bin", "python"))
    script <- file.path(root, "python", script_name)
    vendor <- Sys.getenv("SEFT_ML_VENDOR_ROOT", unset = file.path(root, "vendor"))
    if (!file.exists(python) || !file.exists(script) || !dir.exists(vendor)) {
        stop(
            "The requested optional working model is not installed. Run `",
            .optional_setup_command(), "` and retry.", call. = FALSE
        )
    }
    if (!requireNamespace("jsonlite", quietly = TRUE)) {
        stop("Optional working models require the R package 'jsonlite'.", call. = FALSE)
    }
    list(
        root = normalizePath(root, winslash = "/", mustWork = TRUE),
        python = normalizePath(python, winslash = "/", mustWork = TRUE),
        script = normalizePath(script, winslash = "/", mustWork = TRUE),
        vendor = normalizePath(vendor, winslash = "/", mustWork = TRUE)
    )
}

.write_binary_volume <- function(x, path) {
    connection <- file(path, open = "wb")
    on.exit(close(connection), add = TRUE)
    writeBin(as.double(x), connection, size = 8L, endian = "little")
}

.read_binary_volume <- function(path, dims) {
    connection <- file(path, open = "rb")
    on.exit(close(connection), add = TRUE)
    values <- readBin(connection, what = "double", n = prod(dims), size = 8L, endian = "little")
    if (length(values) != prod(dims)) stop("Optional backend returned an incomplete score volume.", call. = FALSE)
    array(values, dim = dims)
}

.run_optional_process <- function(backend, arguments, paths, model) {
    status <- system2(backend$python, args = c(backend$script, arguments), stdout = paths[["log"]], stderr = paths[["log"]])
    required <- paths[c("log_R", "log_R_til")]
    if (!identical(status, 0L) || any(!file.exists(required))) {
        log_text <- if (file.exists(paths[["log"]])) paste(tail(readLines(paths[["log"]], warn = FALSE), 60L), collapse = "\n") else "No Python log was created."
        stop(model, " backend failed (status ", status, "):\n", log_text, call. = FALSE)
    }
    invisible(TRUE)
}

.fdr_smoothing_scores <- function(x, x_til, mask, seed) {
    mask <- .validate_pair(x, x_til, mask)
    if (!all(mask)) stop("FDR smoothing requires a complete rectangular lattice.", call. = FALSE)
    backend <- .optional_backend("fdr_smoothing_absmax.py")
    background <- .absmax_background(x, x_til, mask)
    work_dir <- tempfile("seft_fdr_smoothing_")
    dir.create(work_dir, recursive = TRUE, showWarnings = FALSE)
    on.exit(unlink(work_dir, recursive = TRUE, force = TRUE), add = TRUE)
    paths <- setNames(file.path(work_dir, c("background.bin", "x.bin", "x_til.bin", "log_R.bin", "log_R_til.bin", "diagnostics.json", "python.log")), c("background", "x", "x_til", "log_R", "log_R_til", "diagnostics", "log"))
    .write_binary_volume(background, paths[["background"]])
    .write_binary_volume(x, paths[["x"]])
    .write_binary_volume(x_til, paths[["x_til"]])
    arguments <- c(
        "--background", paths[["background"]], "--candidate-x", paths[["x"]],
        "--candidate-til", paths[["x_til"]], "--dims", paste(dim(x), collapse = ","),
        "--out-log-r", paths[["log_R"]], "--out-log-rtil", paths[["log_R_til"]],
        "--diagnostics", paths[["diagnostics"]], "--seed", as.character(as.integer(seed)),
        "--num-sweeps", "10", "--fdr-level", "0.1", "--pr-table-size", "524288",
        "--trail-cache", file.path(backend$root, "cache", "smoothfdr_trails"),
        "--trail-mode", "greedy", "--solver-mode", "official", "--gfl-threads", "1",
        "--grid-maxsteps", "400", "--grid-converge", "1e-5",
        "--lambda-min", "0.2", "--lambda-max", "1.5", "--lambda-bins", "30", "--verbose", "0"
    )
    .run_optional_process(backend, arguments, paths, "FDR smoothing")
    log_R <- .read_binary_volume(paths[["log_R"]], dim(x))
    log_R_til <- .read_binary_volume(paths[["log_R_til"]], dim(x))
    list(
        R = exp(log_R), R_til = exp(log_R_til), log_R = log_R,
        log_R_til = log_R_til, background = background,
        fit = jsonlite::fromJSON(paths[["diagnostics"]])
    )
}

.ml_scores <- function(x, x_til, mask, model, seed) {
    mask <- .validate_pair(x, x_til, mask)
    if (model == "deepfdr" && any(dim(x) < 8L)) {
        stop("DeepFDR requires each working dimension to be at least eight voxels.", call. = FALSE)
    }
    backend <- .optional_backend("ml_working_model.py")
    background <- .absmax_background(x, x_til, mask)
    work_dir <- tempfile(paste0("seft_", model, "_"))
    dir.create(work_dir, recursive = TRUE, showWarnings = FALSE)
    on.exit(unlink(work_dir, recursive = TRUE, force = TRUE), add = TRUE)
    paths <- setNames(file.path(work_dir, c("background.bin", "x.bin", "x_til.bin", "appearance.bin", "R.bin", "R_til.bin", "log_R.bin", "log_R_til.bin", "diagnostics.json", "python.log")), c("background", "x", "x_til", "appearance", "R", "R_til", "log_R", "log_R_til", "diagnostics", "log"))
    .write_binary_volume(background, paths[["background"]])
    .write_binary_volume(x, paths[["x"]])
    .write_binary_volume(x_til, paths[["x_til"]])
    method <- paste0(model, "_absmax")
    arguments <- c(
        "--method", method, "--background", paths[["background"]],
        "--candidate-x", paths[["x"]], "--candidate-til", paths[["x_til"]],
        "--dims", paste(dim(x), collapse = ","), "--out-r", paths[["R"]],
        "--out-rtil", paths[["R_til"]], "--out-log-r", paths[["log_R"]],
        "--out-log-rtil", paths[["log_R_til"]], "--diagnostics", paths[["diagnostics"]],
        "--seed", as.character(as.integer(seed)), "--vendor-root", backend$vendor,
        "--kde-mode", "fast", "--kde-grid-size", "16384",
        "--kde-max-grid-size", "65536", "--kde-tolerance", "1e-3"
    )
    if (model == "deepfdr") {
        arguments <- c(arguments, "--epochs", "17", "--em-steps", "1", "--learning-rate", "1e-3", "--channels", "64", "--patch-size", "30", "--patch-stride", "24", "--device", Sys.getenv("SEFT_DEEPFDR_DEVICE", unset = "auto"), "--threshold", "0.1")
    } else {
        .write_binary_volume(background, paths[["appearance"]])
        arguments <- c(arguments, "--appearance", paths[["appearance"]], "--epochs", "1", "--em-steps", "5", "--learning-rate", "1e-4", "--channels", "1", "--patch-size", "30", "--patch-stride", "24", "--device", "cpu", "--threshold", "0.05")
    }
    .run_optional_process(backend, arguments, paths, if (model == "deepfdr") "DeepFDR" else "fcHMRF")
    R <- .read_binary_volume(paths[["R"]], dim(x))
    R_til <- .read_binary_volume(paths[["R_til"]], dim(x))
    log_R <- .read_binary_volume(paths[["log_R"]], dim(x))
    log_R_til <- .read_binary_volume(paths[["log_R_til"]], dim(x))
    R[!mask] <- 1
    R_til[!mask] <- 1
    log_R[!mask] <- 0
    log_R_til[!mask] <- 0
    list(
        R = R, R_til = R_til, log_R = log_R, log_R_til = log_R_til,
        background = background, fit = jsonlite::fromJSON(paths[["diagnostics"]])
    )
}
