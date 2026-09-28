# PLIS-type DeepFDR and fully connected HMRF score adapters.
#
# Source R/working_model_scores.R first. The expensive model fit runs in an
# isolated Python/PyTorch environment and returns pairwise-exchangeable local
# posterior-null scores with the same R/R_til interface as SEFT-CO.

if (!exists(".wm_validate_pair", mode = "function")) {
    stop("Source R/working_model_scores.R before ml_working_model_scores.R.")
}

.wm_resolve_ml_script <- function() {
    configured <- Sys.getenv("SEFT_ML_WORKING_MODEL_SCRIPT", unset = "")
    candidates <- unique(c(
        configured,
        file.path(
            getwd(), "python", "working_models",
            "ml_working_model.py"
        ),
        file.path(getwd(), "python", "ml_working_model.py"),
        file.path(getwd(), "..", "python", "ml_working_model.py")
    ))
    candidates <- candidates[nzchar(candidates) & file.exists(candidates)]
    if (!length(candidates)) {
        stop(
            "Cannot locate python/ml_working_model.py. Set ",
            "SEFT_ML_WORKING_MODEL_SCRIPT."
        )
    }
    normalizePath(candidates[[1]], mustWork = TRUE)
}

.wm_resolve_ml_python <- function() {
    configured <- Sys.getenv("SEFT_ML_PYTHON", unset = "")
    candidates <- unique(c(configured, Sys.which("python3")))
    candidates <- candidates[nzchar(candidates) & file.exists(candidates)]
    if (!length(candidates)) {
        stop(
            "Cannot locate the spatial-FDR Python environment. Set ",
            "SEFT_ML_PYTHON."
        )
    }
    normalizePath(candidates[[1]], mustWork = TRUE)
}

.wm_write_binary_volume <- function(x, path) {
    connection <- file(path, open = "wb")
    on.exit(close(connection), add = TRUE)
    writeBin(as.double(x), connection, size = 8L, endian = "little")
}

.wm_run_ml_working_model <- function(
    x,
    x_til,
    mask,
    appearance = NULL,
    method,
    seed,
    epochs,
    em_steps,
    learning_rate,
    channels,
    patch_size,
    patch_stride,
    device,
    vendor_root,
    threshold = 0.1,
    kde_mode = "fast",
    kde_grid_size = 16384L,
    kde_max_grid_size = 65536L,
    kde_tolerance = 1e-3,
    keep_work_dir = FALSE
) {
    mask <- .wm_validate_pair(x, x_til, mask)
    background <- make_plis_background(x, x_til, mask)
    background[!mask] <- 0
    x_for_model <- x
    x_til_for_model <- x_til
    x_for_model[!mask] <- 0
    x_til_for_model[!mask] <- 0

    work_dir <- tempfile(sprintf("seft_%s_", method))
    dir.create(work_dir, recursive = TRUE, showWarnings = FALSE)
    if (!keep_work_dir) {
        on.exit(unlink(work_dir, recursive = TRUE, force = TRUE), add = TRUE)
    }
    paths <- file.path(
        work_dir,
        c(
            "background.bin", "x.bin", "x_til.bin", "appearance.bin",
            "R.bin", "R_til.bin", "log_R.bin", "log_R_til.bin",
            "diagnostics.json", "python.log"
        )
    )
    names(paths) <- c(
        "background", "x", "x_til", "appearance", "R", "R_til",
        "log_R", "log_R_til", "diagnostics", "log"
    )
    .wm_write_binary_volume(background, paths[["background"]])
    .wm_write_binary_volume(x_for_model, paths[["x"]])
    .wm_write_binary_volume(x_til_for_model, paths[["x_til"]])
    if (!is.null(appearance)) {
        if (!identical(dim(appearance), dim(x)) ||
                any(!is.finite(appearance[mask]))) {
            stop("appearance must be a finite array with the same dimensions as x.")
        }
        appearance_for_model <- appearance
        appearance_for_model[!mask] <- 0
        .wm_write_binary_volume(
            appearance_for_model, paths[["appearance"]]
        )
    }

    arguments <- c(
        .wm_resolve_ml_script(),
        "--method", method,
        "--background", paths[["background"]],
        "--candidate-x", paths[["x"]],
        "--candidate-til", paths[["x_til"]],
        "--dims", paste(dim(x), collapse = ","),
        "--out-r", paths[["R"]],
        "--out-rtil", paths[["R_til"]],
        "--out-log-r", paths[["log_R"]],
        "--out-log-rtil", paths[["log_R_til"]],
        "--diagnostics", paths[["diagnostics"]],
        "--seed", as.character(as.integer(seed)),
        "--epochs", as.character(as.integer(epochs)),
        "--em-steps", as.character(as.integer(em_steps)),
        "--learning-rate", format(learning_rate, scientific = TRUE),
        "--channels", as.character(as.integer(channels)),
        "--patch-size", as.character(as.integer(patch_size)),
        "--patch-stride", as.character(as.integer(patch_stride)),
        "--device", device,
        "--vendor-root", vendor_root,
        "--threshold", format(threshold, scientific = FALSE),
        "--kde-mode", kde_mode,
        "--kde-grid-size", as.character(as.integer(kde_grid_size)),
        "--kde-max-grid-size", as.character(as.integer(kde_max_grid_size)),
        "--kde-tolerance", format(kde_tolerance, scientific = TRUE)
    )
    if (!is.null(appearance)) {
        arguments <- c(
            arguments, "--appearance", paths[["appearance"]]
        )
    }
    status <- system2(
        .wm_resolve_ml_python(),
        args = arguments,
        stdout = paths[["log"]],
        stderr = paths[["log"]]
    )
    if (!identical(status, 0L) ||
            !file.exists(paths[["R"]]) ||
            !file.exists(paths[["R_til"]])) {
        log_text <- if (file.exists(paths[["log"]])) {
            paste(tail(readLines(paths[["log"]], warn = FALSE), 40L),
                  collapse = "\n")
        } else {
            "(Python log was not created.)"
        }
        stop(
            sprintf("%s Python backend failed (status %s):\n%s",
                    method, status, log_text)
        )
    }
    read_score <- function(path) {
        connection <- file(path, open = "rb")
        on.exit(close(connection), add = TRUE)
        values <- readBin(
            connection, what = "double", n = length(x),
            size = 8L, endian = "little"
        )
        if (length(values) != length(x)) {
            stop("Python backend returned an incomplete score volume.")
        }
        array(values, dim = dim(x))
    }
    R <- read_score(paths[["R"]])
    R_til <- read_score(paths[["R_til"]])
    log_R <- if (file.exists(paths[["log_R"]])) {
        read_score(paths[["log_R"]])
    } else {
        log(R)
    }
    log_R_til <- if (file.exists(paths[["log_R_til"]])) {
        read_score(paths[["log_R_til"]])
    } else {
        log(R_til)
    }
    R[!mask] <- 1
    R_til[!mask] <- 1
    log_R[!mask] <- 0
    log_R_til[!mask] <- 0
    diagnostics <- jsonlite::fromJSON(paths[["diagnostics"]])
    diagnostics$python_log <- if (keep_work_dir) paths[["log"]] else NULL
    list(
        R = R,
        R_til = R_til,
        log_R = log_R,
        log_R_til = log_R_til,
        background = background,
        fit = diagnostics,
        model = method,
        work_dir = if (keep_work_dir) work_dir else NULL
    )
}

#' DeepFDR W-Net score in swap-invariant PLIS form.
#'
#' The official W-Net is trained on the shared background. Its unsupervised
#' segmentation output supplies a frozen spatial posterior map, which is then
#' reweighted by the real or mirror candidate's local likelihood ratio.
cal_deepfdr_scores_3d <- function(
    x,
    x_til,
    mask = NULL,
    seed = 0L,
    epochs = 17L,
    learning_rate = 1e-3,
    channels = 64L,
    patch_size = 30L,
    patch_stride = 24L,
    device = "auto",
    vendor_root = Sys.getenv("SEFT_ML_VENDOR_ROOT", unset = file.path(getwd(), ".vendor")),
    keep_work_dir = FALSE
) {
    if (any(as.integer(dim(x)) %% 4L != 0L) &&
            !identical(as.integer(dim(x)), c(30L, 30L, 30L))) {
        stop("DeepFDR paper-scale dimensions must be divisible by four.")
    }
    if (as.integer(seed) != 0L ||
            as.integer(epochs) != 17L ||
            !isTRUE(all.equal(learning_rate, 1e-3)) ||
            as.integer(channels) != 64L) {
        stop(
            "Official DeepFDR defaults are seed=0, epochs=17, ",
            "learning_rate=1e-3, channels=64."
        )
    }
    .wm_run_ml_working_model(
        x = x,
        x_til = x_til,
        mask = mask,
        appearance = NULL,
        method = "deepfdr",
        seed = seed,
        epochs = epochs,
        em_steps = 1L,
        learning_rate = learning_rate,
        channels = channels,
        patch_size = patch_size,
        patch_stride = patch_stride,
        device = device,
        vendor_root = vendor_root,
        threshold = 0.1,
        keep_work_dir = keep_work_dir
    )
}

#' Fully connected HMRF score in swap-invariant PLIS form.
#'
#' This uses the official CRF-RNN and permutohedral-lattice message passing.
#' Official mode also retains the repository's quadratic-memory Gaussian KDE
#' and therefore reproduces only the public 30^3 simulation runner. The
#' scalable real-data runner used in the paper is not fully released. The
#' separate beta/delta-mu appearance map is mandatory.
cal_fchmrf_scores_3d <- function(
    x,
    x_til,
    appearance,
    mask = NULL,
    seed = 1L,
    em_steps = 5L,
    learning_rate = 1e-4,
    threshold = 0.05,
    vendor_root = Sys.getenv("SEFT_ML_VENDOR_ROOT", unset = file.path(getwd(), ".vendor")),
    keep_work_dir = FALSE
) {
    if (!identical(as.integer(dim(x)), c(30L, 30L, 30L))) {
        stop(
            "The public fcHMRF simulation runner is hard-coded to 30^3, but ",
            "the paper applies the method to 439,758 voxels. Its scalable ",
            "real-data KDE/runner is not present in the public repository. ",
            "This repo-exact backend therefore accepts 30^3 only; this is not ",
            "a limitation of the fcHMRF method."
        )
    }
    if (missing(appearance)) {
        stop(
            "Official fcHMRF requires the delta_mu/beta appearance map ",
            "(`betapath` in the authors' CLI). A z-only simulation does not ",
            "contain this input, so x cannot be substituted silently."
        )
    }
    if (as.integer(em_steps) != 5L ||
            !isTRUE(all.equal(learning_rate, 1e-4)) ||
            !isTRUE(all.equal(threshold, 0.05))) {
        stop(
            "Official fcHMRF CLI defaults are em_steps=5, ",
            "learning_rate=1e-4, threshold=0.05."
        )
    }
    .wm_run_ml_working_model(
        x = x,
        x_til = x_til,
        mask = mask,
        appearance = appearance,
        method = "fchmrf",
        seed = seed,
        epochs = 1L,
        em_steps = em_steps,
        learning_rate = learning_rate,
        channels = 1L,
        patch_size = 30L,
        patch_stride = 24L,
        device = "cpu",
        vendor_root = vendor_root,
        threshold = threshold,
        keep_work_dir = keep_work_dir
    )
}

#' DeepFDR with a complete abs-max PLIS baseline adaptation.
cal_deepfdr_scores_3d_absmax <- function(
    x,
    x_til,
    mask = NULL,
    seed = 0L,
    epochs = 17L,
    learning_rate = 1e-3,
    channels = 64L,
    device = "auto",
    kde_mode = "fast",
    kde_grid_size = 16384L,
    kde_max_grid_size = 65536L,
    kde_tolerance = 1e-3,
    vendor_root = Sys.getenv("SEFT_ML_VENDOR_ROOT", unset = file.path(getwd(), ".vendor")),
    keep_work_dir = FALSE
) {
    if (any(as.integer(dim(x)) %% 4L != 0L) &&
            !identical(as.integer(dim(x)), c(30L, 30L, 30L))) {
        stop("DeepFDR paper-scale dimensions must be divisible by four.")
    }
    if (as.integer(seed) != 0L ||
            as.integer(epochs) != 17L ||
            !isTRUE(all.equal(learning_rate, 1e-3)) ||
            as.integer(channels) != 64L) {
        stop(
            "Official DeepFDR defaults are seed=0, epochs=17, ",
            "learning_rate=1e-3, channels=64."
        )
    }
    result <- .wm_run_ml_working_model(
        x = x,
        x_til = x_til,
        mask = mask,
        appearance = NULL,
        method = "deepfdr_absmax",
        seed = seed,
        epochs = epochs,
        em_steps = 1L,
        learning_rate = learning_rate,
        channels = channels,
        patch_size = 30L,
        patch_stride = 24L,
        device = device,
        vendor_root = vendor_root,
        threshold = 0.1,
        kde_mode = kde_mode,
        kde_grid_size = kde_grid_size,
        kde_max_grid_size = kde_max_grid_size,
        kde_tolerance = kde_tolerance,
        keep_work_dir = keep_work_dir
    )
    result$model <- "deepfdr_absmax_adaptation"
    result
}

#' fcHMRF with abs-max emissions and a scalable C++ KDE.
#'
#' By default the swap-invariant background is also the appearance/delta-mu
#' channel.  This is the explicitly labelled z-only simulation adaptation.
cal_fchmrf_scores_3d_absmax <- function(
    x,
    x_til,
    appearance = NULL,
    appearance_mode = "background",
    mask = NULL,
    seed = 1L,
    em_steps = 5L,
    learning_rate = 1e-4,
    threshold = 0.05,
    kde_mode = "fast",
    kde_grid_size = 16384L,
    kde_max_grid_size = 65536L,
    kde_tolerance = 1e-3,
    vendor_root = Sys.getenv("SEFT_ML_VENDOR_ROOT", unset = file.path(getwd(), ".vendor")),
    keep_work_dir = FALSE
) {
    appearance_mode <- match.arg(
        appearance_mode, c("background", "external")
    )
    mask_checked <- .wm_validate_pair(x, x_til, mask)
    if (appearance_mode == "background") {
        if (!is.null(appearance)) {
            stop(
                "appearance must be NULL when appearance_mode='background'."
            )
        }
        appearance <- make_plis_background(x, x_til, mask_checked)
    } else if (is.null(appearance)) {
        stop("appearance is required when appearance_mode='external'.")
    }
    if (as.integer(em_steps) != 5L ||
            !isTRUE(all.equal(learning_rate, 1e-4)) ||
            !isTRUE(all.equal(threshold, 0.05))) {
        stop(
            "Official fcHMRF CLI defaults are em_steps=5, ",
            "learning_rate=1e-4, threshold=0.05."
        )
    }
    result <- .wm_run_ml_working_model(
        x = x,
        x_til = x_til,
        mask = mask_checked,
        appearance = appearance,
        method = "fchmrf_absmax",
        seed = seed,
        epochs = 1L,
        em_steps = em_steps,
        learning_rate = learning_rate,
        channels = 1L,
        patch_size = 30L,
        patch_stride = 24L,
        device = "cpu",
        vendor_root = vendor_root,
        threshold = threshold,
        kde_mode = kde_mode,
        kde_grid_size = kde_grid_size,
        kde_max_grid_size = kde_max_grid_size,
        kde_tolerance = kde_tolerance,
        keep_work_dir = keep_work_dir
    )
    result$model <- "fchmrf_absmax_adaptation"
    result$fit$appearance_mode <- appearance_mode
    result
}

