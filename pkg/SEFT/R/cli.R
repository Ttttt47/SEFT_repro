.seft_cli_usage <- function() {
    paste(
        "Usage: seft (--zmap FILE | --tstat FILE) --atlas FILE --out-dir DIR [options]",
        "",
        "Core options:",
        "  --atlas-labels FILE       Region labels",
        "  --mask FILE               Analysis mask",
        "  --working-model MODELS    Comma-separated model names [co]",
        "  --simes                   Include BHe+Simes",
        "  --prefix NAME             Output prefix [seft]",
        "  --seed INTEGER            Random seed [1]",
        "",
        "Statistic options:",
        "  --df NUMBER               Residual df for --tstat",
        "  --design FILE             FSL VEST or plain design matrix",
        "",
        "Inference options:",
        "  --alpha NUMBER            Multiple-testing level [0.1]",
        "  --pc-levels LIST          Comma-separated levels [0.1,0.2,0.3]",
        "  --denoise                 Apply wavelet denoising [off]",
        "  --bandwidth NUMBER        CO spatial bandwidth [5]",
        "  --neighbor-range INTEGER  CO neighbourhood radius [10]",
        "  --lambda NUMBER           CO tuning parameter [0.5]",
        "  --score-clip NUMBER       CO score clip [0.99]",
        "  --keep-scores             Retain voxelwise scores",
        "  --quiet                   Suppress progress messages",
        "  --help                    Show this message",
        sep = "\n"
    )
}

.parse_cli <- function(args) {
    flags <- c("simes", "plot", "denoise", "keep-scores", "quiet", "help")
    output <- list()
    i <- 1L
    while (i <= length(args)) {
        key <- args[[i]]
        if (!startsWith(key, "--")) stop("Unexpected positional argument: ", key, call. = FALSE)
        name <- substring(key, 3L)
        if (name %in% flags) {
            output[[name]] <- TRUE
            i <- i + 1L
        } else {
            if (i == length(args)) stop("Missing value for ", key, call. = FALSE)
            output[[name]] <- args[[i + 1L]]
            i <- i + 2L
        }
    }
    output
}

.required_cli <- function(options, name) {
    value <- options[[name]]
    if (is.null(value) || !nzchar(value)) stop("Missing required argument --", name, call. = FALSE)
    value
}

.seft_cli <- function(args = commandArgs(trailingOnly = TRUE)) {
    options <- .parse_cli(args)
    if (isTRUE(options$help)) {
        cat(.seft_cli_usage(), "\n")
        return(invisible(0L))
    }
    zmap <- options$zmap
    tstat <- options$tstat
    if (is.null(zmap) == is.null(tstat)) stop("Supply exactly one of --zmap and --tstat.", call. = FALSE)
    result <- seft(
        zmap = zmap, tstat = tstat,
        atlas = .required_cli(options, "atlas"),
        atlas_labels = options[["atlas-labels"]], mask = options$mask,
        df = if (is.null(options$df)) NULL else as.numeric(options$df),
        design = options$design,
        working_model = options[["working-model"]] %||% options[["working-models"]] %||% "co",
        alpha = as.numeric(options$alpha %||% 0.1),
        pc_levels = as.numeric(strsplit(options[["pc-levels"]] %||% "0.1,0.2,0.3", ",", fixed = TRUE)[[1L]]),
        denoise = if (isTRUE(options$denoise)) "wavelet" else "none",
        bandwidth = as.numeric(options$bandwidth %||% 5),
        neighbor_range = as.integer(options[["neighbor-range"]] %||% 10L),
        lambda = as.numeric(options$lambda %||% 0.5),
        score_clip = as.numeric(options[["score-clip"]] %||% options[["score-clip-c"]] %||% 0.99),
        simes = isTRUE(options$simes), seed = as.integer(options$seed %||% 1L),
        out_dir = .required_cli(options, "out-dir"), prefix = options$prefix %||% "seft",
        keep_scores = isTRUE(options[["keep-scores"]]), verbose = !isTRUE(options$quiet)
    )
    print(result)
    invisible(0L)
}
