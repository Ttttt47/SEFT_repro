.read_volume <- function(x, name) {
    path <- NULL
    image <- NULL
    if (is.character(x) && length(x) == 1L) {
        if (!file.exists(x)) stop(name, " does not exist: ", x, call. = FALSE)
        path <- normalizePath(x, winslash = "/", mustWork = TRUE)
        image <- RNifti::readNifti(path)
        data <- as.array(image)
    } else if (inherits(x, "niftiImage")) {
        image <- x
        data <- as.array(x)
    } else if (is.array(x) && (is.numeric(x) || is.logical(x))) {
        data <- x
    } else {
        stop(name, " must be a NIfTI path, RNifti image, or numeric array.", call. = FALSE)
    }
    dims <- dim(data)
    if (length(dims) > 3L && all(dims[-seq_len(3L)] == 1L)) {
        data <- array(data, dim = dims[seq_len(3L)])
    }
    if (length(dim(data)) != 3L || any(dim(data) < 1L)) {
        stop(name, " must be three-dimensional.", call. = FALSE)
    }
    affine <- if (is.null(image)) NULL else tryCatch(unclass(RNifti::xform(image)), error = function(e) NULL)
    list(data = array(as.numeric(data), dim = dim(data)), image = image, affine = affine, path = path)
}

.assert_geometry <- function(reference, candidate, reference_name, candidate_name) {
    if (!identical(dim(reference$data), dim(candidate$data))) {
        stop("Geometry mismatch: ", candidate_name, " dimensions differ from ", reference_name, ".", call. = FALSE)
    }
    if (!is.null(reference$affine) && !is.null(candidate$affine) && !isTRUE(all.equal(reference$affine, candidate$affine, tolerance = 1e-5, check.attributes = FALSE))) {
        stop("Geometry mismatch: ", candidate_name, " affine differs from ", reference_name, ".", call. = FALSE)
    }
    invisible(TRUE)
}

.read_design <- function(design) {
    if (is.matrix(design) || is.data.frame(design)) return(as.matrix(design))
    if (!is.character(design) || length(design) != 1L || !file.exists(design)) {
        stop("design must be a numeric matrix or an existing matrix file.", call. = FALSE)
    }
    lines <- readLines(design, warn = FALSE)
    matrix_line <- which(trimws(lines) == "/Matrix")
    if (length(matrix_line)) {
        rows <- lines[seq.int(matrix_line[[1L]] + 1L, length(lines))]
        rows <- rows[nzchar(trimws(rows))]
        parsed <- lapply(rows, function(line) scan(text = line, quiet = TRUE))
        widths <- lengths(parsed)
        if (!length(parsed) || length(unique(widths)) != 1L) stop("The FSL design matrix is empty or ragged.", call. = FALSE)
        return(do.call(rbind, parsed))
    }
    as.matrix(utils::read.table(design, header = FALSE, check.names = FALSE))
}

.design_df <- function(design) {
    matrix <- .read_design(design)
    storage.mode(matrix) <- "double"
    if (!nrow(matrix) || !ncol(matrix) || any(!is.finite(matrix))) stop("design must be a finite, non-empty matrix.", call. = FALSE)
    rank <- qr(matrix)$rank
    if (rank < ncol(matrix)) stop("design is rank deficient.", call. = FALSE)
    residual <- nrow(matrix) - rank
    if (residual <= 0) stop("design has no residual degrees of freedom.", call. = FALSE)
    as.numeric(residual)
}

.t_to_z <- function(tstat, df) {
    z <- array(0, dim = dim(tstat))
    finite <- is.finite(tstat)
    log_tail <- stats::pt(-abs(tstat[finite]), df = df, log.p = TRUE)
    z_abs <- stats::qnorm(log_tail, lower.tail = FALSE, log.p = TRUE)
    z[finite] <- sign(tstat[finite]) * z_abs
    z[tstat == 0 & finite] <- 0
    z[!is.finite(z)] <- sign(tstat[!is.finite(z)]) * stats::qnorm(.Machine$double.xmin, lower.tail = FALSE)
    z[!is.finite(tstat)] <- 0
    z
}

.read_labels <- function(labels, region_ids) {
    if (is.null(labels)) {
        frame <- data.frame(region_id = region_ids, label = paste0("Region_", region_ids), stringsAsFactors = FALSE)
        return(frame)
    }
    if (is.data.frame(labels)) {
        if (!all(c("region_id", "label") %in% names(labels))) stop("atlas_labels data frame needs region_id and label columns.", call. = FALSE)
        frame <- labels[c("region_id", "label")]
    } else if (is.character(labels) && length(labels) > 1L && !is.null(names(labels))) {
        frame <- data.frame(region_id = as.integer(names(labels)), label = unname(labels), stringsAsFactors = FALSE)
    } else if (is.character(labels) && length(labels) == 1L && file.exists(labels)) {
        lines <- readLines(labels, warn = FALSE)
        lines <- trimws(lines)
        lines <- lines[nzchar(lines) & !startsWith(lines, "#")]
        if (!length(lines)) stop("atlas_labels is empty.", call. = FALSE)
        header <- strsplit(lines[[1L]], "[[:space:],]+")[[1L]]
        if (any(tolower(header) == "region_id")) {
            frame <- utils::read.table(labels, header = TRUE, sep = "", stringsAsFactors = FALSE, check.names = FALSE)
            label_name <- names(frame)[tolower(names(frame)) == "label"]
            id_name <- names(frame)[tolower(names(frame)) == "region_id"]
            if (!length(label_name) || !length(id_name)) stop("atlas_labels needs region_id and label columns.", call. = FALSE)
            frame <- data.frame(region_id = frame[[id_name[[1L]]]], label = frame[[label_name[[1L]]]], stringsAsFactors = FALSE)
        } else {
            pieces <- strsplit(lines, "[[:space:]]+")
            frame <- data.frame(
                region_id = suppressWarnings(as.integer(vapply(pieces, `[`, character(1), 1L))),
                label = vapply(pieces, function(x) if (length(x) >= 2L) x[[2L]] else "", character(1)),
                stringsAsFactors = FALSE
            )
        }
    } else stop("atlas_labels must be a label-file path, data frame, or named character vector.", call. = FALSE)
    frame$region_id <- as.integer(frame$region_id)
    frame$label <- as.character(frame$label)
    frame <- frame[is.finite(frame$region_id) & nzchar(frame$label), , drop = FALSE]
    if (anyDuplicated(frame$region_id)) stop("atlas_labels contains duplicate region identifiers.", call. = FALSE)
    lookup <- setNames(frame$label, frame$region_id)
    resolved <- unname(lookup[as.character(region_ids)])
    resolved[is.na(resolved) | !nzchar(resolved)] <- paste0("Region_", region_ids[is.na(resolved) | !nzchar(resolved)])
    data.frame(region_id = region_ids, label = resolved, stringsAsFactors = FALSE)
}

.write_result <- function(result, out_dir, prefix, signed_z, region_map, template = NULL) {
    out_dir <- normalizePath(out_dir, winslash = "/", mustWork = FALSE)
    tables <- file.path(out_dir, "tables")
    maps <- file.path(out_dir, "maps")
    figures <- file.path(out_dir, "figures")
    dir.create(tables, recursive = TRUE, showWarnings = FALSE)
    dir.create(maps, recursive = TRUE, showWarnings = FALSE)
    dir.create(figures, recursive = TRUE, showWarnings = FALSE)
    files <- list(
        regions = file.path(tables, paste0(prefix, "_region_results.tsv")),
        significant = file.path(tables, paste0(prefix, "_significant_regions.tsv")),
        summary = file.path(tables, paste0(prefix, "_summary.tsv")),
        rds = file.path(out_dir, paste0(prefix, "_result.rds")),
        signed_z = file.path(maps, paste0(prefix, "_signed_z.nii.gz")),
        figure = file.path(figures, paste0(prefix, "_available_methods.pdf")),
        discovery_maps = character()
    )
    utils::write.table(result$regions, files$regions, sep = "\t", row.names = FALSE, quote = FALSE, na = "NA")
    utils::write.table(result$regions[result$regions$significant == 1L, , drop = FALSE], files$significant, sep = "\t", row.names = FALSE, quote = FALSE, na = "NA")
    utils::write.table(result$summary, files$summary, sep = "\t", row.names = FALSE, quote = FALSE, na = "NA")
    RNifti::writeNifti(signed_z, files$signed_z, template = template, datatype = "float32")
    keys <- unique(result$regions[c("pc_level", "method")])
    for (i in seq_len(nrow(keys))) {
        rows <- result$regions$pc_level == keys$pc_level[[i]] & result$regions$method == keys$method[[i]] & result$regions$significant == 1L
        ids <- result$regions$region_id[rows]
        output <- array(as.integer(region_map), dim = dim(region_map))
        output[!output %in% ids] <- 0L
        level <- gsub("\\.", "p", format(keys$pc_level[[i]], trim = TRUE, scientific = FALSE))
        method <- gsub("[^a-z0-9]+", "_", tolower(keys$method[[i]]))
        path <- file.path(maps, paste0(prefix, "_pc", level, "_", method, "_regions.nii.gz"))
        RNifti::writeNifti(output, path, template = template, datatype = "int16")
        files$discovery_maps <- c(files$discovery_maps, path)
    }
    plot(result, file = files$figure)
    files
}
