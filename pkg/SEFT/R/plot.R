#' Plot regional SEFT results
#'
#' Plot the highest rejected partial-conjunction level for each atlas region.
#' Only methods present in the result are shown.
#'
#' @param x A `seft_result` object.
#' @param file Optional PDF filename.
#' @param tfce_coverage Optional data frame containing `region_id` and
#'   `tfce_coverage` columns.
#' @param main Optional plot title.
#' @param ... Unused.
#'
#' @return `x`, invisibly.
#' @export
plot.seft_result <- function(x, file = NULL, tfce_coverage = NULL, main = NULL, ...) {
    if (!inherits(x, "seft_result")) stop("x must be a seft_result.", call. = FALSE)
    region_frame <- unique(x$regions[c("region_id", "region_label")])
    region_frame <- region_frame[order(region_frame$region_id), , drop = FALSE]
    methods <- unique(x$regions$method)
    method_order <- c("SEFT-CO", "SEFT-Ising", "SEFT-FDR smoothing", "SEFT-DeepFDR", "SEFT-fcHMRF", "BHe")
    methods <- c(intersect(method_order, methods), setdiff(methods, method_order))
    add_tfce <- !is.null(tfce_coverage)
    if (add_tfce) {
        if (!is.data.frame(tfce_coverage) || !all(c("region_id", "tfce_coverage") %in% names(tfce_coverage))) {
            stop("tfce_coverage must contain region_id and tfce_coverage columns.", call. = FALSE)
        }
        if (any(!is.finite(tfce_coverage$tfce_coverage)) || any(tfce_coverage$tfce_coverage < 0 | tfce_coverage$tfce_coverage > 1)) {
            stop("TFCE coverage values must be finite and in [0, 1].", call. = FALSE)
        }
    }
    pc_levels <- sort(unique(x$regions$pc_level))
    fixed_colours <- c(
        `0.01` = "#FDC527", `0.05` = "#FA8E47", `0.1` = "#ED5A5F",
        `0.2` = "#C93F73", `0.3` = "#9C2E7F", `0.4` = "#701F81", `0.5` = "#451071"
    )
    level_colours <- unname(fixed_colours[format(pc_levels, trim = TRUE, scientific = FALSE)])
    missing <- is.na(level_colours)
    if (any(missing)) level_colours[missing] <- grDevices::colorRampPalette(c("#FDC527", "#451071"))(length(pc_levels))[missing]
    names(level_colours) <- format(pc_levels, trim = TRUE, scientific = FALSE)

    n_regions <- nrow(region_frame)
    n_rows <- length(methods) + as.integer(add_tfce)
    if (!is.null(file)) {
        dir.create(dirname(file), recursive = TRUE, showWarnings = FALSE)
        grDevices::pdf(file, width = max(11, n_regions * 0.115), height = max(4.8, 2.9 + n_rows * 0.52), onefile = TRUE)
        on.exit(grDevices::dev.off(), add = TRUE)
    }
    old_par <- graphics::par(no.readonly = TRUE)
    on.exit(graphics::par(old_par), add = TRUE)
    graphics::par(mar = c(8.5, 8.3, 5.2, 1.5), xpd = NA)
    graphics::plot.new()
    graphics::plot.window(xlim = c(0.5, n_regions + 0.5), ylim = c(0.5, n_rows + 0.5))
    row_labels <- c(methods, if (add_tfce) "TFCE coverage")
    row_positions <- rev(seq_len(n_rows))
    for (method_index in seq_along(methods)) {
        selected <- x$regions[x$regions$method == methods[[method_index]] & x$regions$significant == 1L, , drop = FALSE]
        highest <- if (nrow(selected)) tapply(selected$pc_level, selected$region_id, max) else numeric()
        for (region_index in seq_len(n_regions)) {
            level <- unname(highest[as.character(region_frame$region_id[[region_index]])])
            colour <- if (!length(level) || is.na(level)) "#F1F1F1" else level_colours[[format(level, trim = TRUE, scientific = FALSE)]]
            graphics::rect(region_index - 0.5, row_positions[[method_index]] - 0.42, region_index + 0.5, row_positions[[method_index]] + 0.42, col = colour, border = "white", lwd = 0.25)
        }
    }
    if (add_tfce) {
        coverage_lookup <- setNames(tfce_coverage$tfce_coverage, tfce_coverage$region_id)
        y <- row_positions[[n_rows]]
        for (region_index in seq_len(n_regions)) {
            coverage <- coverage_lookup[[as.character(region_frame$region_id[[region_index]])]]
            if (is.null(coverage) || is.na(coverage)) coverage <- 0
            graphics::rect(region_index - 0.5, y - 0.42, region_index + 0.5, y + 0.42, col = grDevices::gray(1 - 0.85 * coverage), border = "white", lwd = 0.25)
        }
    }
    graphics::axis(2, at = row_positions, labels = row_labels, las = 1, tick = FALSE, cex.axis = 0.9)
    graphics::axis(1, at = seq_len(n_regions), labels = region_frame$region_label, las = 2, tick = FALSE, cex.axis = if (n_regions > 100) 0.43 else 0.65)
    groups <- .aal3_groups(region_frame$region_id)
    if (!all(is.na(groups))) {
        runs <- rle(groups)
        ends <- cumsum(runs$lengths)
        starts <- c(1L, head(ends, -1L) + 1L)
        for (i in seq_along(starts)) {
            if (i > 1L) {
                graphics::segments(
                    x0 = starts[[i]] - 0.5, y0 = 0.5,
                    x1 = starts[[i]] - 0.5, y1 = n_rows + 0.5,
                    col = "#69717A", lwd = 1.5, xpd = FALSE
                )
            }
            graphics::text((starts[[i]] + ends[[i]]) / 2, n_rows + 0.72, labels = runs$values[[i]], cex = 0.65, font = 2)
        }
    }
    graphics::box(col = "#69717A")
    if (is.null(main)) main <- "Highest significant partial-conjunction level by region"
    graphics::title(main = main, line = 3.6, adj = 0, cex.main = 1.05)
    graphics::legend(
        x = n_regions + 0.5, y = n_rows + 1.32,
        xjust = 1, yjust = 0.5, horiz = TRUE, bty = "n", xpd = NA,
        legend = c("Not selected", paste0("PC=", format(pc_levels, trim = TRUE))),
        fill = c("#F1F1F1", unname(level_colours)), border = NA,
        cex = 0.72, x.intersp = 0.55
    )
    invisible(x)
}

.aal3_groups <- function(ids) {
    divisions <- list(
        c(1, 16, "Frontal-motor"), c(17, 32, "Medial/OFC"),
        c(33, 46, "Insula-limbic"), c(47, 60, "Occipital"),
        c(61, 74, "Parietal-somatic"), c(75, 82, "BG/thalamus"),
        c(83, 94, "Temporal"), c(95, 120, "Cerebellum"),
        c(121, 150, "Thalamic nuclei"), c(151, 158, "ACC/NAcc"),
        c(159, 170, "Midbrain")
    )
    result <- rep(NA_character_, length(ids))
    for (division in divisions) {
        inside <- ids >= as.integer(division[[1L]]) & ids <= as.integer(division[[2L]])
        result[inside] <- division[[3L]]
    }
    result
}
