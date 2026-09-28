#' @export
print.seft_result <- function(x, ...) {
    cat("SEFT regional inference\n")
    cat(sprintf("  Regions: %d; voxels: %d\n", x$metadata$n_regions, x$metadata$n_mask_voxels))
    cat(sprintf("  Working models: %s\n", paste(x$metadata$working_models, collapse = ", ")))
    print(x$summary, row.names = FALSE)
    invisible(x)
}
