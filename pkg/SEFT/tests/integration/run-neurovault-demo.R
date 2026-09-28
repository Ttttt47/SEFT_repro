#!/usr/bin/env Rscript

suppressPackageStartupMessages(library(SEFT))
demo <- system.file("extdata", "neurovault_790848", package = "SEFT")
if (!nzchar(demo)) stop("The installed SEFT package does not contain the public demo.")
output <- Sys.getenv("SEFT_DEMO_OUTPUT", unset = file.path(tempdir(), "seft-neurovault-demo"))
result <- seft(
    zmap = file.path(demo, "speech_vs_reversed_zmap_aal3_2mm.nii.gz"),
    atlas = file.path(demo, "AAL3v1.nii.gz"),
    atlas_labels = file.path(demo, "AAL3v1.nii.txt"),
    working_model = c("co", "ising"), simes = TRUE,
    denoise = "wavelet", neighbor_range = 1L, seed = 1L,
    out_dir = output, prefix = "neurovault_790848", verbose = TRUE
)
stopifnot(
    result$metadata$n_regions == 166L,
    setequal(unique(result$regions$method), c("SEFT-CO", "SEFT-Ising", "BHe")),
    all(is.finite(result$regions$pc_p_value)),
    all(file.exists(unlist(result$output_files[c("regions", "summary", "rds", "signed_z", "figure")]))),
    file.info(result$output_files$figure)$size > 0
)
cat("NeuroVault integration test passed. Output:", output, "\n")
