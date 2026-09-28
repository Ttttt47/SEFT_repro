test_that("the bundled NeuroVault map matches the AAL3 grid", {
    demo <- system.file("extdata", "neurovault_790848", package = "SEFT")
    z <- RNifti::readNifti(file.path(demo, "speech_vs_reversed_zmap_aal3_2mm.nii.gz"))
    atlas <- RNifti::readNifti(file.path(demo, "AAL3v1.nii.gz"))
    expect_identical(dim(z), dim(atlas))
    expect_equal(unclass(RNifti::xform(z)), unclass(RNifti::xform(atlas)), tolerance = 1e-6, ignore_attr = TRUE)
    atlas_values <- as.array(atlas)
    z_values <- as.array(z)
    expect_equal(length(unique(atlas_values[atlas_values > 0])), 166L)
    expect_true(all(is.finite(z_values)))
    expect_true(all(z_values[atlas_values <= 0] == 0))
    expect_true(any(z_values[atlas_values > 0] > 0))
    expect_true(any(z_values[atlas_values > 0] < 0))
    figure <- file.path(demo, "seft_available_methods.pdf")
    expect_true(file.exists(figure))
    expect_gt(file.info(figure)$size, 0)
})

test_that("the public demo runs through compiled CO inference", {
    skip_on_cran()
    demo <- system.file("extdata", "neurovault_790848", package = "SEFT")
    z_path <- file.path(demo, "speech_vs_reversed_zmap_aal3_2mm.nii.gz")
    atlas_path <- file.path(demo, "AAL3v1.nii.gz")
    result <- seft(
        zmap = z_path, atlas = atlas_path,
        atlas_labels = file.path(demo, "AAL3v1.nii.txt"),
        working_model = "co", neighbor_range = 1L,
        pc_levels = c(0.1, 0.2), seed = 1L, verbose = FALSE
    )
    expect_equal(result$metadata$n_regions, 166L)
    expect_true(all(is.finite(result$regions$pc_p_value)))
    expect_true(all(result$regions$significant %in% 0:1))
})
