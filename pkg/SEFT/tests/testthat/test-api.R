test_that("seft provides one consistent API for built-in models", {
    fixture <- tiny_seft_fixture()
    before <- if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) get(".Random.seed", envir = .GlobalEnv) else NULL
    result <- seft(
        zmap = fixture$z, atlas = fixture$atlas,
        working_model = c("co", "ising"), simes = TRUE,
        pc_levels = c(0.1, 0.2), neighbor_range = 1L,
        seed = 11L, verbose = FALSE
    )
    after <- if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) get(".Random.seed", envir = .GlobalEnv) else NULL
    expect_s3_class(result, "seft_result")
    expect_identical(names(result$models), c("co", "ising"))
    expect_setequal(unique(result$regions$method), c("SEFT-CO", "SEFT-Ising", "BHe"))
    expect_true(all(is.finite(result$regions$pc_p_value)))
    expect_true(all(result$regions$significant %in% 0:1))
    expect_identical(after, before)
})

test_that("fixed seeds reproduce regional output", {
    fixture <- tiny_seft_fixture()
    run <- function() seft(zmap = fixture$z, atlas = fixture$atlas, neighbor_range = 1L, seed = 7L, verbose = FALSE)
    expect_identical(run()$regions, run()$regions)
})

test_that("t maps, masks, labels, and retained scores use the public API", {
    fixture <- tiny_seft_fixture()
    mask <- fixture$mask
    mask[1, , ] <- FALSE
    labels <- c(`1` = "Left", `2` = "Right")
    result <- seft(
        tstat = fixture$z, df = 25, atlas = fixture$atlas, mask = mask,
        atlas_labels = labels, keep_scores = TRUE,
        neighbor_range = 1L, pc_levels = 0.2, verbose = FALSE
    )
    expect_identical(result$metadata$statistic, "signed Z converted from t")
    expect_equal(result$metadata$residual_df, 25)
    expect_setequal(unique(result$regions$region_label), unname(labels))
    expect_named(result$models$co$scores, c("R", "R_til", "log_R", "log_R_til", "background"))

    design_result <- seft(
        tstat = fixture$z, design = cbind(1, seq_len(12)), atlas = fixture$atlas,
        neighbor_range = 1L, pc_levels = 0.2, verbose = FALSE
    )
    expect_equal(design_result$metadata$residual_df, 10)
})

test_that("input validation rejects ambiguous and malformed inputs", {
    fixture <- tiny_seft_fixture()
    expect_error(seft(atlas = fixture$atlas), "exactly one")
    expect_error(seft(zmap = fixture$z, tstat = fixture$z, atlas = fixture$atlas), "exactly one")
    expect_error(seft(zmap = fixture$z, atlas = array(1, c(2, 2, 2))), "Geometry mismatch")
    bad_atlas <- fixture$atlas
    bad_atlas[[1L]] <- 1.5
    expect_error(seft(zmap = fixture$z, atlas = bad_atlas), "integer")
    expect_error(seft(zmap = fixture$z, atlas = fixture$atlas, working_model = "unknown"), "working_model")
    expect_error(seft(zmap = fixture$z, atlas = fixture$atlas, df = 3), "only used with tstat")
    expect_error(seft(tstat = fixture$z, atlas = fixture$atlas), "requires either df or design")
    expect_error(seft(tstat = fixture$z, atlas = fixture$atlas, df = 3, design = diag(4)), "not both")
    expect_error(seft(tstat = fixture$z, atlas = fixture$atlas, df = 0), "positive")
    expect_error(seft(zmap = fixture$z, atlas = fixture$atlas, alpha = 0), "allowed range")
    expect_error(seft(zmap = fixture$z, atlas = fixture$atlas, bandwidth = 0), "allowed range")
    expect_error(seft(zmap = fixture$z, atlas = fixture$atlas, neighbor_range = 0), "positive integer")
    expect_error(seft(zmap = fixture$z, atlas = fixture$atlas, pc_levels = 0), "pc_levels")
    expect_error(seft(zmap = fixture$z, atlas = fixture$atlas, seed = 1.2), "single integer")
    expect_error(seft(zmap = fixture$z, atlas = fixture$atlas, prefix = "bad/name"), "filename stem")
    expect_error(seft(zmap = fixture$z, atlas = fixture$atlas, mask = array(0, dim(fixture$z))), "No atlas voxels")
})

test_that("optional models fail with an actionable setup command", {
    fixture <- tiny_seft_fixture()
    old <- Sys.getenv("SEFT_OPTIONAL_ROOT", unset = NA_character_)
    on.exit(if (is.na(old)) Sys.unsetenv("SEFT_OPTIONAL_ROOT") else Sys.setenv(SEFT_OPTIONAL_ROOT = old), add = TRUE)
    Sys.setenv(SEFT_OPTIONAL_ROOT = tempfile("missing-seft-options-"))
    expect_error(
        seft(zmap = fixture$z, atlas = fixture$atlas, working_model = "fdr-smoothing", seed = 1L, verbose = FALSE),
        "setup_working_models"
    )
})

test_that("wavelet denoising remains an optional path", {
    skip_if_not_installed("waveslim")
    set.seed(4)
    dims <- c(8L, 8L, 8L)
    result <- seft(
        zmap = array(rnorm(prod(dims)), dims), atlas = array(1L, dims),
        denoise = "wavelet", neighbor_range = 1L,
        seed = 2L, verbose = FALSE
    )
    expect_s3_class(result, "seft_result")
    expect_identical(result$metadata$denoise, "wavelet")
})
