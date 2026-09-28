test_that("the command-line wrapper exposes help and runs a compact analysis", {
    expect_output(expect_invisible(SEFT:::.seft_cli("--help")), "Usage: seft")
    expect_error(SEFT:::.seft_cli("input.nii.gz"), "positional")
    expect_error(SEFT:::.seft_cli("--zmap"), "Missing value")
    expect_error(SEFT:::.seft_cli(c("--zmap", "z.nii.gz", "--tstat", "t.nii.gz")), "exactly one")
    expect_true(isTRUE(SEFT:::.parse_cli("--denoise")$denoise))

    fixture <- tiny_seft_fixture()
    input <- tempfile("seft-cli-input-")
    output <- tempfile("seft-cli-output-")
    dir.create(input)
    z_path <- file.path(input, "z.nii.gz")
    atlas_path <- file.path(input, "atlas.nii.gz")
    RNifti::writeNifti(fixture$z, z_path)
    RNifti::writeNifti(fixture$atlas, atlas_path)
    args <- c(
        "--zmap", z_path, "--atlas", atlas_path, "--out-dir", output,
        "--neighbor-range", "1", "--pc-levels", "0.1,0.2",
        "--working-model", "covariate-adaptive", "--simes", "--keep-scores",
        "--prefix", "cli", "--seed", "8", "--quiet"
    )
    expect_output(expect_invisible(SEFT:::.seft_cli(args)), "SEFT regional inference")
    expect_true(file.exists(file.path(output, "cli_result.rds")))
})

test_that("print reports the compact result summary", {
    fixture <- tiny_seft_fixture()
    result <- seft(zmap = fixture$z, atlas = fixture$atlas, neighbor_range = 1L, verbose = FALSE)
    expect_output(returned <- print(result), "Working models: co")
    expect_identical(returned, result)
})
