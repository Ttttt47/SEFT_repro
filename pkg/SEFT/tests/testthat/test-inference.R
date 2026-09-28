test_that("BH and e-BH agree under reciprocal transformation", {
    e <- c(100, 30, 3, 1)
    expect_identical(SEFT:::.e_bh_decision(e, 0.1), SEFT:::.bh_decision(pmin(1, 1 / e), 0.1))
})

test_that("compiled Simes partial-conjunction test matches its reference", {
    p <- c(0.01, 0.04, 0.2, 0.7, 0.9)
    for (u in seq_along(p)) {
        ordered <- sort(p)
        reference <- min((length(p) - u + 1L) * ordered[u:length(p)] / seq_len(length(p) - u + 1L))
        expect_equal(SEFT:::Simes_PC_test_cpp(p, u), reference, tolerance = 1e-14)
    }
    expect_error(SEFT:::Simes_PC_test_cpp(p, 0L), "u must be")
})

test_that("compiled CO and regional e-values validate their compact interfaces", {
    x <- array(seq(-1, 1, length.out = 8), c(2, 2, 2))
    p <- 2 * pnorm(-abs(x))
    scores <- SEFT:::cal_CLAW_scores_3d(x, -x, p, p, neighbor_range = 1L)
    expect_named(scores, c("pi", "log_f", "log_f_til", "R", "R_til"))
    expect_error(SEFT:::cal_CLAW_scores_3d(x, array(0, c(2, 2, 3)), p, p), "same dimensions")
    expect_error(SEFT:::cal_CLAW_scores_3d(x, -x, p, p, bandwidth = 0), "Invalid CO")

    log_scores <- rbind(c(-1, -2, -3), c(-3, -2, -1))
    value <- SEFT:::cal_ELIS_cpp(log_scores, 1L, til_mth = "neglog")
    expect_true(is.finite(value) && value >= 0)
    expect_error(SEFT:::cal_ELIS_cpp(log_scores, 0L, til_mth = "neglog"), "u must be")
    expect_error(SEFT:::cal_ELIS_cpp(log_scores, 1L), "uses Sym_miny")
})

test_that("t statistics convert to signed Z with the requested df", {
    t <- array(c(-3, -1, 0, 1, 3, 0, 0, 0), c(2, 2, 2))
    observed <- SEFT:::.t_to_z(t, 20)
    reference <- sign(t) * qnorm(pt(-abs(t), 20), lower.tail = FALSE)
    reference[t == 0] <- 0
    expect_equal(observed, reference, tolerance = 1e-12)
    expect_equal(SEFT:::.design_df(cbind(1, seq_len(8))), 6)
})

test_that("FSL VEST designs are parsed and checked", {
    path <- tempfile(fileext = ".mat")
    writeLines(c("/NumWaves 2", "/NumPoints 5", "/Matrix", "1 0", "1 1", "1 2", "1 3", "1 4"), path)
    expect_equal(SEFT:::.design_df(path), 3)
    expect_error(SEFT:::.design_df(cbind(1:4, 2 * (1:4))), "rank deficient")
})
