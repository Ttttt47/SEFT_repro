test_that("the 2 by 2 by 2 neighbour graph is exact", {
    mask <- array(TRUE, c(2, 2, 2))
    neighbours <- SEFT:::hmrf_build_neighbors_6n(mask)
    expect_length(neighbours$mask_idx, 8L)
    expect_true(all(rowSums(neighbours$neigh >= 0L) == 3L))
    expect_equal(sum(neighbours$neigh >= 0L) / 2, 12L)
})

test_that("2 by 2 by 2 Ising posterior agrees with exact enumeration", {
    mask <- array(TRUE, c(2, 2, 2))
    neighbours <- SEFT:::hmrf_build_neighbors_6n(mask)
    baseline <- array(c(-1.1, -0.5, 0.2, 0.8, -0.7, 0.4, 1.2, -0.2), dim = dim(mask))
    beta <- 0.18
    h <- -1.1
    p <- 1
    mu <- 1.2
    sig2 <- 1
    llr <- SEFT:::hmrf_llr_mixnorm(baseline, p, mu, sig2, TRUE, TRUE)
    states <- as.matrix(expand.grid(rep(list(0:1), 8L)))
    edges <- which(neighbours$neigh >= 0L, arr.ind = TRUE)
    edge_pairs <- unique(t(apply(edges, 1L, function(row) sort(c(row[[1L]], neighbours$neigh[row[[1L]], row[[2L]]] + 1L)))))
    edge_pairs <- edge_pairs[edge_pairs[, 1L] < edge_pairs[, 2L], , drop = FALSE]
    log_weight <- apply(states, 1L, function(theta) beta * sum(theta[edge_pairs[, 1L]] * theta[edge_pairs[, 2L]]) + sum((h + llr) * theta))
    weight <- exp(log_weight - max(log_weight))
    exact <- colSums(states * weight) / sum(weight)
    sampled <- SEFT:::hmrf_estimate_gamma_base(
        neighbours, baseline, beta, h, p, mu, sig2,
        burnin = 1000L, sweeps = 8000L, thin = 1L,
        init_prob_theta = 0.5, seed = 91L, n_chains = 4L,
        random_scan = TRUE, init_mode = 3L,
        f0_absmax2 = TRUE, f1_absmax01 = TRUE, clamp_eps = 1e-10
    )$gamma_mask
    expect_equal(sampled, unname(exact), tolerance = 0.035)
})

test_that("Ising baseline and candidate scores obey swaps", {
    set.seed(19)
    x <- array(rnorm(8), c(2, 2, 2))
    x_til <- array(rnorm(8), c(2, 2, 2))
    mask <- array(TRUE, dim(x))
    background <- SEFT:::.absmax_background(x, x_til, mask)
    expect_identical(background, SEFT:::.absmax_background(x_til, x, mask))
    neighbours <- SEFT:::hmrf_build_neighbors_6n(mask)
    gamma <- rep(0.4, 8L)
    left <- SEFT:::hmrf_plis_reweight_absmax(neighbours, x, background, gamma, 1, 1, 1, 1e-10)
    right <- SEFT:::hmrf_plis_reweight_absmax(neighbours, x_til, background, gamma, 1, 1, 1, 1e-10)
    swapped_background <- SEFT:::.absmax_background(x_til, x, mask)
    swapped_left <- SEFT:::hmrf_plis_reweight_absmax(neighbours, x_til, swapped_background, gamma, 1, 1, 1, 1e-10)
    swapped_right <- SEFT:::hmrf_plis_reweight_absmax(neighbours, x, swapped_background, gamma, 1, 1, 1, 1e-10)
    expect_equal(left, swapped_right, tolerance = 0)
    expect_equal(right, swapped_left, tolerance = 0)
})
