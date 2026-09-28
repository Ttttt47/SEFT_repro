library(glue)

if (!file.exists(file.path("simulations", "sim_3d_grf_iid", "run_single_case_3d.R"))) {
    stop("Run this script from the SEFT project root.")
}

mu_s <- seq(from = 2, to = 5, by = 0.5)
num_points_s <- c(10, 20, 30)
L <- 64
shape <- "sphere"
radius <- 12
B <- 100
cal_scores <- TRUE

scenario_cfg <- list(
    list(prefix = "3d_iid", denoise_level = 0),
    list(prefix = "3d_grf_denoised", denoise_level = 1),
    list(prefix = "3d_grf_nodenoise", denoise_level = 0)
)

rscript_bin <- file.path(R.home("bin"), "Rscript")
single_case_script <- file.path("simulations", "sim_3d_grf_iid", "run_single_case_3d.R")

for (cfg in scenario_cfg) {
    for (mu in mu_s) {
        for (num_points in num_points_s) {
            cat(glue("Running 3D case: scenario={cfg$prefix}, mu={mu}, num_points={num_points}\n"))
            status <- system2(
                rscript_bin,
                args = c(
                    single_case_script,
                    mu,
                    num_points,
                    L,
                    shape,
                    radius,
                    B,
                    cal_scores,
                    cfg$denoise_level,
                    cfg$prefix
                )
            )
            if (status != 0) {
                stop(glue("Failed 3D case: scenario={cfg$prefix}, mu={mu}, num_points={num_points}"))
            }
        }
    }
}
