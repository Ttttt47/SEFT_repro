library(glue)

if (!file.exists(file.path("simulations", "sim_2d", "run_single_case_2d.R"))) {
    stop("Run this script from the SEFT project root.")
}

L <- 500
mu_s <- seq(1.5, 3.5, 0.2)
shape_s <- c("disk", "iid")
B <- 100
cal_scores <- TRUE
prefix <- "2d_main"

rscript_bin <- file.path(R.home("bin"), "Rscript")
single_case_script <- file.path("simulations", "sim_2d", "run_single_case_2d.R")

for (shape in shape_s) {
    for (mu in mu_s) {
        cat(glue("Running 2D case: mu={mu}, shape={shape}\n"))
        status <- system2(
            rscript_bin,
            args = c(single_case_script, mu, L, shape, B, cal_scores, prefix)
        )
        if (status != 0) {
            stop(glue("Failed on 2D case: mu={mu}, shape={shape}"))
        }
    }
}
