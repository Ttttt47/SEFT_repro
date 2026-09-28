library(glue)

if (!file.exists(file.path("simulations", "sim_ising", "generate_data_ising.R"))) {
    stop("Run this script from the SEFT project root.")
}

L <- 64
beta <- 0.8
h_s <- seq(from = -2.5, to = -2.3, by = 0.1)
mu_s <- seq(from = 1.5, to = 4, by = 0.5)
B <- 100
cal_scores <- TRUE

prefix_claw <- "3d_Ising_beta0.8_CLAW"
prefix_plis <- "3d_Ising_beta0.8_Lmix1_f0_f1_maxabs_newplis"

rscript_bin <- file.path(R.home("bin"), "Rscript")
gen_script <- file.path("simulations", "sim_ising", "generate_data_ising.R")
claw_script <- file.path("simulations", "sim_ising", "run_single_case_ising_claw.R")
plis_script <- file.path("simulations", "sim_ising", "run_single_case_ising_plis.R")
sim_data_root <- file.path("outputs", "intermediate", "simulation_data")

for (h in h_s) {
    for (mu in mu_s) {
        cat(glue("Generating Ising data: h={h}, mu={mu}\n"))
        status <- system2(rscript_bin, args = c(gen_script, L, h, mu, B, beta, sim_data_root))
        if (status != 0) {
            stop(glue("Data generation failed: h={h}, mu={mu}"))
        }

        cat(glue("Running Ising CLAW: h={h}, mu={mu}\n"))
        status <- system2(
            rscript_bin,
            args = c(
                claw_script,
                "--beta", beta,
                "--h_true", h,
                "--mu", mu,
                "--L", L,
                "--B", B,
                "--cal_scores", cal_scores,
                "--prefix", prefix_claw
            )
        )
        if (status != 0) {
            stop(glue("CLAW case failed: h={h}, mu={mu}"))
        }

        cat(glue("Running Ising PLIS: h={h}, mu={mu}\n"))
        status <- system2(
            rscript_bin,
            args = c(
                plis_script,
                "--beta", beta,
                "--h_true", h,
                "--mu", mu,
                "--L", L,
                "--B", B,
                "--cal_scores", cal_scores,
                "--prefix", prefix_plis,
                "--iter_max", 1000,
                "--sweep_b", 20,
                "--sweep_r", 40,
                "--burnin_lis", 1000,
                "--sweep_lis", 2000,
                "--Lmix", 1,
                "--f0_absmax2", TRUE,
                "--f1_absmax01", TRUE
            )
        )
        if (status != 0) {
            stop(glue("PLIS case failed: h={h}, mu={mu}"))
        }
    }
}
