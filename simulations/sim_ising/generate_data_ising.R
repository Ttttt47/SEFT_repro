args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 5) {
    stop("Usage: Rscript simulations/sim_ising/generate_data_ising.R <L> <h> <mu> <B> <beta> [out_dir]")
}

L <- as.integer(args[[1]])
h <- as.numeric(args[[2]])
mu <- as.numeric(args[[3]])
B <- as.integer(args[[4]])
beta <- as.numeric(args[[5]])
out_dir <- if (length(args) >= 6) args[[6]] else file.path("outputs", "intermediate", "simulation_data")

if (!file.exists(file.path("R", "core", "Ising_functions.R"))) {
    stop("Run this script from the SEFT project root.")
}
source(file.path("R", "core", "Ising_functions.R"))

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

data_list <- vector("list", B)
for (b in seq_len(B)) {
    set.seed(b)
    sim_data <- simulate_hmrf_data(
        L = L,
        beta = beta,
        h = h,
        f1_weights = c(1),
        f1_means = c(mu),
        f1_vars = c(1),
        burnin_theta = 1000,
        sweeps_theta = 500,
        seed = b
    )
    data_list[[b]] <- list(sim_data)
    cat(sprintf("Done: L=%d beta=%.2f h=%.2f mu=%.2f b=%d\n", L, beta, h, mu, b))
}

fn <- file.path(out_dir, sprintf("3d_Ising_data_beta%.2f_h%.2f_mu%.2f.rds", beta, h, mu))
saveRDS(data_list, file = fn)
cat(sprintf("Saved: %s\n", fn))
