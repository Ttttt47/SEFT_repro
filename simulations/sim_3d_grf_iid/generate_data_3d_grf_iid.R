library(glue)
library(neuRosim)

if (!file.exists(file.path("R", "core", "CLAW_functions.R"))) {
    stop("Run this script from the SEFT project root.")
}
source(file.path("R", "core", "CLAW_functions.R"))

L <- 64
mu_s <- seq(from = 2, to = 5, by = 0.5)
nums_points <- c(10, 20, 30)
B <- as.integer(Sys.getenv("SEFT_SIM_B", "100"))
if (nzchar(Sys.getenv("SEFT_SIM_MUS"))) {
    mu_s <- as.numeric(strsplit(Sys.getenv("SEFT_SIM_MUS"), ",", fixed = TRUE)[[1]])
}
if (nzchar(Sys.getenv("SEFT_SIM_NUM_POINTS"))) {
    nums_points <- as.integer(strsplit(Sys.getenv("SEFT_SIM_NUM_POINTS"), ",", fixed = TRUE)[[1]])
}
noise_types <- strsplit(Sys.getenv("SEFT_SIM_NOISE_TYPES", "grf,iid"), ",", fixed = TRUE)[[1]]
radius <- 12
signal_FWHM <- 24
noise_FWHM <- 4

sim_data_dir <- Sys.getenv(
    "SEFT_SIM_DATA_DIR",
    file.path("outputs", "intermediate", "simulation_data")
)
dir.create(sim_data_dir, recursive = TRUE, showWarnings = FALSE)

noise_file <- file.path(sim_data_dir, glue("3d_L{L}_FWHM{noise_FWHM}_noise.rds"))
if ("grf" %in% noise_types && !file.exists(noise_file)) {
    cat("Generating shared GRF noise bank...\n")
    set.seed(123)
    all_noise <- spatialnoise(
        dim = c(L, L, L),
        sigma = 1,
        nscan = length(mu_s) * length(nums_points) * B,
        method = "gaussRF",
        FWHM = noise_FWHM
    )
    saveRDS(all_noise, noise_file)
}
if ("grf" %in% noise_types) all_noise <- readRDS(noise_file)

# GRF noise data
id <- 1
if ("grf" %in% noise_types) for (num_points in nums_points) {
    for (mu in mu_s) {
        cat(glue("Generating 3D GRF data: mu={mu}, num_points={num_points}\n"))
        all_data <- vector("list", B)
        for (b in seq_len(B)) {
            x <- sample.int(L, num_points)
            y <- sample.int(L, num_points)
            z <- sample.int(L, num_points)
            grid_points <- matrix(c(x, y, z), ncol = 3)

            noise <- all_noise[, , , id]
            signal_data <- generate_3d_signal(
                dim = c(L, L, L),
                mu = mu,
                shape = "sphere",
                radius = radius,
                coord = grid_points,
                decay_method = "gaussian",
                mix_method = "max",
                FWHM = signal_FWHM
            )
            data <- list(
                Xs = signal_data$signal + noise,
                thetas = signal_data$mask,
                mu = mu
            )
            all_data[[glue("L{L}_mu{mu}_num_points{num_points}_b{b}")]] <- data
            id <- id + 1
        }
        saveRDS(all_data, file.path(sim_data_dir, glue("3d_L{L}_mu{mu}_num_points{num_points}_radius{radius}_sphere.rds")))
    }
}

# i.i.d. noise data
if ("iid" %in% noise_types) for (num_points in nums_points) {
    for (mu in mu_s) {
        cat(glue("Generating 3D iid data: mu={mu}, num_points={num_points}\n"))
        all_data <- vector("list", B)
        for (b in seq_len(B)) {
            x <- sample.int(L, num_points)
            y <- sample.int(L, num_points)
            z <- sample.int(L, num_points)
            grid_points <- matrix(c(x, y, z), ncol = 3)

            noise <- array(rnorm(L^3), dim = c(L, L, L))
            signal_data <- generate_3d_signal(
                dim = c(L, L, L),
                mu = mu,
                shape = "sphere",
                radius = radius,
                coord = grid_points,
                decay_method = "gaussian",
                mix_method = "max",
                FWHM = signal_FWHM
            )
            data <- list(
                Xs = signal_data$signal + noise,
                thetas = signal_data$mask,
                mu = mu
            )
            all_data[[glue("L{L}_mu{mu}_num_points{num_points}_b{b}")]] <- data
        }
        saveRDS(all_data, file.path(sim_data_dir, glue("3d_L{L}_mu{mu}_num_points{num_points}_radius{radius}_sphere_iid.rds")))
    }
}
