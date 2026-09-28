library(glue)

ensure_project_root <- function() {
    if (file.exists(file.path("R", "core", "CLAW_functions.R"))) return(invisible())
    file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
    if (length(file_arg) > 0) {
        script_path <- sub("^--file=", "", file_arg[1])
        candidate <- normalizePath(file.path(dirname(script_path), "..", ".."), mustWork = FALSE)
        if (file.exists(file.path(candidate, "R", "core", "CLAW_functions.R"))) {
            setwd(candidate)
            return(invisible())
        }
    }
    stop("Cannot locate SEFT project root.")
}

ensure_project_root()
source(file.path("R", "core", "CLAW_functions.R"))

L <- 500
mu_s <- seq(1.5, 3.5, 0.2)
shape_s <- c("disk", "iid")
B <- as.integer(Sys.getenv("SEFT_SIM_B", "100"))
if (nzchar(Sys.getenv("SEFT_SIM_MUS"))) {
    mu_s <- as.numeric(strsplit(Sys.getenv("SEFT_SIM_MUS"), ",", fixed = TRUE)[[1]])
}
if (nzchar(Sys.getenv("SEFT_SIM_SHAPES"))) {
    shape_s <- strsplit(Sys.getenv("SEFT_SIM_SHAPES"), ",", fixed = TRUE)[[1]]
}

sim_data_dir <- Sys.getenv(
    "SEFT_SIM_DATA_DIR",
    file.path("outputs", "intermediate", "simulation_data")
)
dir.create(sim_data_dir, recursive = TRUE, showWarnings = FALSE)

for (mu in mu_s) {
    for (shape in shape_s) {
        cat(glue("Generating 2D data: L={L}, mu={mu}, shape={shape}\n"))
        all_data <- vector("list", B)
        for (b in seq_len(B)) {
            data <- generate_dense_shape_2d_data(L, mu, shape = shape, relsize = 1 / 2.5, sparsity = 0.3)
            all_data[[glue("L{L}_mu{mu}_shape{shape}_b{b}")]] <- data
        }
        saveRDS(all_data, file.path(sim_data_dir, glue("L{L}_mu{mu}_shape{shape}.rds")))
    }
}
