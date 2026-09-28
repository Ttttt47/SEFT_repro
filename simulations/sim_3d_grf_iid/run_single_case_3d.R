library(glue)
library(data.table)

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

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 9) {
    stop("Usage: Rscript simulations/sim_3d_grf_iid/run_single_case_3d.R <mu> <num_points> <L> <shape> <radius> <B> <cal_scores> <denoise_level> <prefix>")
}

mu <- as.numeric(args[1])
num_points <- as.numeric(args[2])
L <- as.numeric(args[3])
shape <- as.character(args[4])
radius <- as.numeric(args[5])
B <- as.numeric(args[6])
cal_scores <- as.logical(args[7])
denoise_level <- as.numeric(args[8])
prefix <- as.character(args[9])

n <- 8
uprop_s <- c(0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5)
alpha <- 0.1
para_sets <- list(
    list("Sym_miny", "oneminus", TRUE),
    list("Sym_adapt", "neglog", FALSE),
    list("Sym_adapt_grid", "neglog", FALSE),
    list("Sym_miny", "neglog", FALSE),
    list("Avg", "oneminus", FALSE),
    list("Sym_fdr", "oneminus", FALSE),
    list("Sym_fdr", "neglog", FALSE),
    list("Simes"),
    list("Fisher")
)[c(2, 3, 4, 8, 9)]

sim_data_dir <- Sys.getenv(
    "SEFT_SIM_DATA_DIR",
    file.path("outputs", "intermediate", "simulation_data")
)
sim_result_root <- Sys.getenv(
    "SEFT_SIM_RESULT_ROOT",
    file.path("outputs", "intermediate", "simulation_results")
)
sim_result_dir <- file.path(sim_result_root, prefix)
dir.create(sim_result_dir, recursive = TRUE, showWarnings = FALSE)

if (grepl("iid", prefix)) {
    case_data <- readRDS(file.path(sim_data_dir, glue("3d_L{L}_mu{mu}_num_points{num_points}_radius{radius}_{shape}_iid.rds")))
} else {
    case_data <- readRDS(file.path(sim_data_dir, glue("3d_L{L}_mu{mu}_num_points{num_points}_radius{radius}_{shape}.rds")))
}

region_masks <- array(-1, dim = c(L, L, L))
for (i in seq_len(ceiling(L / n))) {
    for (j in seq_len(ceiling(L / n))) {
        for (k in seq_len(ceiling(L / n))) {
            region_masks[((i - 1) * n + 1):min(i * n, L), ((j - 1) * n + 1):min(j * n, L), ((k - 1) * n + 1):min(k * n, L)] <- i + (j - 1) * ceiling(L / n) + (k - 1) * ceiling(L / n)^2
        }
    }
}
region_ids <- unique(as.vector(region_masks))

FDP_table <- array(-1, dim = c(B, length(para_sets), length(uprop_s)))
MSR_table <- array(-1, dim = c(B, length(para_sets), length(uprop_s)))
if (cal_scores) {
    scores_list <- vector("list", B)
} else {
    scores_list <- readRDS(file.path(sim_result_dir, glue("scores_L{L}_mu{mu}_num_points{num_points}_shape{shape}.rds")))
}

all_PC_e_values <- array(NA, dim = c(B, length(uprop_s), length(para_sets), length(region_ids)))
all_PC_p_values <- array(NA, dim = c(B, length(uprop_s), length(para_sets), length(region_ids)))
all_set_theta <- array(NA, dim = c(B, length(uprop_s), length(region_ids)))
all_PC_decisions <- array(NA, dim = c(B, length(uprop_s), length(para_sets), length(region_ids)))

for (b in seq_len(B)) {
    print(glue("L{L}_mu{mu}_shape{shape}_num_points{num_points}_b{b} score calculation start."))
    data <- case_data[[glue("L{L}_mu{mu}_num_points{num_points}_b{b}")]]
    t_data <- data$Xs

    if (denoise_level != 0) {
        t_data <- apply_wavelet_denoising_3d(t_data, wf = "d8", target_level = denoise_level, J = 6, verbose = FALSE)
    }

    t_data_til <- array(rnorm(prod(dim(t_data))), dim = dim(t_data))
    t_data[t_data == 0] <- 1e-10
    t_data_til[t_data_til == 0] <- 1e-10

    h <- density(sample(c(t_data_til, t_data), 10^3), bw = "nrd0")$bw
    if (cal_scores) {
        results <- cal_CLAW_scores_3d(
            t_data, t_data_til,
            2 * (1 - pnorm(abs(t_data))),
            2 * (1 - pnorm(abs(t_data_til))),
            h = h, bandwidth = 5, lambda = 0.5, neighbor_range = 10
        )
        scores_list[[b]] <- results
    } else {
        results <- scores_list[[b]]
    }

    for (i_uprop in seq_along(uprop_s)) {
        print(glue("{i_uprop}/{length(uprop_s)} u testing start."))
        PC_e_values <- array(NA, dim = c(length(para_sets), length(region_ids)))
        PC_p_values <- array(NA, dim = c(length(para_sets), length(region_ids)))
        PC_decisions <- array(NA, dim = c(length(para_sets), length(region_ids)))
        set_theta <- array(NA, dim = c(length(region_ids)))

        for (i_region in seq_along(region_ids)) {
            region_id <- region_ids[i_region]
            u <- ceiling(sum(region_masks == region_id) * uprop_s[i_uprop])
            set_theta[i_region] <- merge_PC_theta(data$thetas[region_masks == region_id], u)

            for (k in seq_along(para_sets)) {
                if (para_sets[[k]][[1]] %in% c("Simes", "Fisher")) {
                    region_pvals <- 2 * (1 - pnorm(abs(t_data[region_masks == region_id])))
                    if (para_sets[[k]][[1]] == "Simes") {
                        PC_p_values[k, i_region] <- Simes_PC_test_cpp(region_pvals, u)
                    } else {
                        PC_p_values[k, i_region] <- Fisher_PC_test_cpp(region_pvals, u)
                    }
                    next
                }

                mth <- para_sets[[k]][[1]]
                til_mth <- para_sets[[k]][[2]]
                signmin_flag <- para_sets[[k]][[3]]
                log_PLIS_scores <- matrix(c(results$R[region_masks == region_id], results$R_til[region_masks == region_id]), nrow = 2, byrow = TRUE)
                log_PLIS_scores <- log_PLIS_scores / max(log_PLIS_scores)
                log_PLIS_scores <- log(log_PLIS_scores)
                PC_e_values[k, i_region] <- cal_ELIS_cpp(log_PLIS_scores, u, method = mth, til_mth = til_mth, signmin_flag = signmin_flag)
                PC_p_values[k, i_region] <- 1 / PC_e_values[k, i_region]
            }
        }

        all_set_theta[b, i_uprop, ] <- set_theta
        for (k in seq_along(para_sets)) {
            PC_decisions[k, ] <- BH(PC_p_values[k, ], alpha)
            FDP_table[b, k, i_uprop] <- cal_FDP(set_theta, PC_decisions[k, ])
            MSR_table[b, k, i_uprop] <- cal_MSR(set_theta, PC_decisions[k, ])
        }

        all_PC_e_values[b, i_uprop, , ] <- PC_e_values
        all_PC_p_values[b, i_uprop, , ] <- PC_p_values
        all_PC_decisions[b, i_uprop, , ] <- PC_decisions
    }
}

saveRDS(list(MSR = MSR_table, FSR = FDP_table), file.path(sim_result_dir, glue("MSR_FDP_L{L}_mu{mu}_num_points{num_points}_shape{shape}.rds")))
if (cal_scores) {
    saveRDS(scores_list, file.path(sim_result_dir, glue("scores_L{L}_mu{mu}_num_points{num_points}_shape{shape}.rds")))
}
saveRDS(
    list(
        all_PC_e_values = all_PC_e_values,
        all_PC_p_values = all_PC_p_values,
        all_set_theta = all_set_theta,
        all_PC_decisions = all_PC_decisions
    ),
    file.path(sim_result_dir, glue("PC_results_L{L}_mu{mu}_num_points{num_points}_shape{shape}.rds"))
)
