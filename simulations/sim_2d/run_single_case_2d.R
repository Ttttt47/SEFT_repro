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
    stop("Cannot locate SEFT project root. Run from project root or via Rscript.")
}

ensure_project_root()
source(file.path("R", "core", "CLAW_functions.R"))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 6) {
    stop("Usage: Rscript simulations/sim_2d/run_single_case_2d.R <mu> <L> <shape> <B> <cal_scores> <prefix>")
}

mu <- as.numeric(args[1])
L <- as.numeric(args[2])
shape <- as.character(args[3])
B <- as.numeric(args[4])
cal_scores <- as.logical(args[5])
prefix <- as.character(args[6])

n_div_s <- c(10, 20)
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

case_data <- readRDS(file.path(sim_data_dir, glue("L{L}_mu{mu}_shape{shape}.rds")))

FDP_table <- array(-1, dim = c(B, length(para_sets), length(n_div_s), length(uprop_s)))
MSR_table <- array(-1, dim = c(B, length(para_sets), length(n_div_s), length(uprop_s)))
all_PC_e_values <- list()
all_PC_p_values <- list()
all_set_theta <- list()
all_PC_decisions <- list()

if (cal_scores) {
    scores_list <- vector("list", B)
} else {
    scores_list <- readRDS(file.path(sim_result_dir, glue("scores_L{L}_mu{mu}_shape{shape}.rds")))
}

for (b in seq_len(B)) {
    print(glue("L{L}_mu{mu}_shape{shape}_b{b} score calculation start."))
    data <- case_data[[glue("L{L}_mu{mu}_shape{shape}_b{b}")]]
    t_data <- data$Xs
    t_data_til <- array(rnorm(prod(dim(t_data))), dim = dim(t_data))
    h <- density(sample(c(t_data_til, t_data), 20^2), bw = "nrd0")$bw

    if (cal_scores) {
        results <- cal_CLAW_scores_2d(
            t_data, t_data_til,
            2 * (1 - pnorm(abs(t_data))),
            2 * (1 - pnorm(abs(t_data_til))),
            h = h, bandwidth = 10, lambda = 0.5, neighbor_range = 20
        )
        scores_list[[b]] <- results
    } else {
        results <- scores_list[[b]]
    }

    for (i_n_div in seq_along(n_div_s)) {
        print(glue("{i_n_div}/{length(n_div_s)} region division testing start."))
        for (i_uprop in seq_along(uprop_s)) {
            print(glue("{i_uprop}/{length(uprop_s)} u testing start."))
            n <- L / n_div_s[i_n_div]
            u <- ceiling(uprop_s[i_uprop] * n^2)
            region_masks <- matrix(0, nrow = L, ncol = L)
            for (i in seq_len(L / n)) {
                for (j in seq_len(L / n)) {
                    region_masks[((i - 1) * n + 1):(i * n), ((j - 1) * n + 1):(j * n)] <- (i - 1) + (j - 1) * (L / n) + 1
                }
            }

            PC_e_values <- array(NA, dim = c(length(para_sets), (L / n)^2))
            PC_p_values <- array(NA, dim = c(length(para_sets), (L / n)^2))
            PC_decisions <- array(NA, dim = c(length(para_sets), (L / n)^2))
            set_theta <- array(NA, dim = c((L / n)^2))

            for (i in seq_len(L / n)) {
                for (j in seq_len(L / n)) {
                    set_id <- (i - 1) + (j - 1) * (L / n) + 1
                    for (k in seq_along(para_sets)) {
                        if (para_sets[[k]][[1]] %in% c("Simes", "Fisher")) {
                            region_pvals <- 2 * (1 - pnorm(abs(t_data[region_masks == set_id])))
                            if (para_sets[[k]][[1]] == "Simes") {
                                PC_p_values[k, set_id] <- Simes_PC_test_cpp(region_pvals, u)
                            } else {
                                PC_p_values[k, set_id] <- Fisher_PC_test_cpp(region_pvals, u)
                            }
                            next
                        }

                        mth <- para_sets[[k]][[1]]
                        til_mth <- para_sets[[k]][[2]]
                        signmin_flag <- para_sets[[k]][[3]]
                        log_PLIS_scores <- matrix(
                            c(results$R[region_masks == set_id], results$R_til[region_masks == set_id]),
                            nrow = 2,
                            byrow = TRUE
                        )
                        log_PLIS_scores <- log_PLIS_scores / max(log_PLIS_scores)
                        log_PLIS_scores <- log(log_PLIS_scores)
                        PC_e_values[k, set_id] <- cal_ELIS_cpp(
                            log_PLIS_scores,
                            u,
                            method = mth,
                            til_mth = til_mth,
                            signmin_flag = signmin_flag
                        )
                        PC_p_values[k, set_id] <- 1 / PC_e_values[k, set_id]
                    }
                    set_theta[set_id] <- merge_PC_theta(data$thetas[region_masks == set_id], u)
                }
            }

            for (k in seq_along(para_sets)) {
                PC_decisions[k, ] <- BH(PC_p_values[k, ], alpha)
                FDP_table[b, k, i_n_div, i_uprop] <- cal_FDP(set_theta, PC_decisions[k, ])
                MSR_table[b, k, i_n_div, i_uprop] <- cal_MSR(set_theta, PC_decisions[k, ])
            }

            all_PC_e_values[[glue("{b}_{i_n_div}_{i_uprop}")]] <- PC_e_values
            all_PC_p_values[[glue("{b}_{i_n_div}_{i_uprop}")]] <- PC_p_values
            all_set_theta[[glue("{b}_{i_n_div}_{i_uprop}")]] <- set_theta
            all_PC_decisions[[glue("{b}_{i_n_div}_{i_uprop}")]] <- PC_decisions
        }
    }
}

saveRDS(list(MSR = MSR_table, FSR = FDP_table), file.path(sim_result_dir, glue("MSR_FDP_L{L}_mu{mu}_shape{shape}.rds")))
if (cal_scores) {
    saveRDS(scores_list, file.path(sim_result_dir, glue("scores_L{L}_mu{mu}_shape{shape}.rds")))
}
saveRDS(
    list(
        all_PC_e_values = all_PC_e_values,
        all_PC_p_values = all_PC_p_values,
        all_set_theta = all_set_theta,
        all_PC_decisions = all_PC_decisions
    ),
    file.path(sim_result_dir, glue("PC_results_L{L}_mu{mu}_shape{shape}.rds"))
)
