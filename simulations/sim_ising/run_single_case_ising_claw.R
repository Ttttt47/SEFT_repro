library(glue)
library(data.table)
library(optparse)

if (!file.exists(file.path("R", "core", "Ising_functions.R"))) {
  stop("Run this script from the SEFT project root.")
}
source(file.path("R", "core", "Ising_functions.R"))

library(optparse)

option_list <- list(
  make_option("--beta",       type="double",  default=NA),
  make_option("--h_true",     type="double",  default=NA),
  make_option("--mu",         type="double",  default=NA),
  make_option("--L",          type="integer", default=NA),
  make_option("--B",          type="integer", default=NA),
  make_option("--cal_scores", type="logical", default=FALSE),
  make_option("--prefix",     type="character", default=""),

  # optional
  make_option("--fixed_pi",   type="logical", default=FALSE),
  make_option("--pi_true",    type="double",  default=NA)
)


parser <- OptionParser(option_list = option_list)
opt <- parse_args(parser)
if (!opt$fixed_pi) opt$pi_true = NULL

beta        <- opt$beta
h_true      <- opt$h_true
mu          <- opt$mu
L           <- opt$L
B           <- opt$B
cal_scores  <- opt$cal_scores
prefix      <- opt$prefix

fixed_pi    <- opt$fixed_pi
pi_true     <- opt$pi_true


# -----------------------------
# fixed settings (same as dct_sub_task)
# -----------------------------
n <- 8
uprop_s <- c(0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5)
alpha <- 0.1

eps   <- 1e-32

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

sim_result_root <- file.path("outputs", "intermediate", "simulation_results")
sim_data_root <- file.path("outputs", "intermediate", "simulation_data")
sim_result_dir <- file.path(sim_result_root, prefix)
if (!dir.exists(sim_result_dir)) {
  dir.create(sim_result_dir, recursive = TRUE)
}

# -----------------------------
# load Ising simulation data
# -----------------------------
beta_tag <- sprintf("%.2f", beta)
h_tag    <- sprintf("%.2f", h_true)
mu_tag   <- sprintf("%.2f", mu)

if (fixed_pi) {
  pi_tag <- sprintf("%.2f", pi_true)
  data_file <- file.path(sim_data_root, glue("3d_Ising_data_beta{beta_tag}_pi{pi_tag}_mu{mu_tag}.rds"))
} else{
  data_file <- file.path(sim_data_root, glue("3d_Ising_data_beta{beta_tag}_h{h_tag}_mu{mu_tag}.rds"))
}
case_data <- readRDS(data_file)

# -----------------------------
# region masks (same as dct_sub_task)
# -----------------------------
region_masks <- array(-1L, dim = c(L, L, L))
nx <- ceiling(L / n)
for (i in 1:nx) {
  for (j in 1:nx) {
    for (k in 1:nx) {
      region_masks[((i - 1) * n + 1):min(i * n, L),
                   ((j - 1) * n + 1):min(j * n, L),
                   ((k - 1) * n + 1):min(k * n, L)] <- i + (j - 1) * nx + (k - 1) * nx^2
    }
  }
}
region_ids <- unique(as.vector(region_masks))

# -----------------------------
# neighbors (once)
# -----------------------------
mask_full <- array(TRUE, dim = c(L, L, L))
if (!exists("hmrf_build_neighbors_6n", mode = "function")) {
  stop("hmrf_build_neighbors_6n() not found. Please define it in Ising_functions.R.")
}
nb <- hmrf_build_neighbors_6n(mask_full)

# -----------------------------
# allocate outputs (same as dct_sub_task)
# -----------------------------
FDP_table <- array(-1, dim = c(B, length(para_sets), length(uprop_s)))
MSR_table <- array(-1, dim = c(B, length(para_sets), length(uprop_s)))

if (fixed_pi) {
  scores_path <- file.path(sim_result_dir, glue("scores_L{L}_beta{beta_tag}_pi{pi_tag}_mu{mu_tag}.rds"))
} else{
  scores_path <- file.path(sim_result_dir, glue("scores_L{L}_beta{beta_tag}_h{h_tag}_mu{mu_tag}.rds"))
}
if (cal_scores) {
  scores_list <- vector("list", B)
} else {
  scores_list <- readRDS(scores_path)
}

all_PC_e_values   <- array(NA, dim = c(B, length(uprop_s), length(para_sets), length(region_ids)))
all_PC_p_values   <- array(NA, dim = c(B, length(uprop_s), length(para_sets), length(region_ids)))
all_set_theta     <- array(NA, dim = c(B, length(uprop_s), length(region_ids)))
all_PC_decisions  <- array(NA, dim = c(B, length(uprop_s), length(para_sets), length(region_ids)))

# -----------------------------
# helper: robust extraction of b-th replication
# -----------------------------
extract_rep <- function(obj, b) {
  dat <- obj[[b]]
  if (is.list(dat) && length(dat) == 1 && is.list(dat[[1]]) && !is.null(dat[[1]]$x3d)) {
    dat <- dat[[1]]
  }
  dat
}

# -----------------------------
# main loop
# -----------------------------
for (b in 1:B) {
  if (fixed_pi) {
    cat(glue("L{L}_beta{beta_tag}_pi{pi_tag}_mu{mu_tag}_b{b} start.\n"))
  } else{
    cat(glue("L{L}_beta{beta_tag}_h{h_tag}_mu{mu_tag}_b{b} start.\n"))
  }

  sim_dat <- extract_rep(case_data, b)

  x3d <- sim_dat$x3d
  if (is.null(x3d)) stop("Missing x3d in simulation data element.")
  if (!all(dim(x3d) == c(L, L, L))) x3d <- array(as.vector(x3d), dim = c(L, L, L))

  theta_true <- sim_dat$theta
  if (is.null(theta_true)) stop("Missing theta in simulation data element.")
  if (!all(dim(theta_true) == c(L, L, L))) theta_true <- array(as.integer(theta_true), dim = c(L, L, L))

  dat <- list(Xs = x3d, thetas = theta_true)

  # mirror iid map
  x_til <- array(rnorm(L^3), dim = c(L, L, L))

  if (cal_scores) {
    # CLAW scores
    t_data = x3d
    t_data_til = x_til

    # baseline iid mirror
    t_data_til <- array(rnorm(prod(dim(t_data))), dim = dim(t_data))

    # prevent 0s (your original logic)
    t_data[t_data == 0] <- 1e-10
    t_data_til[t_data_til == 0] <- 1e-10

    # bandwidth h
    h <- density(sample(c(t_data_til, t_data), 10^3), bw = 'nrd0')$bw

    results <- cal_CLAW_scores_3d(
      t_data, t_data_til,
      2 * (1 - pnorm(abs(t_data))),
      2 * (1 - pnorm(abs(t_data_til))),
      h = h, bandwidth = 5, lambda = 0.5, neighbor_range = 10
    )

    # score = posterior non-null probability
    R     <- pmax(results$R, eps)
    R_til <- pmax(results$R_til, eps)

    scores_list[[b]] <- list(R = R, R_til = R_til)
  } else {
    R     <- scores_list[[b]]$R
    R_til <- scores_list[[b]]$R_til
  }

  # prevent exact zeros
  x3d[x3d == 0]     <- eps
  x_til[x_til == 0] <- eps

  for (i_uprop in seq_along(uprop_s)) {
    cat(glue("{i_uprop}/{length(uprop_s)} u testing start.\n"))

    PC_e_values  <- array(NA, dim = c(length(para_sets), length(region_ids)))
    PC_p_values  <- array(NA, dim = c(length(para_sets), length(region_ids)))
    PC_decisions <- array(NA, dim = c(length(para_sets), length(region_ids)))
    set_theta    <- array(NA, dim = c(length(region_ids)))

    for (i_region in 1:length(region_ids)) {
      region_id <- region_ids[i_region]
      u <- ceiling(sum(region_masks == region_id) * uprop_s[i_uprop])

      set_theta[i_region] <- merge_PC_theta(dat$thetas[region_masks == region_id], u)

      for (k in 1:length(para_sets)) {
        if (para_sets[[k]][[1]] %in% c("Simes", "Fisher")) {
          pvals_region <- 2 * (1 - pnorm(abs(x3d[region_masks == region_id])))
          if (para_sets[[k]][[1]] == "Simes") {
            PC_p_values[k, i_region] <- Simes_PC_test_cpp(pvals_region, u)
          } else {
            PC_p_values[k, i_region] <- Fisher_PC_test_cpp(pvals_region, u)
          }
          next
        }

        mth <- para_sets[[k]][[1]]
        til_mth <- para_sets[[k]][[2]]
        signmin_flag <- para_sets[[k]][[3]]

        # 2 x m matrix of log-scores (same format as dct_sub_task)
        log_PLIS_scores <- matrix(
          c(R[region_masks == region_id], R_til[region_masks == region_id]),
          nrow = 2, byrow = TRUE
        )
        log_PLIS_scores <- log_PLIS_scores / max(log_PLIS_scores)
        log_PLIS_scores <- log(log_PLIS_scores)

        PC_e_values[k, i_region] <- cal_ELIS_cpp(
          log_PLIS_scores, u,
          method = mth, til_mth = til_mth, signmin_flag = signmin_flag
        )
        PC_p_values[k, i_region] <- 1 / PC_e_values[k, i_region]
      }
    }

    all_set_theta[b, i_uprop, ] <- set_theta

    for (k in 1:length(para_sets)) {
      PC_decisions[k, ] <- BH(PC_p_values[k, ], alpha)
      FDP_table[b, k, i_uprop] <- cal_FDP(set_theta, PC_decisions[k, ])
      MSR_table[b, k, i_uprop] <- cal_MSR(set_theta, PC_decisions[k, ])
    }
    print('FNR:')
    print(MSR_table[b, , i_uprop])
    print('FDP:')
    print(FDP_table[b, , i_uprop])

    all_PC_e_values[b, i_uprop, , ]  <- PC_e_values
    all_PC_p_values[b, i_uprop, , ]  <- PC_p_values
    all_PC_decisions[b, i_uprop, , ] <- PC_decisions
  }
}

# -----------------------------
# save outputs (same style)
# -----------------------------
if (fixed_pi) {
  table_dir <- file.path(sim_result_dir, glue("MSR_FDP_L{L}_beta{beta_tag}_pi{pi_tag}_mu{mu_tag}.rds"))
  PC_res_dir <- file.path(sim_result_dir, glue("PC_results_L{L}_beta{beta_tag}_pi{pi_tag}_mu{mu_tag}.rds"))
} else {
  table_dir <- file.path(sim_result_dir, glue("MSR_FDP_L{L}_beta{beta_tag}_h{h_tag}_mu{mu_tag}.rds"))
  PC_res_dir <- file.path(sim_result_dir, glue("PC_results_L{L}_beta{beta_tag}_h{h_tag}_mu{mu_tag}.rds"))
}


saveRDS(
  list(MSR = MSR_table, FSR = FDP_table), 
  table_dir
)

if (cal_scores) {
  saveRDS(scores_list, scores_path)
}

saveRDS(
  list(all_PC_e_values = all_PC_e_values,
       all_PC_p_values = all_PC_p_values,
       all_set_theta = all_set_theta,
       all_PC_decisions = all_PC_decisions),
  PC_res_dir
)
