library(glue)
library(data.table)
library(cowplot)
library(ggplot2)
library(ComplexHeatmap)
library(plot3D)

if (!file.exists(file.path("R", "core", "CLAW_functions.R"))) {
    stop("Run this script from the SEFT project root.")
}
source(file.path("R", "core", "CLAW_functions.R"))

mu_s <- seq(from = 2, to = 5, by = 0.5)
L <- 64
shape <- "sphere"
num_points_s <- c(10, 20, 30)
uprop_s <- c(0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5)
alpha <- 0.1
n <- 8
B <- 100

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

sim_data_dir <- file.path("outputs", "intermediate", "simulation_data")
sim_result_root <- file.path("outputs", "intermediate", "simulation_results")
figure_dir <- file.path("outputs", "figures")
dir.create(figure_dir, recursive = TRUE, showWarnings = FALSE)

build_summary_tables <- function(prefix) {
    all_MSR_table <- array(-1, dim = c(length(mu_s), length(num_points_s), length(para_sets), length(uprop_s)))
    all_FSR_table <- array(-1, dim = c(length(mu_s), length(num_points_s), length(para_sets), length(uprop_s)))

    for (i_mu in seq_along(mu_s)) {
        for (i_num_points in seq_along(num_points_s)) {
            mu <- mu_s[i_mu]
            num_points <- num_points_s[i_num_points]
            tables <- readRDS(file.path(sim_result_root, prefix, glue("MSR_FDP_L{L}_mu{mu}_num_points{num_points}_shape{shape}.rds")))
            all_MSR_table[i_mu, i_num_points, , ] <- apply(tables[[1]], MARGIN = c(2, 3), FUN = mean)
            all_FSR_table[i_mu, i_num_points, , ] <- apply(tables[[2]], MARGIN = c(2, 3), FUN = mean)
        }
    }
    list(MSR = all_MSR_table, FSR = all_FSR_table)
}

make_seftq0_plots <- function(summary_tables, fnr_name, fdr_name) {
    method_names <- c("SEFT", "BHe+Simes", "BHe+Fisher")
    method_colors <- c("SEFT" = "black", "BHe+Simes" = "green", "BHe+Fisher" = "purple")
    method_shapes <- c("SEFT" = 15, "BHe+Simes" = 18, "BHe+Fisher" = 4)

    MSR_plot_list <- list()
    FSR_plot_list <- list()

    for (i_uprop in c(1, 3, 5, 7)) {
        for (i_num_points in seq_along(num_points_s)) {
            uprop <- uprop_s[i_uprop]
            num_points <- num_points_s[i_num_points]

            msr_dt <- rbindlist(lapply(seq_along(method_names), function(i) {
                data.table(
                    mu = mu_s,
                    MSR = summary_tables$MSR[, i_num_points, i + 2, i_uprop],
                    method = method_names[i]
                )
            }))

            fsr_dt <- rbindlist(lapply(seq_along(method_names), function(i) {
                data.table(
                    mu = mu_s,
                    FSR = summary_tables$FSR[, i_num_points, i + 2, i_uprop],
                    method = method_names[i]
                )
            }))

            key <- glue("num_points{num_points}_uprop{uprop}")
            MSR_plot_list[[key]] <- ggplot(msr_dt, aes(x = mu, y = MSR, color = method, shape = method, group = method)) +
                geom_line() +
                geom_point(size = 2) +
                scale_color_manual(values = method_colors) +
                scale_shape_manual(values = method_shapes) +
                labs(title = glue("u proportion={uprop}, {num_points} clusters"), x = "mu", y = "FNR") +
                scale_y_continuous(limits = c(0, 1)) +
                theme_bw() +
                theme(legend.position = "bottom")

            FSR_plot_list[[key]] <- ggplot(fsr_dt, aes(x = mu, y = FSR, color = method, shape = method, group = method)) +
                geom_line() +
                geom_point(size = 2) +
                geom_hline(aes(yintercept = alpha), colour = "red", linetype = "dashed") +
                scale_color_manual(values = method_colors) +
                scale_shape_manual(values = method_shapes) +
                labs(title = glue("u proportion={uprop}, {num_points} clusters"), x = "mu", y = "FDR") +
                scale_y_continuous(limits = c(0, 0.5)) +
                theme_bw() +
                theme(legend.position = "bottom")
        }
    }

    pdf(file.path(figure_dir, fnr_name), width = 12, height = 6)
    combined_plot <- plot_grid(
        plotlist = lapply(MSR_plot_list, function(p) p + theme(legend.position = "none")),
        nrow = length(num_points_s),
        ncol = length(c(1, 3, 5, 7)),
        align = "hv",
        byrow = FALSE
    )
    final_plot <- plot_grid(combined_plot, get_legend(MSR_plot_list[[1]]), nrow = 2, rel_heights = c(3, 0.3))
    print(final_plot)
    dev.off()

    pdf(file.path(figure_dir, fdr_name), width = 12, height = 6)
    combined_plot <- plot_grid(
        plotlist = lapply(FSR_plot_list, function(p) p + theme(legend.position = "none")),
        nrow = length(num_points_s),
        ncol = length(c(1, 3, 5, 7)),
        align = "hv",
        byrow = FALSE
    )
    final_plot <- plot_grid(combined_plot, get_legend(FSR_plot_list[[1]]), nrow = 2, rel_heights = c(3, 0.3))
    print(final_plot)
    dev.off()
}

# 3D iid figures
iid_summary <- build_summary_tables("3d_iid")
make_seftq0_plots(iid_summary, "L64_shapesphere_FNR_iid_SEFTq0.pdf", "L64_shapesphere_FDR_iid_SEFTq0.pdf")

# 3D GRF with denoising figures
grf_denoised_summary <- build_summary_tables("3d_grf_denoised")
make_seftq0_plots(grf_denoised_summary, "L64_shapesphere_FNR_denoisedGRF_SEFTq0.pdf", "L64_shapesphere_FDR_denoisedGRF_SEFTq0.pdf")

# 3D GRF without denoising figures
grf_nodenoise_summary <- build_summary_tables("3d_grf_nodenoise")
make_seftq0_plots(grf_nodenoise_summary, "L64_shapesphere_FNR_SEFTq0.pdf", "L64_shapesphere_FDR_SEFTq0.pdf")

# 3D demonstration figure (denoised GRF)
mu_demo <- 4
num_points_demo <- 20
radius_demo <- 12
b_demo <- 1
demo_prefix <- "3d_grf_denoised"

case_data <- readRDS(file.path(sim_data_dir, glue("3d_L{L}_mu{mu_demo}_num_points{num_points_demo}_radius{radius_demo}_{shape}.rds")))
data_demo <- case_data[[glue("L{L}_mu{mu_demo}_num_points{num_points_demo}_b{b_demo}")]]

region_masks <- array(-1, dim = c(L, L, L))
for (i in seq_len(ceiling(L / n))) {
    for (j in seq_len(ceiling(L / n))) {
        for (k in seq_len(ceiling(L / n))) {
            region_masks[((i - 1) * n + 1):min(i * n, L), ((j - 1) * n + 1):min(j * n, L), ((k - 1) * n + 1):min(k * n, L)] <- i + (j - 1) * ceiling(L / n) + (k - 1) * ceiling(L / n)^2
        }
    }
}
region_ids <- unique(as.vector(region_masks))
all_set_theta <- array(NA, dim = c(B, length(uprop_s), length(region_ids)))

for (i_uprop in seq_along(uprop_s)) {
    set_theta <- array(NA, dim = c(length(region_ids)))
    for (i_region in seq_along(region_ids)) {
        region_id <- region_ids[i_region]
        u <- ceiling(sum(region_masks == region_id) * uprop_s[i_uprop])
        set_theta[i_region] <- merge_PC_theta(data_demo$thetas[region_masks == region_id], u)
    }
    all_set_theta[b_demo, i_uprop, ] <- set_theta
}

d <- L / n
coords <- expand.grid(x = 1:d, y = 1:d, z = 1:d)
coords$region_id <- coords$x + (coords$y - 1) * d + (coords$z - 1) * d^2

pdf(file.path(figure_dir, "3d_demenstration_denoised.pdf"), height = 7, width = 14)
par(mfrow = c(1, 2))
for (i_uprop in c(4, 7)) {
    PC_results <- readRDS(file.path(sim_result_root, demo_prefix, glue("PC_results_L{L}_mu{mu_demo}_num_points{num_points_demo}_shape{shape}.rds")))
    PC_decisions <- PC_results$all_PC_decisions[b_demo, i_uprop, , ]
    set_theta <- all_set_theta[b_demo, i_uprop, ]

    ids_truth <- which(set_theta == 1)
    ids_seft <- which(PC_decisions[2, ] == 1)
    ids_bh <- which(PC_decisions[4, ] == 1)

    scatter3D(
        coords$x[ids_truth], coords$y[ids_truth], coords$z[ids_truth],
        cex = 1.2, theta = 60, phi = 10, colvar = NULL, col = "grey",
        bty = "b2", pch = 19, main = glue("u proportion={uprop_s[i_uprop]}, mu={mu_demo}")
    )
    scatter3D(coords$x[ids_bh], coords$y[ids_bh], coords$z[ids_bh], cex = 1.2, theta = 60, phi = 10, colvar = NULL, col = "green", bty = "b2", pch = 2, add = TRUE)
    scatter3D(coords$x[ids_seft], coords$y[ids_seft], coords$z[ids_seft], cex = 1.2, theta = 60, phi = 10, colvar = NULL, col = "red", bty = "b2", pch = 4, add = TRUE)
    legend(x = 0.27, y = -0.37, inset = .05, c("BHe", "SEFT", "Ground truth"), pch = c(2, 4, 19), col = c("green", "red", "grey"), bty = "n")
}
dev.off()

# Symmetrization figure
sym_prefix <- "3d_iid"
sym_para_sets <- list(
    list("Sym_miny", "oneminus", TRUE),
    list("Sym_miny", "neglog", FALSE),
    list("Sym_miny", "oneminus", FALSE)
)

region_masks <- array(-1, dim = c(L, L, L))
for (i in seq_len(ceiling(L / n))) {
    for (j in seq_len(ceiling(L / n))) {
        for (k in seq_len(ceiling(L / n))) {
            region_masks[((i - 1) * n + 1):min(i * n, L), ((j - 1) * n + 1):min(j * n, L), ((k - 1) * n + 1):min(k * n, L)] <- i + (j - 1) * ceiling(L / n) + (k - 1) * ceiling(L / n)^2
        }
    }
}
region_ids <- unique(as.vector(region_masks))

all_PC_e_values <- array(NA, dim = c(length(sym_para_sets), B, length(region_ids), length(mu_s), length(num_points_s)))
for (i_mu in seq_along(mu_s)) {
    for (i_num_points in seq_along(num_points_s)) {
        mu <- mu_s[i_mu]
        num_points <- num_points_s[i_num_points]
        scores_list <- readRDS(file.path(sim_result_root, sym_prefix, glue("scores_L{L}_mu{mu}_num_points{num_points}_shape{shape}.rds")))

        for (b in seq_len(B)) {
            results <- scores_list[[b]]
            for (k in seq_along(sym_para_sets)) {
                mth <- sym_para_sets[[k]][[1]]
                til_mth <- sym_para_sets[[k]][[2]]
                signmin_flag <- sym_para_sets[[k]][[3]]
                u <- 1
                for (i_region in seq_along(region_ids)) {
                    region_id <- region_ids[i_region]
                    log_PLIS_scores <- matrix(c(results$R[region_masks == region_id], results$R_til[region_masks == region_id]), nrow = 2, byrow = TRUE)
                    log_PLIS_scores <- log_PLIS_scores / max(log_PLIS_scores)
                    log_PLIS_scores <- log(log_PLIS_scores)
                    all_PC_e_values[k, b, i_region, i_mu, i_num_points] <- cal_ELIS_cpp(
                        log_PLIS_scores,
                        u,
                        method = mth,
                        til_mth = til_mth,
                        signmin_flag = signmin_flag
                    )
                }
            }
        }
    }
}

average_PC_e_values <- apply(all_PC_e_values, MARGIN = c(1, 4, 5), FUN = function(x) mean(x, na.rm = TRUE))
average_PC_e_values_dt <- data.table::melt(average_PC_e_values, varnames = c("Method", "mu", "num_points"), value.name = "Average_e_value")
average_PC_e_values_dt$Method <- factor(
    average_PC_e_values_dt$Method,
    levels = 1:length(sym_para_sets),
    labels = c("Signed maximum", "Subtraction of logarithm", "Subtraction")
)
average_PC_e_values_dt$mu <- mu_s[average_PC_e_values_dt$mu]

method_shapes <- c("Subtraction of logarithm" = 15, "Signed maximum" = 18, "Subtraction" = 4)
plot_list <- list()
for (i_num_points in seq_along(num_points_s)) {
    p <- ggplot(
        average_PC_e_values_dt[average_PC_e_values_dt$num_points == i_num_points, ],
        aes(x = mu, y = Average_e_value, color = Method, shape = Method, linetype = Method)
    ) +
        labs(title = glue("Number of clusters: {num_points_s[i_num_points]}"), x = "mu", y = "Average e-value") +
        scale_shape_manual(values = method_shapes) +
        geom_line() +
        geom_point(size = 2) +
        scale_y_continuous(limits = c(0, 220)) +
        theme_bw() +
        theme(legend.position = "bottom", legend.title = element_blank())
    plot_list[[length(plot_list) + 1]] <- p
}

pdf(file.path(figure_dir, "Comparing_symmetrization_by_evalue_iid.pdf"), width = 13 / 1.2, height = 3 / 1.2)
combined_plot <- plot_grid(plotlist = lapply(plot_list, function(p) p + theme(legend.position = "none")), nrow = length(num_points_s), align = "hv", byrow = TRUE)
final_plot <- plot_grid(combined_plot, get_legend(plot_list[[1]]), nrow = 2, rel_heights = c(3, 0.3))
print(final_plot)
dev.off()
