#!/usr/bin/env Rscript

# Recreate the legacy E.6 demonstration and no-denoise comparison figures
# from a minimal simulation run. This keeps the original plotting logic but
# permits B=1 instead of requiring the historical B=100 result bank.

suppressPackageStartupMessages({
    library(glue)
    library(data.table)
    library(cowplot)
    library(ggplot2)
    library(plot3D)
})

find_repro_root <- function() {
    file_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
    candidates <- c(
        getwd(),
        if (length(file_arg)) file.path(
            dirname(sub("^--file=", "", file_arg[[1]])), "..", ".."
        )
    )
    for (candidate in candidates) {
        candidate <- normalizePath(candidate, mustWork = FALSE)
        if (file.exists(file.path(candidate, "R", "core", "CLAW_functions.R"))) {
            return(candidate)
        }
    }
    stop("Cannot locate SEFT_repro.")
}

repro_root <- find_repro_root()
args <- commandArgs(TRUE)
result_root <- if (length(args)) normalizePath(args[[1]], mustWork = TRUE) else
    file.path(repro_root, "outputs", "intermediate", "simulation_results")
data_dir <- if (length(args) >= 2L) normalizePath(args[[2]], mustWork = TRUE) else
    file.path(repro_root, "outputs", "intermediate", "simulation_data")
figure_dir <- if (length(args) >= 3L) normalizePath(args[[3]], mustWork = FALSE) else
    file.path(repro_root, "outputs", "figures")
prefix <- if (length(args) >= 4L) args[[4]] else "3d_grf_nodenoise_b1"
dir.create(figure_dir, recursive = TRUE, showWarnings = FALSE)

mu_s <- seq(2, 5, by = 0.5)
num_points_s <- c(10L, 20L, 30L)
uprop_s <- c(0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5)
selected_pc <- c(1L, 3L, 5L, 7L)
alpha <- 0.1

all_msr <- all_fdr <- array(
    NA_real_, dim = c(length(mu_s), length(num_points_s), 5L, length(uprop_s))
)
for (i_mu in seq_along(mu_s)) for (i_points in seq_along(num_points_s)) {
    path <- file.path(
        result_root, prefix,
        glue("MSR_FDP_L64_mu{mu_s[[i_mu]]}_num_points{num_points_s[[i_points]]}_shapesphere.rds")
    )
    tables <- readRDS(path)
    all_msr[i_mu, i_points, , ] <- apply(tables$MSR, c(2L, 3L), mean)
    all_fdr[i_mu, i_points, , ] <- apply(tables$FSR, c(2L, 3L), mean)
}

method_names <- c("SEFT", "BHe+Simes", "BHe+Fisher")
method_colors <- c("SEFT" = "black", "BHe+Simes" = "green", "BHe+Fisher" = "purple")
method_shapes <- c("SEFT" = 15, "BHe+Simes" = 18, "BHe+Fisher" = 4)

make_plot <- function(values, metric) {
    panels <- list()
    for (i_pc in selected_pc) for (i_points in seq_along(num_points_s)) {
        frame <- rbindlist(lapply(seq_along(method_names), function(i_method) {
            data.table(
                mu = mu_s,
                value = values[, i_points, i_method + 2L, i_pc],
                method = factor(method_names[[i_method]], levels = method_names)
            )
        }))
        panel <- ggplot(
            frame,
            aes(x = mu, y = value, color = method, shape = method, group = method)
        ) +
            geom_line() + geom_point(size = 2) +
            scale_color_manual(values = method_colors, name = "Method") +
            scale_shape_manual(values = method_shapes, name = "Method") +
            labs(
                title = glue(
                    "u proportion={uprop_s[[i_pc]]}, {num_points_s[[i_points]]} clusters"
                ),
                x = "mu", y = metric
            ) +
            theme_bw() + theme(
                legend.position = "bottom",
                legend.text = element_text(size = 14),
                legend.title = element_text(size = 14)
            )
        if (metric == "FDR") {
            panel <- panel +
                geom_hline(yintercept = alpha, color = "red", linetype = "dashed") +
                scale_y_continuous(limits = c(0, 0.5))
        } else {
            panel <- panel + scale_y_continuous(limits = c(0, 1))
        }
        panels[[length(panels) + 1L]] <- panel
    }
    body <- plot_grid(
        plotlist = lapply(panels, function(panel) panel + theme(legend.position = "none")),
        nrow = 3L, ncol = 4L, align = "hv", byrow = FALSE
    )
    plot_grid(body, get_legend(panels[[1L]]), nrow = 2L, rel_heights = c(3, 0.3))
}

for (metric in c("FDR", "FNR")) {
    figure <- make_plot(if (metric == "FDR") all_fdr else all_msr, metric)
    pdf(
        file.path(figure_dir, glue("L64_shapesphere_{metric}_SEFTq0.pdf")),
        width = 12, height = 6
    )
    print(figure)
    dev.off()
}

# E.6: same original 3-D scatter layout, using the B=1, mu=4, 20-cluster run.
demo <- readRDS(file.path(
    result_root, prefix,
    "PC_results_L64_mu4_num_points20_shapesphere.rds"
))
coords <- expand.grid(x = 1:8, y = 1:8, z = 1:8)
pdf(file.path(figure_dir, "3d_demenstration_denoised.pdf"), height = 4.1, width = 9.2)
demo_pc_indices <- c(4L, 7L)
demo_panel_regions <- list(c(0.00, 0.58, 0, 1), c(0.42, 1.00, 0, 1))
for (panel_index in seq_along(demo_pc_indices)) {
    i_pc <- demo_pc_indices[[panel_index]]
    par(
        fig = demo_panel_regions[[panel_index]],
        mar = c(2.6, 1.1, 2.6, 0.1),
        new = panel_index > 1L
    )
    set_theta <- demo$all_set_theta[1L, i_pc, ]
    decisions <- demo$all_PC_decisions[1L, i_pc, , ]
    ids_truth <- which(set_theta == 1)
    ids_seft <- which(decisions[2L, ] == 1)
    ids_bhe <- which(decisions[4L, ] == 1)
    scatter3D(
        coords$x[ids_truth], coords$y[ids_truth], coords$z[ids_truth],
        cex = 1.2, theta = 60, phi = 10, colvar = NULL, col = "grey",
        bty = "b2", pch = 19, xlab = "X", ylab = "Y", zlab = "Z",
        cex.main = 1.5, cex.lab = 0.95, cex.axis = 1.1,
        main = glue("PC level c={uprop_s[[i_pc]]}, mu=4")
    )
    scatter3D(
        coords$x[ids_bhe], coords$y[ids_bhe], coords$z[ids_bhe],
        cex = 1.2, theta = 60, phi = 10, colvar = NULL,
        col = "green", bty = "b2", pch = 2, add = TRUE
    )
    scatter3D(
        coords$x[ids_seft], coords$y[ids_seft], coords$z[ids_seft],
        cex = 1.2, theta = 60, phi = 10, colvar = NULL,
        col = "red", bty = "b2", pch = 4, add = TRUE
    )
    legend(
        x = 0.28, y = -0.33, xpd = NA,
        c("BHe", "SEFT", "Ground truth"),
        pch = c(2, 4, 19), col = c("green", "red", "grey"),
        bty = "n", cex = 1.1
    )
}
dev.off()
