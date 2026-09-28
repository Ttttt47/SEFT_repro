library(glue)
library(data.table)
library(cowplot)
library(ggplot2)
library(ComplexHeatmap)
library(circlize)

if (!file.exists(file.path("R", "core", "CLAW_functions.R"))) {
    stop("Run this script from the SEFT project root.")
}
source(file.path("R", "core", "CLAW_functions.R"))

mu_s <- seq(1.5, 3.5, 0.2)
L <- 500
shape_s <- c("disk", "iid")
prefix <- Sys.getenv("SEFT_SIM_PREFIX", "2d_main")
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
sim_result_dir <- file.path(Sys.getenv(
    "SEFT_SIM_RESULT_ROOT",
    file.path("outputs", "intermediate", "simulation_results")
), prefix)
figure_dir <- Sys.getenv("SEFT_SIM_FIGURE_DIR", file.path("outputs", "figures"))
dir.create(figure_dir, recursive = TRUE, showWarnings = FALSE)

all_MSR_table <- array(-1, dim = c(length(shape_s), length(mu_s), length(para_sets), length(n_div_s), length(uprop_s)))
all_FSR_table <- array(-1, dim = c(length(shape_s), length(mu_s), length(para_sets), length(n_div_s), length(uprop_s)))

for (i_shape in seq_along(shape_s)) {
    for (i_mu in seq_along(mu_s)) {
        shape <- shape_s[i_shape]
        mu <- mu_s[i_mu]
        tables <- readRDS(file.path(sim_result_dir, glue("MSR_FDP_L{L}_mu{mu}_shape{shape}.rds")))
        all_MSR_table[i_shape, i_mu, , , ] <- apply(tables[[1]], MARGIN = c(2, 3, 4), FUN = mean)
        all_FSR_table[i_shape, i_mu, , , ] <- apply(tables[[2]], MARGIN = c(2, 3, 4), FUN = mean)
    }
}

method_names <- c("SEFT", "BHe+Simes", "BHe+Fisher")
method_colors <- c("SEFT" = "black", "BHe+Simes" = "green", "BHe+Fisher" = "purple")
method_shapes <- c("SEFT" = 15, "BHe+Simes" = 18, "BHe+Fisher" = 4)

for (i_shape in seq_along(shape_s)) {
    shape <- shape_s[i_shape]
    MSR_plot_list <- list()
    FSR_plot_list <- list()

    for (i_n_div in seq_along(n_div_s)) {
        for (i_uprop in c(1, 2, 3, 5)) {
            n_div <- n_div_s[i_n_div]
            uprop <- uprop_s[i_uprop]

            msr_dt <- rbindlist(lapply(seq_along(method_names), function(i) {
                data.table(
                    mu = mu_s,
                    MSR = all_MSR_table[i_shape, , i + 2, i_n_div, i_uprop],
                    method = method_names[i]
                )
            }))

            fsr_dt <- rbindlist(lapply(seq_along(method_names), function(i) {
                data.table(
                    mu = mu_s,
                    FSR = all_FSR_table[i_shape, , i + 2, i_n_div, i_uprop],
                    method = method_names[i]
                )
            }))

            plot_key <- glue("L{L}_shape{shape}_n_div{n_div}_uprop{uprop}")
            MSR_plot_list[[plot_key]] <- ggplot(msr_dt, aes(x = mu, y = MSR, color = method, shape = method, group = method)) +
                geom_line() +
                geom_point(size = 2) +
                scale_color_manual(values = method_colors) +
                scale_shape_manual(values = method_shapes) +
                labs(title = glue("u proportion={uprop}, n^2={(L / n_div)^2}"), x = "mu", y = "FNR") +
                scale_y_continuous(limits = c(0, 1)) +
                theme_bw() +
                theme(
                    legend.position = "bottom",
                    legend.text = element_text(size = 14),
                    legend.title = element_text(size = 14)
                )

            FSR_plot_list[[plot_key]] <- ggplot(fsr_dt, aes(x = mu, y = FSR, color = method, shape = method, group = method)) +
                geom_line() +
                geom_point(size = 2) +
                scale_color_manual(values = method_colors) +
                scale_shape_manual(values = method_shapes) +
                labs(title = glue("u proportion={uprop}, n^2={(L / n_div)^2}"), x = "mu", y = "FDR") +
                scale_y_continuous(limits = c(0, 1)) +
                theme_bw() +
                theme(
                    legend.position = "bottom",
                    legend.text = element_text(size = 14),
                    legend.title = element_text(size = 14)
                )
        }
    }

    suffix <- ifelse(shape == "iid", "iid", "disk")
    fnr_file <- file.path(figure_dir, glue("L{L}_shape{suffix}_FNR_SEFTq0.pdf"))
    fdr_file <- file.path(figure_dir, glue("L{L}_shape{suffix}_FDR_SEFTq0.pdf"))

    pdf(fnr_file, width = 12, height = 4)
    combined_plot <- plot_grid(
        plotlist = lapply(MSR_plot_list, function(p) p + theme(legend.position = "none")),
        ncol = length(c(1, 2, 3, 5)),
        nrow = length(n_div_s),
        align = "hv",
        byrow = TRUE
    )
    final_plot <- plot_grid(combined_plot, get_legend(MSR_plot_list[[1]]), nrow = 2, rel_heights = c(3, 0.3))
    print(final_plot)
    dev.off()

    pdf(fdr_file, width = 12, height = 4)
    combined_plot <- plot_grid(
        plotlist = lapply(FSR_plot_list, function(p) p + theme(legend.position = "none")),
        ncol = length(c(1, 2, 3, 5)),
        nrow = length(n_div_s),
        align = "hv",
        byrow = TRUE
    )
    final_plot <- plot_grid(combined_plot, get_legend(FSR_plot_list[[1]]), nrow = 2, rel_heights = c(3, 0.3))
    print(final_plot)
    dev.off()
}

# Region-finding demonstration (disk only)
shape <- "disk"
b <- 1
i_n_div <- 2
plot_list <- list()
color_func <- colorRamp2(c(uprop_s)[c(1, 3, 5, 7)], c("#FFC100", "#FF8A08", "#FF6500", "#C40C0C"))
colors <- structure(color_func(uprop_s), names = as.character(uprop_s))

for (i_mu in c(1, 3, 5)) {
    mu <- mu_s[i_mu]
    PC_results <- readRDS(file.path(sim_result_dir, glue("PC_results_L{L}_mu{mu}_shape{shape}.rds")))
    simulated_data <- readRDS(file.path(sim_data_dir, glue("L{L}_mu{mu}_shape{shape}.rds")))[[glue("L{L}_mu{mu}_shape{shape}_b{b}")]]

    p1 <- Heatmap(
        simulated_data$Xs,
        width = unit(8, "cm"),
        height = unit(8, "cm"),
        cluster_rows = FALSE,
        cluster_columns = FALSE,
        show_row_names = FALSE,
        show_column_names = FALSE,
        show_row_dend = FALSE,
        show_column_dend = FALSE,
        column_title = glue("Observed data, mu={mu}"),
        name = " ",
        border_gp = gpar(col = "black")
    )
    plot_list[[length(plot_list) + 1]] <- grid.grabExpr(draw(p1))

    all_PC_decisions <- array(NA, dim = c(length(uprop_s), length(para_sets), n_div_s[i_n_div]^2))
    for (i_uprop in seq_along(uprop_s)) {
        PC_p_values <- PC_results$all_PC_p_values[[glue("{b}_{i_n_div}_{i_uprop}")]]
        PC_decisions <- array(NA, dim = c(length(para_sets), n_div_s[i_n_div]^2))
        for (k in seq_along(para_sets)) {
            PC_decisions[k, ] <- BH(PC_p_values[k, ], alpha)
        }
        all_PC_decisions[i_uprop, , ] <- PC_decisions
    }

    max_PC_decisions <- apply(all_PC_decisions, MARGIN = c(2, 3), FUN = sum)
    max_PC_decisions[max_PC_decisions == 0] <- NA
    for (i_uprop in seq_along(uprop_s)) {
        max_PC_decisions[max_PC_decisions == i_uprop] <- uprop_s[i_uprop]
    }

    # Manuscript order: first = observed Z-statistics, second = BH, third = SEFT
    p2 <- Heatmap(
        matrix(max_PC_decisions[4, ], n_div_s[i_n_div]),
        width = unit(8, "cm"),
        height = unit(8, "cm"),
        cluster_rows = FALSE,
        cluster_columns = FALSE,
        show_row_names = FALSE,
        show_column_names = FALSE,
        show_row_dend = FALSE,
        show_column_dend = FALSE,
        column_title = "Regions found by BHe",
        name = " ",
        border_gp = gpar(col = "black"),
        show_heatmap_legend = FALSE,
        rect_gp = gpar(col = "grey", lwd = 2),
        col = colors,
        na_col = "white"
    )

    p3 <- Heatmap(
        matrix(max_PC_decisions[2, ], n_div_s[i_n_div]),
        width = unit(8, "cm"),
        height = unit(8, "cm"),
        cluster_rows = FALSE,
        cluster_columns = FALSE,
        show_row_names = FALSE,
        show_column_names = FALSE,
        show_row_dend = FALSE,
        show_column_dend = FALSE,
        column_title = "Regions found by SEFT",
        name = " ",
        heatmap_legend_param = list(title = "Proportion of signals", title_position = "leftcenter-rot"),
        border_gp = gpar(col = "black"),
        rect_gp = gpar(col = "grey", lwd = 2),
        col = colors,
        na_col = "white"
    )

    plot_list[[length(plot_list) + 1]] <- grid.grabExpr(draw(p2))  # second column: BH
    plot_list[[length(plot_list) + 1]] <- grid.grabExpr(draw(p3))  # third column: SEFT
}

png(file.path(figure_dir, glue("L{L}_shapedisk_region_found.png")), width = 30, height = 28, units = "cm", res = 300)
plot_grid(plotlist = plot_list, ncol = 3, nrow = 3, byrow = TRUE, rel_widths = c(1, 1, 1.25), rel_heights = c(1, 1, 1))
dev.off()
