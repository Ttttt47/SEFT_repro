#!/usr/bin/env Rscript

suppressPackageStartupMessages({
    library(data.table)
    library(ggplot2)
    library(cowplot)
})
args <- commandArgs(TRUE)
file_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
repro_root <- normalizePath(file.path(dirname(sub("^--file=", "", file_arg[[1]])), "..", ".."), mustWork = TRUE)
root <- if (length(args)) normalizePath(args[[1]], mustWork = TRUE) else file.path(repro_root, "outputs", "sim_3d")
combined <- if (length(args) >= 2L) normalizePath(args[[2]], mustWork = FALSE) else file.path(root, "figures")
dir.create(combined, recursive = TRUE, showWarnings = FALSE)
inputs <- c("common", "fdrs_absmax", "deepfdr_absmax", "fchmrf_absmax")
missing <- inputs[!file.exists(file.path(root, inputs, "per_replication_metrics.csv"))]
if (length(missing)) stop("Missing method result directories: ", paste(missing, collapse = ", "))
metrics <- rbindlist(lapply(inputs, function(name) fread(file.path(root, name, "per_replication_metrics.csv"))), fill = TRUE)
methods <- c(
    "SEFT-CO",
    "SEFT-FDRSmoothing-AbsMaxAdapt",
    "SEFT-DeepFDR-AbsMaxAdapt",
    "SEFT-fcHMRF-AbsMaxAdapt",
    "BH+Simes",
    "BH+Fisher"
)
plot_methods <- methods
key <- c(
    "method", "noise", "mu", "num_points", "repetition", "pc_level"
)
if (anyDuplicated(metrics, by = key)) stop("Duplicate B100 rows.")
expected <- CJ(
    method = methods,
    noise = c("iid", "grf"),
    mu = c(2, 2.5, 3, 3.5, 4, 4.5, 5),
    num_points = c(10L, 20L, 30L),
    repetition = 1:100,
    pc_level = c(0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5),
    sorted = TRUE
)
if (!fsetequal(metrics[, ..key], expected)) {
    stop("B100 factorial-grid validation failed.")
}
summary <- metrics[, .(
    mean_fdp = mean(fdp),
    se_fdp = sd(fdp) / sqrt(.N),
    mean_fnr = mean(msr),
    se_fnr = sd(msr) / sqrt(.N),
    mean_discoveries = mean(discoveries),
    repetitions = uniqueN(repetition)
), by = .(method, noise, mu, num_points, pc_level)]
if (any(summary$repetitions != 100L)) stop("Not all cells have B=100.")
fwrite(
    metrics,
    file.path(combined, "per_replication_metrics_with_bh_baselines.csv")
)
fwrite(summary, file.path(combined, "summary_with_bh_baselines.csv"))

labels <- c(
    "SEFT-CO" = "SEFT-CO",
    "SEFT-FDRSmoothing-AbsMaxAdapt" = "SEFT-FDR smoothing",
    "SEFT-DeepFDR-AbsMaxAdapt" = "SEFT-DeepFDR",
    "SEFT-fcHMRF-AbsMaxAdapt" = "SEFT-fcHMRF",
    "BH+Simes" = "BHe+Simes",
    "BH+Fisher" = "BHe+Fisher"
)
colours <- c(
    "SEFT-CO" = "black",
    "SEFT-FDRSmoothing-AbsMaxAdapt" = "#0072B2",
    "SEFT-DeepFDR-AbsMaxAdapt" = "#CC79A7",
    "SEFT-fcHMRF-AbsMaxAdapt" = "#009E73",
    "BH+Simes" = "green",
    "BH+Fisher" = "purple"
)
shapes <- c(
    "SEFT-CO" = 15,
    "SEFT-FDRSmoothing-AbsMaxAdapt" = 16,
    "SEFT-DeepFDR-AbsMaxAdapt" = 17,
    "SEFT-fcHMRF-AbsMaxAdapt" = 18,
    "BH+Simes" = 18,
    "BH+Fisher" = 4
)
selected_pc <- c(0.01, 0.1, 0.3, 0.5)
source <- summary[
    pc_level %in% selected_pc & method %in% plot_methods
]
fwrite(source, file.path(
    combined, "repro_style_FDR_FNR_plot_source_with_bh_baselines.csv"
))

make_repro_plots <- function(noise_value, metric) {
    value <- if (metric == "FDR") "mean_fdp" else "mean_fnr"
    plot_list <- list()

    # Keep the loop order and column-wise cowplot arrangement identical to
    # SEFT_repro/simulations/sim_3d_grf_iid/make_figures_3d.R.
    for (pc in selected_pc) {
        for (clusters in c(10L, 20L, 30L)) {
            panel <- source[
                noise == noise_value & num_points == clusters &
                    abs(pc_level - pc) < 1e-12
            ]
            plot <- ggplot(
                panel,
                aes(
                    x = mu, y = .data[[value]], color = method,
                    shape = method, group = method
                )
            ) +
                geom_line() +
                geom_point(size = 2) +
                scale_color_manual(
                    values = colours, breaks = plot_methods,
                    labels = labels[plot_methods], drop = FALSE
                ) +
                scale_shape_manual(
                    values = shapes, breaks = plot_methods,
                    labels = labels[plot_methods], drop = FALSE
                ) +
                labs(
                    title = sprintf(
                        "u proportion=%s, %d clusters",
                        format(pc, trim = TRUE), clusters
                    ),
                    x = "mu", y = metric,
                    color = "Method", shape = "Method"
                ) +
                theme_bw() +
                theme(
                    legend.position = "bottom",
                    legend.direction = "horizontal",
                    legend.box = "horizontal",
                    legend.text = element_text(size = 14),
                    legend.title = element_text(size = 14),
                    legend.key.width = grid::unit(0.55, "cm"),
                    legend.spacing.x = grid::unit(0.04, "cm")
                ) +
                guides(
                    color = guide_legend(nrow = 1, byrow = TRUE),
                    shape = guide_legend(nrow = 1, byrow = TRUE)
                )
            if (metric == "FDR") {
                plot <- plot +
                    geom_hline(
                        aes(yintercept = 0.1),
                        colour = "red", linetype = "dashed"
                    ) +
                    scale_y_continuous(limits = c(0, 0.5))
            } else {
                plot <- plot + scale_y_continuous(limits = c(0, 1))
            }
            plot_list[[length(plot_list) + 1L]] <- plot
        }
    }

    combined_plot <- plot_grid(
        plotlist = lapply(
            plot_list, function(plot) plot + theme(legend.position = "none")
        ),
        nrow = 3L, ncol = 4L, align = "hv", byrow = FALSE
    )
    plot_grid(
        combined_plot, get_legend(plot_list[[1]]),
        nrow = 2L, rel_heights = c(3, 0.3)
    )
}

paper_figure_dir <- combined
dir.create(paper_figure_dir, recursive = TRUE, showWarnings = FALSE)
for (noise_value in c("iid", "grf")) {
    noise_tag <- if (noise_value == "iid") "iid" else "denoisedGRF"
    for (metric in c("FDR", "FNR")) {
        figure <- make_repro_plots(noise_value, metric)
        name <- sprintf("L64_six_methods_%s_%s", metric, noise_tag)
        pdf(
            file.path(paper_figure_dir, paste0(name, ".pdf")),
            width = 12, height = 6
        )
        print(figure)
        dev.off()
        ggsave(
            file.path(paper_figure_dir, paste0(name, ".png")),
            figure, width = 12, height = 6, dpi = 240, bg = "white"
        )
    }
}
writeLines(
    "COMPLETE",
    file.path(combined, "SIMULATION_PLOTS_COMPLETE")
)
cat("Merged method results and wrote manuscript simulation figures.\n")
