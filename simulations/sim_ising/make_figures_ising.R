#!/usr/bin/env Rscript

# D.13--D.14 plotting entry. The panel layout, aesthetics, dimensions and
# output names match the manuscript figures.

suppressPackageStartupMessages({
  library(glue)
  library(data.table)
  library(cowplot)
  library(ggplot2)
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
    if (file.exists(file.path(candidate, "R", "core", "Ising_functions.R"))) return(candidate)
  }
  stop("Cannot locate SEFT_repro.")
}

args <- commandArgs(TRUE)
repro_root <- find_repro_root()
run_root <- if (length(args)) normalizePath(args[[1]], mustWork = TRUE) else file.path(repro_root, "outputs", "sim_ising")
input_root <- file.path(run_root, "legacy")
input_is <- file.path(input_root, "seft_is")
input_co <- file.path(input_root, "seft_co")
figure_dir <- if (length(args) >= 2L) normalizePath(args[[2]], mustWork = FALSE) else file.path(run_root, "figures")
dir.create(figure_dir, recursive = TRUE, showWarnings = FALSE)

mu_s <- seq(1.5, 4, by = 0.5)
h_s <- c(-2.5, -2.4, -2.3)
pc_levels <- c(0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5)
pc_indices <- 1:4
L <- 64L
beta <- 0.8
all_msr <- all_fdr <- array(
  NA_real_, dim = c(2L, length(mu_s), length(h_s), 5L, length(pc_levels))
)

for (i_mu in seq_along(mu_s)) for (i_h in seq_along(h_s)) {
  file_name <- sprintf(
    "MSR_FDP_L%d_beta%.2f_h%.2f_mu%.2f.rds",
    L, beta, h_s[[i_h]], mu_s[[i_mu]]
  )
  is_table <- readRDS(file.path(input_is, file_name))
  co_table <- readRDS(file.path(input_co, file_name))
  all_msr[1L, i_mu, i_h, , ] <- apply(is_table$MSR, c(2L, 3L), mean)
  all_fdr[1L, i_mu, i_h, , ] <- apply(is_table$FSR, c(2L, 3L), mean)
  all_msr[2L, i_mu, i_h, , ] <- apply(co_table$MSR, c(2L, 3L), mean)
  all_fdr[2L, i_mu, i_h, , ] <- apply(co_table$FSR, c(2L, 3L), mean)
}

method_names <- c("BHe+Fisher", "BHe+Simes", "SEFT-CO", "SEFT-Ising")
method_colors <- c(
  "BHe+Fisher" = "purple", "BHe+Simes" = "green",
  "SEFT-CO" = "orange", "SEFT-Ising" = "black"
)
method_shapes <- c(
  "BHe+Fisher" = 4, "BHe+Simes" = 18,
  "SEFT-CO" = 5, "SEFT-Ising" = 15
)

extract_values <- function(metric_array, i_h, i_pc) {
  rbindlist(lapply(method_names, function(method) {
    values <- switch(
      method,
      "BHe+Fisher" = metric_array[1L, , i_h, 5L, i_pc],
      "BHe+Simes" = metric_array[1L, , i_h, 4L, i_pc],
      "SEFT-CO" = metric_array[2L, , i_h, 3L, i_pc],
      "SEFT-Ising" = metric_array[1L, , i_h, 3L, i_pc]
    )
    data.table(
      mu = mu_s,
      value = values,
      method = factor(method, levels = method_names)
    )
  }))
}

make_panel <- function(metric_array, i_h, i_pc, y_label, y_limits, add_alpha) {
  plot <- ggplot(
    extract_values(metric_array, i_h, i_pc),
    aes(x = mu, y = value, color = method, shape = method, group = method)
  ) +
    geom_line() +
    geom_point(size = 2) +
    scale_color_manual(values = method_colors, name = "Method") +
    scale_shape_manual(values = method_shapes, name = "Method") +
    labs(
      title = glue(
        "h0={format(h_s[[i_h]], trim = TRUE)}, PC level c={format(pc_levels[[i_pc]], trim = TRUE)}"
      ),
      x = "mu", y = y_label
    ) +
    scale_y_continuous(limits = y_limits) +
    theme_bw() +
    theme(
      legend.position = "bottom",
      legend.text = element_text(size = 14),
      legend.title = element_text(size = 14)
    )
  if (add_alpha) {
    plot <- plot + geom_hline(yintercept = 0.10, colour = "red", linetype = "dashed")
  }
  plot
}

write_figure <- function(metric_array, y_label, y_limits, add_alpha, file_name) {
  panels <- list()
  for (i_h in seq_along(h_s)) for (i_pc in pc_indices) {
    panels[[length(panels) + 1L]] <- make_panel(
      metric_array, i_h, i_pc, y_label, y_limits, add_alpha
    )
  }
  combined <- plot_grid(
    plotlist = lapply(panels, function(plot) plot + theme(legend.position = "none")),
    nrow = 3L, ncol = 4L, align = "hv", byrow = TRUE
  )
  final <- plot_grid(
    combined, get_legend(panels[[1L]]),
    nrow = 2L, rel_heights = c(3, 0.3)
  )
  pdf(file.path(figure_dir, file_name), width = 12, height = 6)
  print(final)
  dev.off()
}

write_figure(
  all_msr, "FNR", c(0, 1), FALSE,
  "L64_FNR_SEFT_Ising_PLIS_vs_CO.pdf"
)
write_figure(
  all_fdr, "FDR", c(0, 0.5), TRUE,
  "L64_FDR_SEFT_Ising_PLIS_vs_CO.pdf"
)
