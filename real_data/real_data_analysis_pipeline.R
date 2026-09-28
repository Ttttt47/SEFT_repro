# ==============================================================================
# CLAW-based Real Data Analysis Pipeline
# ==============================================================================
# This script provides a modular pipeline for analyzing real fMRI data
# using CLAW scores and Partial Conjunction (PC) testing.
#
# Usage:
#   config <- create_config()
#   results <- run_full_pipeline(config)
# ==============================================================================

# Load required libraries
library(glue)
library(MASS)
library(oro.nifti)
library(Rcpp)
library(RcppArmadillo)
library(data.table)
library(ggplot2)
library(ComplexHeatmap)
library(circlize)
library(cowplot)

# Source CLAW functions
if (!file.exists(file.path("R", "core", "CLAW_functions.R"))) {
    stop("Run this script from the SEFT project root.")
}
source(file.path("R", "core", "CLAW_functions.R"))

# ==============================================================================
# Configuration
# ==============================================================================

create_config <- function(
    # Data paths
    data_path = 'real_data/input/reward_MDD/zstat1.nii.gz',
    atlas_path = 'real_data/input/atlas/AAL3v1.nii.gz',
    atlas_label_path = 'real_data/input/atlas/AAL3v1.nii.txt',
    output_dir = 'outputs/intermediate/real_data/analysis',
    output_prefix = 'MDD_denoised_zstat1',
    data_format = 'nifti',
    
    # CLAW parameters
    bandwidth = 5,
    lambda = 0.5,
    neighbor_range = 10,
    
    # PC testing parameters
    uprop_s = c(0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5),
    alpha = 0.1,
    para_sets = list(
        list('Sym_miny', 'oneminus', T),
        list('Sym_miny', 'neglog', F),
        list('Avg', 'oneminus', F),
        list('Sym_fdr', 'oneminus', F),
        list('Sym_fdr', 'neglog', F),
        list('Simes'),
        list('Fisher')
    ),
    
    # Preprocessing options
    use_wavelet_denoising = TRUE,
    wavelet_params = list(
        wf = "d8",
        J = 7,
        expand_to = c(128, 128, 128),
        threshold_type = "hard"
    ),
    
    # Visualization parameters
    create_visualizations = TRUE,
    fig_dir = file.path("outputs", "figures"),
    slice_coords = list(
        list(type = 'z', coord = -6),
        list(type = 'y', coord = -10),
        list(type = 'x', coord = 0)
    ),
    
    # Analysis options
    save_results = TRUE,
    verbose = TRUE
) {
    list(
        data_path = data_path,
        atlas_path = atlas_path,
        atlas_label_path = atlas_label_path,
        output_dir = output_dir,
        output_prefix = output_prefix,
        data_format = data_format,
        bandwidth = bandwidth,
        lambda = lambda,
        neighbor_range = neighbor_range,
        uprop_s = uprop_s,
        alpha = alpha,
        para_sets = para_sets,
        use_wavelet_denoising = use_wavelet_denoising,
        wavelet_params = wavelet_params,
        create_visualizations = create_visualizations,
        fig_dir = fig_dir,
        slice_coords = slice_coords,
        save_results = save_results,
        verbose = verbose
    )
}

# ==============================================================================
# Data Loading and Preprocessing
# ==============================================================================

load_data <- function(config) {
    if (config$verbose) cat("Loading data and atlas...\n")
    
    # Load fMRI data
    fmri_data <- readNIfTI(config$data_path)
    t_data <- fmri_data@.Data[,,]
    data_mask <- t_data != 0
    
    # Load atlas
    atlas <- readNIfTI(config$atlas_path)
    
    # Apply mask to atlas
    region_masks <- atlas@.Data[,,] * data_mask
    region_ids <- unique(c(region_masks))
    region_ids <- region_ids[region_ids != 0]

    # Build transform matrix (NIfTI)
    transformM <- matrix(c(fmri_data@srow_x, fmri_data@srow_y, 
                           fmri_data@srow_z, c(0,0,0,1)), nrow=4, byrow=T)

    # Optional: region label map from atlas label file
    region_labels_map <- NULL
    if (!is.null(config$atlas_label_path) && file.exists(config$atlas_label_path)) {
        labels_df <- tryCatch({ fread(config$atlas_label_path)[, 1:2] }, error = function(e) NULL)
        if (!is.null(labels_df)) {
            colnames(labels_df) <- c("V1", "V2")
            region_labels_map <- data.frame(
                region_id = seq_len(nrow(labels_df)),
                label = as.character(labels_df$V2),
                stringsAsFactors = FALSE
            )
        }
    }

    list(
        t_data = t_data,
        data_mask = data_mask,
        atlas = atlas,
        region_masks = region_masks,
        region_ids = region_ids,
        transformM = transformM,
        region_labels_map = region_labels_map
    )
}

load_data_cifti <- function(config) {
if (config$verbose) cat("Loading CIFTI data...\n")
    
    if (!requireNamespace("ciftiTools", quietly = TRUE)) {
        stop("`ciftiTools` package is required. Please install it.")
    }
    
    # 1. Load CIFTI scalar data (e.g., F-stats)
    cifti_data <- ciftiTools::read_cifti(config$data_path)
    
    # --- Extract Subcortical Data Only ---
    
    # 2. Get 3D dimensions from the subcortical MASK
    subcort_dims <- dim(cifti_data$meta$subcort$mask)
    if (is.null(subcort_dims)) {
        stop("Could not find subcortical dimensions in cifti_data$meta$subcort$mask.")
    }
    
    # 3. Get the 1D data vector
    subcort_data_vec <- cifti_data$data$subcort
    # transform F-stats to Z-scores
    subcort_data_vec <- qnorm(1 - subcort_data_vec)

    # 4. Get the 1D label vector (factor) from the metadata
    subcort_atlas_factor <- cifti_data$meta$subcort$labels
    
    # 5. Get 1D indices (the mapping from 1D vector to 3D array)
    subcort_indices <- which(cifti_data$meta$subcort$mask) 
    
    # 6. Sanity Checks
    if (is.null(subcort_data_vec)) stop("cifti_data$data$subcort is NULL. Data not found.")
    if (is.null(subcort_atlas_factor)) stop("cifti_data$meta$subcort$labels is NULL. Atlas labels not found.")
    
    if (length(subcort_data_vec) != length(subcort_indices)) {
        stop(glue::glue("Data vector length ({length(subcort_data_vec)}) does not match number of voxels in mask ({length(subcort_indices)})."))
    }
    if (length(subcort_atlas_factor) != length(subcort_indices)) {
        stop(glue::glue("Atlas label vector length ({length(subcort_atlas_factor)}) does not match number of voxels in mask ({length(subcort_indices)})."))
    }

    # --- Reconstruct 3D NIfTI-like Arrays ---
    
    # 7. Create empty 3D arrays
    t_data <- array(0, dim = subcort_dims)
    region_masks <- array(0, dim = subcort_dims)
    
    # 8. Populate the arrays using the 1D indices
    t_data[subcort_indices] <- subcort_data_vec
    # Map subcortical labels (strings) to integer region IDs and save the mapping

    # Get the unique string labels as they appear in the atlas (excluding background/blank, which is usually NA or blank)
    label_strings <- as.character(subcort_atlas_factor)
    # Exclude empty (background) labels
    valid_labels <- sort(unique(label_strings[label_strings != "" & !is.na(label_strings)]))
    # Create region_id <-> string label map (region_id: 1, 2, ..., N)
    region_labels_map <- data.frame(
        region_id = seq_along(valid_labels),
        label = valid_labels,
        stringsAsFactors = FALSE
    )
    # Create a string label -> integer region_id vector for fast mapping
    label_to_id <- setNames(region_labels_map$region_id, region_labels_map$label)
    id_to_label <- setNames(region_labels_map$label, region_labels_map$region_id)
    # Assign mapped region ID for each voxel (background remains 0)
    region_masks[subcort_indices] <- ifelse(
        label_strings != "" & !is.na(label_strings),
        label_to_id[label_strings],
        0
    )
    # 9. Create the final mask and get region IDs
    data_mask <- (region_masks != 0)
    region_ids <- unique(c(region_masks))
    region_ids <- region_ids[region_ids != 0] # Remove 0 (background)
    
    if (config$verbose) {
        cat(glue::glue("Successfully extracted subcortex with dimensions: {paste(subcort_dims, collapse=' x ')}\n"))
        cat(glue::glue("Found {length(subcort_data_vec)} voxels across {length(region_ids)} regions.\n"))
    }
    
    # 11. Return a list with the same names as the original load_data function
    return(
        list(
        t_data = t_data,
        data_mask = data_mask,
        region_masks = region_masks,
        region_ids = region_ids,
        region_labels_map = region_labels_map, # NEW: Include this for results analysis
        transformM = cifti_data$meta$subcort$trans_mat
    ))
    
}

apply_wavelet_denoising <- function(t_data, data_mask, config) {
    if (!config$use_wavelet_denoising) {
        return(t_data)
    }
    
    # Use the unified wavelet denoising function from CLAW_functions.R
    expand_dims <- config$wavelet_params$expand_to
    wf <- config$wavelet_params$wf
    J <- config$wavelet_params$J
    
    return(apply_wavelet_denoising_3d(
        z_map = t_data,
        expand_to = expand_dims,
        data_mask = data_mask,
        wf = wf,
        J = J,
        verbose = config$verbose
    ))
}

# ==============================================================================
# CLAW Score Calculation
# ==============================================================================

calculate_CLAW_scores <- function(data, config) {
    if (config$verbose) cat("Calculating CLAW scores...\n")
    
    t_data <- data$t_data
    data_mask <- data$data_mask
    
    # Generate null data
    t_data_til <- array(rnorm(prod(dim(t_data))), dim = dim(t_data)) * data_mask
    
    # Estimate bandwidth
    h <- density(sample(c(t_data_til, t_data), 10^3), bw = 'nrd0')$bw
    
    # Calculate CLAW scores
    results <- cal_CLAW_scores_3d(
        t_data, t_data_til,
        2 * (1 - pnorm(abs(t_data))), 
        2 * (1 - pnorm(abs(t_data_til))),
        h = h, 
        bandwidth = config$bandwidth, 
        lambda = config$lambda, 
        neighbor_range = config$neighbor_range
    )
    
    # Save results if requested
    if (config$save_results) {
        output_file <- file.path(config$output_dir, 
                                paste0(config$output_prefix, '_bw', config$bandwidth, 
                                      '_nr', config$neighbor_range, '_CLAW_results.rds'))
        dir.create(config$output_dir, showWarnings = FALSE, recursive = TRUE)
        saveRDS(results, output_file)
        if (config$verbose) cat("CLAW results saved to:", output_file, "\n")
    }
    
    return(results)
}

# ==============================================================================
# Partial Conjunction Testing
# ==============================================================================

run_PC_testing <- function(data, results, config) {
    if (config$verbose) cat("Running PC testing...\n")
    
    t_data <- data$t_data
    data_mask <- data$data_mask
    region_masks <- data$region_masks
    region_ids <- data$region_ids
    
    # Initialize arrays
    all_PC_e_values <- array(NA, dim = c(length(config$uprop_s), 
                                        length(config$para_sets), 
                                        length(region_ids)))
    all_PC_p_values <- array(NA, dim = c(length(config$uprop_s), 
                                        length(config$para_sets), 
                                        length(region_ids)))
    all_PC_decisions <- array(NA, dim = c(length(config$uprop_s), 
                                         length(config$para_sets), 
                                         length(region_ids)))
    nk_s <- array(NA, dim = length(region_ids))
    
    # Process each uprop
    for (i_uprop in seq_along(config$uprop_s)) {
        if (config$verbose) {
            cat(glue('{i_uprop}/{length(config$uprop_s)} u testing start.'),'\n')
        }
        
        PC_e_values <- array(NA, dim = c(length(config$para_sets), length(region_ids)))
        PC_p_values <- array(NA, dim = c(length(config$para_sets), length(region_ids)))
        PC_decisions <- array(NA, dim = c(length(config$para_sets), length(region_ids)))
        
        for (i_region in seq_along(region_ids)) {
            if (i_region %% 50 == 0 && config$verbose) {
                cat(glue('{i_region}/{length(region_ids)} region testing start.'),'\n')
            }
            
            region_id <- region_ids[i_region]
            u <- ceiling(sum(region_masks == region_id) * config$uprop_s[i_uprop])
            nk_s[i_region] <- sum(region_masks == region_id)
            
            for (k in seq_along(config$para_sets)) {
                if (config$para_sets[[k]][[1]] %in% c('Simes', 'Fisher')) {
                    pvals <- 2 * (1 - pnorm(abs(t_data[region_masks == region_id])))
                    if (config$para_sets[[k]][[1]] == 'Simes') {
                        PC_p_values[k, i_region] <- Simes_PC_test_cpp(pvals, u)
                    } else if (config$para_sets[[k]][[1]] == 'Fisher') {
                        PC_p_values[k, i_region] <- Fisher_PC_test_cpp(pvals, u)
                    }
                    next
                }
                
                mth <- config$para_sets[[k]][[1]]
                til_mth <- config$para_sets[[k]][[2]]
                signmin_flag <- config$para_sets[[k]][[3]]
                
                log_PLIS_scores <- matrix(c(
                    results$R[region_masks == region_id],
                    results$R_til[region_masks == region_id]
                ), nrow = 2, byrow = TRUE)
                
                log_PLIS_scores <- log_PLIS_scores / max(log_PLIS_scores)
                log_PLIS_scores <- log(log_PLIS_scores)
                
                PC_e_values[k, i_region] <- cal_ELIS_cpp(
                    log_PLIS_scores, u, 
                    method = mth, 
                    til_mth = til_mth, 
                    signmin_flag = signmin_flag
                )
                PC_p_values[k, i_region] <- 1 / PC_e_values[k, i_region]
            }
        }
        
        if (config$verbose) {
            cat("Mean e-values:", rowMeans(PC_e_values, na.rm = TRUE), "\n")
        }
        
        for (k in seq_along(config$para_sets)) {
            PC_decisions[k,] <- BH(PC_p_values[k,], config$alpha)
        }
        
        all_PC_e_values[i_uprop, , ] <- PC_e_values
        all_PC_p_values[i_uprop, , ] <- PC_p_values
        all_PC_decisions[i_uprop, , ] <- PC_decisions
    }
    
    PC_results <- list(
        all_PC_e_values = all_PC_e_values,
        all_PC_p_values = all_PC_p_values,
        all_PC_decisions = all_PC_decisions,
        nk_s = nk_s,
        region_ids = region_ids,
        para_sets = config$para_sets,
        uprop_s = config$uprop_s,
        region_labels_map = if (!is.null(data$region_labels_map)) data$region_labels_map else NULL
    )
    
    # Save results if requested
    if (config$save_results) {
        output_file <- file.path(config$output_dir, 
                                paste0(config$output_prefix, '_bw', config$bandwidth, 
                                      '_nr', config$neighbor_range, '_CLAW_PC_results.rds'))
        saveRDS(PC_results, output_file)
        if (config$verbose) cat("PC results saved to:", output_file, "\n")
    }
    
    return(PC_results)
}

# ==============================================================================
# Visualization Functions
# ==============================================================================

create_region_visualization <- function(data, PC_results, config) {
    if (!config$create_visualizations) return(NULL)
    
    if (config$verbose) cat("Creating visualizations...\n")
    
    t_data <- data$t_data
    region_masks <- data$region_masks
    region_ids <- data$region_ids
    transformM <- data$transformM

    # Color functions
    color_func <- colorRamp2(c(0, config$uprop_s[c(1,3,5,7)]), 
                            c("grey", "#FFC100", "#FF8A08", "#FF6500", "#C40C0C"))
    colors <- structure(color_func(c(0, config$uprop_s)), 
                       names = as.character(c(0, config$uprop_s)))
    color_func2 <- colorRamp2(c(-4, 0, 4), c("blue", "white", "red"))
    
    plot_list <- list()

    save_plot_grid_png <- function(filename, plot_obj, width, height, units = "cm", res = 300) {
        if (capabilities("cairo")) {
            png(filename, width = width, height = height, units = units, res = res, type = "cairo")
        } else {
            png(filename, width = width, height = height, units = units, res = res)
        }
        on.exit(dev.off(), add = TRUE)
        print(plot_obj)
    }

    # Create plots for each slice
    for (i in seq_along(config$slice_coords)) {
        slice <- config$slice_coords[[i]]
        
        dims <- dim(t_data)
        if (slice$type == 'z') {
            labels <- glue('z={slice$coord}')
            z_coord <- (solve(transformM) %*% c(1, 1, slice$coord, 1))[3]
            ids <- matrix(c(1, dim(t_data)[1],
                          1, dim(t_data)[2],
                          z_coord, z_coord),
                         nrow=3, byrow=T)
        } else if (slice$type == 'y') {
            labels <- glue('y={slice$coord}')
            y_coord <- (solve(transformM) %*% c(1, slice$coord, 1, 1))[2]
            ids <- matrix(c(1, dim(t_data)[1],
                          y_coord, y_coord,
                          1, dim(t_data)[3]),
                         nrow=3, byrow=T)
        } else if (slice$type == 'x') {
            labels <- glue('x={slice$coord}')
            x_coord <- (solve(transformM) %*% c(slice$coord, 1, 1, 1))[1]
            ids <- matrix(c(x_coord, x_coord,
                          1, dim(t_data)[2],
                          1, dim(t_data)[3]),
                         nrow=3, byrow=T)
        }
        
        mask <- region_masks[ids[1,1]:ids[1,2], ids[2,1]:ids[2,2], ids[3,1]:ids[3,2]]
        data_mat <- t_data[ids[1,1]:ids[1,2], ids[2,1]:ids[2,2], ids[3,1]:ids[3,2]]
        data_mat[mask == 0] <- NA

        # Z statistics plot
        p1 <- Heatmap(t(data_mat[, dim(data_mat)[2]:1]), 
                     width = unit(8, "cm"), height = unit(8, "cm"), 
                     cluster_rows = FALSE, cluster_columns = FALSE, 
                     show_row_names = FALSE, show_column_names = FALSE,
                     show_row_dend = FALSE, show_column_dend = FALSE, 
                     column_title = glue('Z statistics, {labels}'), name=' ',
                     border_gp = gpar(col = "black"), show_heatmap_legend = T, 
                     na_col = "#EEEEEE", col = color_func2)
        plot_list[[length(plot_list) + 1]] <- grid::grid.grabExpr(ComplexHeatmap::draw(p1))
        # Regions found by different methods, in manuscript order:
        # second column = BH, third column = SEFT
        for (method_idx in c(6, 2)) {  # BH/Simes (6), then SEFT (2)
            method_name <- if (method_idx == 6) "BH" else "SEFT"
            max_PC_decisions <- apply(PC_results$all_PC_decisions, 
                                    MARGIN = c(2,3), FUN = sum)[method_idx,]
            
            sig_regions <- region_masks
            for (j in seq_along(region_ids)) {
                sig_regions[sig_regions == region_ids[j]] <- max_PC_decisions[j]
            }
            
            mat <- sig_regions[ids[1,1]:ids[1,2], ids[2,1]:ids[2,2], ids[3,1]:ids[3,2]]
            mat[mask == 0] <- NA
            for (i_uprop in seq_along(config$uprop_s)) {
                mat[which(mat == i_uprop)] <- config$uprop_s[i_uprop]
            }
            p2 <- Heatmap(t(mat[, dim(mat)[2]:1]), 
                         width = unit(8, "cm"), height = unit(8, "cm"), 
                         cluster_rows = FALSE, cluster_columns = FALSE, 
                         show_row_names = FALSE, show_column_names = FALSE,
                         show_row_dend = FALSE, show_column_dend = FALSE, 
                         column_title = glue('Regions found by {method_name}'), 
                         name=' ', border_gp = gpar(col = "black"), 
                         show_heatmap_legend = T,
                         heatmap_legend_param = list(
                             title = 'Proportion of signals', 
                              title_position = "leftcenter-rot"
                          ),
                          na_col = "#EEEEEE", col = colors)
            plot_list[[length(plot_list) + 1]] <- grid::grid.grabExpr(ComplexHeatmap::draw(p2))
        }
        
    }

    # Save combined plot
    dir.create(config$fig_dir, showWarnings = FALSE, recursive = TRUE)
    output_file <- file.path(config$fig_dir, 
                           paste0(config$output_prefix, '_region_found.png'))

    p_combined <- plot_grid(plotlist = plot_list, ncol = 3, nrow = length(config$slice_coords), 
                            byrow = T, rel_widths = c(1, 1.1, 1.1), rel_heights = rep(1, length(config$slice_coords)))
    save_plot_grid_png(output_file, p_combined, width = 30, height = 28, units = "cm", res = 300)

    if (config$verbose) cat("Visualization saved to:", output_file, "\n")
}

create_summary_plot <- function(PC_results, config) {
    if (!config$create_visualizations) return(NULL)
    
    if (config$verbose) cat("Creating summary plot...\n")
    
    # Number of regions found plot
    max_PC_decisions <- apply(PC_results$all_PC_decisions, 
                            MARGIN = c(2,3), FUN = sum)[c(2, 6),]
    
    df <- data.frame(
        num_rejected = sapply(1:length(config$uprop_s), 
                            function(i) sum(max_PC_decisions[1,] >= i)),
        num_rejected_BH = sapply(1:length(config$uprop_s), 
                                function(i) sum(max_PC_decisions[2,] >= i)),
        l = 1:length(config$uprop_s)
    )
    
    melt_df <- melt(df, id.vars = 'l', 
                   variable.name = 'method', 
                   value.name = 'num_rejected')
    
    p <- ggplot(melt_df, aes(x = l, y = num_rejected, group = method, 
                            shape = method, colour = method, linetype = method)) + 
        geom_line() + geom_point() + 
        labs(x = 'Proportion of signals', y = 'Number of regions found') + 
        theme_bw() + 
        scale_x_discrete(limits = as.character(config$uprop_s)) + 
        scale_color_manual(
            values = c("black", "green"), name = "Method", 
            labels = c("SEFT", "BH")
        ) + 
        scale_shape_manual(
            values = c(3, 4), name = "Method", labels = c("SEFT", "BH")
        ) +
        scale_linetype_manual(
            values = c(1, 2), name = "Method", labels = c("SEFT", "BH")
        ) +  
        theme(legend.position = c(0.8, 0.7), legend.title = element_blank(), 
             legend.text = element_text(size = 8),
             legend.background = element_rect(color = "black", 
                                             linetype = "solid", linewidth = 0.2))
    
    dir.create(config$fig_dir, showWarnings = FALSE, recursive = TRUE)
    output_file <- file.path(config$fig_dir, 
                           paste0(config$output_prefix, '_region_found_number.png'))
    
    png(output_file, width = 13, height = 7, units = 'cm', res = 200)
    print(p)
    dev.off()
    
    if (config$verbose) cat("Summary plot saved to:", output_file, "\n")
}

create_combined_summary_plot <- function(PC_results_1, PC_results_2, config, 
                                         title1 = "Reward Reactivity", 
                                         title2 = "Reward Magnitude Tracking") {
    if (!config$create_visualizations) return(NULL)
    if (config$verbose) cat("Creating combined summary plot...\n")
    
    # Helper to extract df
    extract_df <- function(PC_results, dataset_name) {
        max_PC_decisions <- apply(PC_results$all_PC_decisions, 
                                  MARGIN = c(2,3), FUN = sum)[c(2, 6),]
        df <- data.frame(
            num_rejected = sapply(1:length(config$uprop_s), 
                                  function(i) sum(max_PC_decisions[1,] >= i)),
            num_rejected_BH = sapply(1:length(config$uprop_s), 
                                     function(i) sum(max_PC_decisions[2,] >= i)),
            l = factor(config$uprop_s, levels = config$uprop_s),
            Dataset = dataset_name
        )
        return(df)
    }
    
    df1 <- extract_df(PC_results_1, title1)
    df2 <- extract_df(PC_results_2, title2)
    
    # Combine datasets
    df_combined <- rbind(df1, df2)
    df_combined$Dataset <- factor(df_combined$Dataset, levels = c(title1, title2))
    
    melt_df <- melt(df_combined, id.vars = c('l', 'Dataset'), 
                    variable.name = 'method', 
                    value.name = 'num_rejected')
    
    # Plot
    p <- ggplot(melt_df, aes(x = l, y = num_rejected, group = method, 
                             shape = method, colour = method, linetype = method)) + 
        geom_line() + 
        geom_point() + 
        facet_wrap(~ Dataset, scales = "free_y", ncol = 2) + 
        labs(x = 'Proportion of signals (c)', y = 'Number of regions found') + 
        theme_bw() + 
        scale_color_manual(
            values = c("black", "green"), name = "Method", labels = c("SEFT", "BH")
        ) + 
        scale_shape_manual(
            values = c(3, 4), name = "Method", labels = c("SEFT", "BH")
        ) +
        scale_linetype_manual(
            values = c(1, 2), name = "Method", labels = c("SEFT", "BH")
        ) +  
        theme(
            legend.position = "bottom", 
            legend.title = element_blank(), 
            legend.text = element_text(size = 10),
            legend.background = element_rect(color = "black", linetype = "solid", linewidth = 0.2),
            strip.text = element_text(size = 10, face = "bold"),
            strip.background = element_rect(fill = "grey95")
        )
    
    # Save plot
    dir.create(config$fig_dir, showWarnings = FALSE, recursive = TRUE)
    output_file <- file.path(config$fig_dir, 
                             paste0(config$output_prefix, '_combined_region_found.png'))
    
    png(output_file, width = 20, height = 8, units = 'cm', res = 300)
    print(p)
    dev.off()
    
    if (config$verbose) cat("Combined summary plot saved to:", output_file, "\n")
}

# ==============================================================================
# Results Analysis
# ==============================================================================

analyze_results <- function(PC_results, config) {
    if (config$verbose) cat("Analyzing results...\n")
    
    region_ids <- PC_results$region_ids
    all_PC_e_values <- PC_results$all_PC_e_values
    all_PC_decisions <- PC_results$all_PC_decisions
    nk_s <- PC_results$nk_s
    region_labels_map <- PC_results$region_labels_map
    
    max_PC_decisions <- apply(all_PC_decisions, MARGIN = c(2,3), FUN = sum)
    
    n_regions <- length(region_ids)
    # Safe access checks
    n_uprop <- if (!is.null(dim(all_PC_e_values))) dim(all_PC_e_values)[1] else 0
    n_parasets <- if (!is.null(dim(all_PC_e_values))) dim(all_PC_e_values)[2] else 0
    
    # Choose the same indices used in the original cat() printing:
    # all_PC_e_values[1,2,i] if available, otherwise NA
    e_values_chosen <- rep(NA_real_, n_regions)
    if (n_uprop >= 1 && n_parasets >= 2) {
        for (i in seq_len(n_regions)) {
            e_values_chosen[i] <- all_PC_e_values[1, 2, i]
        }
    }
    
    # Build a region name vector using region_labels_map when available
    region_names <- rep(NA_character_, n_regions)
    if (!is.null(region_labels_map)) {
        # region_labels_map expected to have columns region_id and label
        for (i in seq_len(n_regions)) {
            region_names[i] <- region_labels_map$label[match(region_ids[i], region_labels_map$region_id)]
        }
    }
    
    # Get SEFT and BH counts safely (original code used indices 2 and 6)
    seft_vals <- rep(NA_integer_, n_regions)
    bh_vals <- rep(NA_integer_, n_regions)
    if (!is.null(max_PC_decisions)) {
        if (nrow(max_PC_decisions) >= 2) {
            seft_vals <- max_PC_decisions[2, ]
        }
        if (nrow(max_PC_decisions) >= 6) {
            bh_vals <- max_PC_decisions[6, ]
        }
    }
    
    # Create the table (data.frame) containing the same content as the cat(...) output
    region_table <- data.frame(
        region_id = region_ids,
        region_name = region_names,
        e_value = e_values_chosen,
        nk_s = nk_s,
        e_value_per_nk = ifelse(!is.na(e_values_chosen) & nk_s > 0, e_values_chosen / nk_s, NA_real_),
        SEFT = seft_vals,
        BH = bh_vals,
        stringsAsFactors = FALSE
    )
    
    # Optionally print the same information (preserve original behavior)
    if (config$verbose) {
        for (i in seq_len(n_regions)) {
            cat(glue::glue("{region_table$region_id[i]}: {ifelse(is.na(region_table$region_name[i]), '<NA>', region_table$region_name[i])}"), "\n")
            cat(glue::glue("  e-values: {region_table$e_value[i]}"), "\n")
            cat(glue::glue("  nk_s: {region_table$nk_s[i]}"), "\n")
            cat(glue::glue("  e-values/nk_s: {region_table$e_value_per_nk[i]}"), "\n")
            cat(glue::glue("  SEFT: {region_table$SEFT[i]}"), "\n")
            cat(glue::glue("  BH: {region_table$BH[i]}"), "\n")
        }
    }
    
    # Summarize the number of regions found at different uprop for SEFT (method 2) and BH (method 6)
    regions_found_seft_by_uprop <- regions_found_bh_by_uprop <- NULL
    uprop_values <- if (!is.null(config$uprop_s)) config$uprop_s else {
        if (!is.null(dim(all_PC_decisions))) seq_len(dim(all_PC_decisions)[1]) else NA
    }
    if (!is.null(dim(all_PC_decisions))) {
        n_uprop_dec <- dim(all_PC_decisions)[1]
        regions_found_seft_by_uprop <- numeric(n_uprop_dec)
        regions_found_bh_by_uprop <- numeric(n_uprop_dec)
        for (i in seq_len(n_uprop_dec)) {
            if (dim(all_PC_decisions)[2] >= 2) {
                regions_found_seft_by_uprop[i] <- sum(all_PC_decisions[i, 2, ] > 0, na.rm = TRUE)
            }
            if (dim(all_PC_decisions)[2] >= 6) {
                regions_found_bh_by_uprop[i] <- sum(all_PC_decisions[i, 6, ] > 0, na.rm = TRUE)
            }
        }
    }
    
    summary_stats <- list(
        total_regions = n_regions,
        regions_found_seft = if (!is.null(max_PC_decisions) && nrow(max_PC_decisions) >= 2) sum(max_PC_decisions[2, ] > 0) else 0,
        regions_found_bh = if (!is.null(max_PC_decisions) && nrow(max_PC_decisions) >= 6) sum(max_PC_decisions[6, ] > 0) else 0,
        max_PC_decisions = max_PC_decisions,
        regions_found_seft_by_uprop = if (!is.null(regions_found_seft_by_uprop)) data.frame(uprop = uprop_values, regions_found = regions_found_seft_by_uprop) else NULL,
        regions_found_bh_by_uprop = if (!is.null(regions_found_bh_by_uprop)) data.frame(uprop = uprop_values, regions_found = regions_found_bh_by_uprop) else NULL,
        top_regions_seft = if (!is.null(region_labels_map) && !is.null(max_PC_decisions) && nrow(max_PC_decisions) >= 2) {
            ordered_ids <- region_ids[order(max_PC_decisions[2, ], decreasing = TRUE)]
            region_labels_map$label[match(ordered_ids, region_labels_map$region_id)]
        } else NULL,
        top_regions_bh = if (!is.null(region_labels_map) && !is.null(max_PC_decisions) && nrow(max_PC_decisions) >= 6) {
            ordered_ids <- region_ids[order(max_PC_decisions[6, ], decreasing = TRUE)]
            region_labels_map$label[match(ordered_ids, region_labels_map$region_id)]
        } else NULL,
        region_table = region_table
    )
    
    return(summary_stats)
}

# ==============================================================================
# Main Pipeline Function
# ==============================================================================

run_full_pipeline <- function(config) {
    if (config$verbose) {
        cat("=", rep("=", 60), "\n", sep = "")
        cat("Starting CLAW-based Real Data Analysis Pipeline\n")
        cat("=", rep("=", 60), "\n\n", sep = "")
    }
    
    # Step 1: Load data
    data <- if (!is.null(config$data_format) && tolower(config$data_format) == 'cifti') {
        load_data_cifti(config)
    } else {
        load_data(config)
    }
    
    # Step 2: Apply preprocessing (wavelet denoising if requested)
    if (config$use_wavelet_denoising) {
        data$t_data <- apply_wavelet_denoising(data$t_data, data$data_mask, config)
        data$data_mask <- data$t_data != 0
        # Re-mask region masks without assuming NIfTI slots
        data$region_masks[data$data_mask == 0] <- 0
        data$region_ids <- unique(c(data$region_masks))
        data$region_ids <- data$region_ids[data$region_ids != 0]
    }
    
    # Step 3: Calculate CLAW scores
    results <- calculate_CLAW_scores(data, config)
    
    # Step 4: Run PC testing
    PC_results <- run_PC_testing(data, results, config)
    
    # Step 5: Create visualizations
    if (config$create_visualizations) {
        create_region_visualization(data, PC_results, config)
        create_summary_plot(PC_results, config)
    }
    
    # Step 6: Analyze results
    summary_stats <- analyze_results(PC_results, config)
    
    if (config$verbose) {
        cat("\n", "=", rep("=", 60), "\n", sep = "")
        cat("Pipeline completed successfully!", "\n")
        cat("=", rep("=", 60), "\n", sep = "")
    }
    
    return(list(
        data = data,
        CLAW_results = results,
        PC_results = PC_results,
        summary_stats = summary_stats
    ))
}

create_mdd_motivating_figure <- function(data_obj, output_file = file.path("outputs", "figures", "MNI152_reward_region_2.png")) {
    reward_data <- data_obj$t_data
    atlas_aal3 <- data_obj$atlas
    region_masks <- data_obj$region_masks
    transformM <- data_obj$transformM
    color_func2 <- colorRamp2(c(-4, 0, 4), c("blue", "white", "red"))

    z_idx <- round((solve(transformM) %*% c(1, 1, -6, 1))[3])
    ids <- matrix(c(1, dim(reward_data)[1], 1, dim(reward_data)[2], z_idx, z_idx), nrow = 3, byrow = TRUE)

    mask <- region_masks[ids[1, 1]:ids[1, 2], ids[2, 1]:ids[2, 2], ids[3, 1]:ids[3, 2]]
    data_mat <- reward_data[ids[1, 1]:ids[1, 2], ids[2, 1]:ids[2, 2], ids[3, 1]:ids[3, 2]]
    data_mat[mask == 0] <- NA
    na_ids <- which(data_mat == 0)
    data_mat[na_ids] <- NA

    p1 <- Heatmap(
        t(data_mat[, dim(data_mat)[2]:1]),
        width = unit(8, "cm"),
        height = unit(8, "cm"),
        cluster_rows = FALSE,
        cluster_columns = FALSE,
        show_row_names = FALSE,
        show_column_names = FALSE,
        show_row_dend = FALSE,
        show_column_dend = FALSE,
        column_title = "Reward Reactivity (Z-stat), z=-6",
        name = "Z-score",
        border_gp = gpar(col = "black"),
        show_heatmap_legend = TRUE,
        na_col = "#EEEEEE",
        col = color_func2
    )

    mat <- atlas_aal3[ids[1, 1]:ids[1, 2], ids[2, 1]:ids[2, 2], ids[3, 1]:ids[3, 2]]
    mat[mask == 0] <- NA
    mat[na_ids] <- NA
    region_vals <- sort(unique(as.vector(mat)))
    region_vals <- region_vals[!is.na(region_vals)]
    region_cols <- setNames(circlize::rand_color(length(region_vals), luminosity = "bright"), as.character(region_vals))
    p2 <- Heatmap(
        t(mat[, dim(mat)[2]:1]),
        width = unit(8, "cm"),
        height = unit(8, "cm"),
        cluster_rows = FALSE,
        cluster_columns = FALSE,
        show_row_names = FALSE,
        show_column_names = FALSE,
        show_row_dend = FALSE,
        show_column_dend = FALSE,
        column_title = "AAL3 Brain Regions",
        name = "Region",
        border_gp = gpar(col = "black"),
        na_col = "#EEEEEE",
        show_heatmap_legend = FALSE,
        col = region_cols
    )

    dir.create(dirname(output_file), recursive = TRUE, showWarnings = FALSE)
    if (capabilities("cairo")) {
        png(output_file, width = 20, height = 9, units = "cm", res = 300, type = "cairo")
    } else {
        png(output_file, width = 20, height = 9, units = "cm", res = 300)
    }
    on.exit(dev.off(), add = TRUE)

    p_combined <- plot_grid(
        plotlist = list(
            grid::grid.grabExpr(ComplexHeatmap::draw(p1)),
            grid::grid.grabExpr(ComplexHeatmap::draw(p2))
        ),
        ncol = 2, nrow = 1, byrow = FALSE, rel_widths = c(1, 1), align = "hv"
    )
    print(p_combined)
}

run_mdd_paper_workflow <- function(
    data_dir = file.path("real_data", "input", "reward_MDD"),
    atlas_path = file.path("real_data", "input", "atlas", "AAL3v1.nii.gz"),
    atlas_label_path = file.path("real_data", "input", "atlas", "AAL3v1.nii.txt"),
    analysis_dir = file.path("outputs", "intermediate", "real_data", "analysis"),
    figure_dir = file.path("outputs", "figures"),
    table_dir = file.path("outputs", "tables")
) {
    dir.create(analysis_dir, recursive = TRUE, showWarnings = FALSE)
    dir.create(figure_dir, recursive = TRUE, showWarnings = FALSE)
    dir.create(table_dir, recursive = TRUE, showWarnings = FALSE)

    wavelet_params <- list(wf = "d8", J = 7, expand_to = c(128, 128, 128), threshold_type = "hard")

    cfg_reactivity <- create_config(
        data_path = file.path(data_dir, "zstat1.nii.gz"),
        atlas_path = atlas_path,
        atlas_label_path = atlas_label_path,
        output_dir = analysis_dir,
        output_prefix = "MDD_denoised_zstat1",
        use_wavelet_denoising = TRUE,
        data_format = "nifti",
        fig_dir = figure_dir,
        wavelet_params = wavelet_params
    )
    res_reactivity <- run_full_pipeline(cfg_reactivity)

    cfg_tracking <- create_config(
        data_path = file.path(data_dir, "zstat1_3.nii.gz"),
        atlas_path = atlas_path,
        atlas_label_path = atlas_label_path,
        output_dir = analysis_dir,
        output_prefix = "MDD_denoised_zstat1_3",
        use_wavelet_denoising = TRUE,
        data_format = "nifti",
        fig_dir = figure_dir,
        wavelet_params = wavelet_params
    )
    res_tracking <- run_full_pipeline(cfg_tracking)

    create_combined_summary_plot(res_reactivity$PC_results, res_tracking$PC_results, cfg_reactivity)
    create_mdd_motivating_figure(res_reactivity$data, file.path(figure_dir, "MNI152_reward_region_2.png"))

    region_table <- res_tracking$summary_stats$region_table
    region_table <- region_table[region_table$SEFT >= 4, ]
    region_table <- region_table[order(region_table$SEFT, decreasing = TRUE), ]
    fwrite(region_table, file.path(table_dir, "tab_brain_region_raw.csv"))

    invisible(list(
        reward_reactivity = res_reactivity,
        reward_tracking = res_tracking,
        region_table = region_table
    ))
}

# ==============================================================================
# Command-line Entry
# ==============================================================================

if (sys.nframe() == 0) {
    args <- commandArgs(trailingOnly = TRUE)
    task <- if (length(args) > 0) args[[1]] else "run_mdd_paper"

    if (task == "run_mdd_paper") {
        run_mdd_paper_workflow()
    } else {
        stop("Unknown task. Supported task: run_mdd_paper")
    }
}
