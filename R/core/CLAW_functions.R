library(Rcpp)
library(RcppArmadillo)
library(waveslim)

resolve_project_file <- function(rel_path) {
    candidates <- c(
        rel_path,
        file.path("..", rel_path),
        file.path("..", "..", rel_path),
        file.path("..", "..", "..", rel_path)
    )
    hits <- candidates[file.exists(candidates)]
    if (length(hits) == 0) {
        stop(sprintf("Cannot locate required file: %s", rel_path))
    }
    normalizePath(hits[[1]], winslash = "/", mustWork = TRUE)
}

sourceCpp(resolve_project_file(file.path("src", "CLAW_functions.cpp")), rebuild = FALSE)
print('CLAW cpp functions loaded.')

generate_dense_shape_2d_data <- function(L, mu, shape = 'circle', relsize=0.5, sparsity=0.5) {
  # Create a 2d grid
  # Generate latent states
  thetas = matrix(0, nrow = L, ncol = L)
  
  # Define the coordinates of the center
  center_x <- floor(L / 2) + 1
  center_y <- floor(L / 2) + 1
  
  # Create a mesh grid
  x_grid <- rep(1:L, each = L)
  y_grid <- rep(1:L, times = L)
  x_grid <- matrix(x_grid, nrow = L, ncol = L, byrow = F)
  y_grid <- matrix(y_grid, nrow = L, ncol = L, byrow = F)
  if (shape == 'circle') {  
    # Calculate distance from the center
    distances <- sqrt((x_grid - center_x)^2 + (y_grid - center_y)^2)
    # Assign 1 to the elements within the triangle region
    thetas[distances <= 1/2*relsize*L] <- 1
  } else if (shape == 'disk') {
    # Calculate distance from the center
    distances <- sqrt((x_grid - center_x)^2 + (y_grid - center_y)^2)
    # Assign 1 to the elements within the triangle region
    thetas[distances <= relsize*L & distances >= 1/3*relsize*L] <- 1
  
  } else if (shape == 'triangle') {
    vertex1 <- c(floor(L/2), floor(L/2+relsize*L/4*sqrt(3)))
    vertex2 <- c(floor(L/2-relsize*L/2), floor(L/2-relsize*L/4*sqrt(3)))
    vertex3 <- c(floor(L/2+relsize*L/2), floor(L/2-relsize*L/4*sqrt(3)))

    # Calculate the signs of the cross products
    cross_product1 <- (x_grid - vertex1[1]) * (vertex2[2] - vertex1[2]) - (y_grid - vertex1[2]) * (vertex2[1] - vertex1[1])
    cross_product2 <- (x_grid - vertex2[1]) * (vertex3[2] - vertex2[2]) - (y_grid - vertex2[2]) * (vertex3[1] - vertex2[1])
    cross_product3 <- (x_grid - vertex3[1]) * (vertex1[2] - vertex3[2]) - (y_grid - vertex3[2]) * (vertex1[1] - vertex3[1])
    
    # Check if the signs are all the same (inside the triangle)
    inside_triangle <- (cross_product1 >= 0 & cross_product2 >= 0 & cross_product3 >= 0) |
                       (cross_product1 <= 0 & cross_product2 <= 0 & cross_product3 <= 0)
    
    # Assign 1 to the elements within the triangle region
    thetas[inside_triangle] <- 1
    thetas = t(thetas)
  } else if (shape == 'rectangle') {
    thetas[(center_x - floor(relsize*L/2)):(center_x + floor(relsize*L/2)), (center_y - floor(relsize*L/2)):(center_y + floor(relsize*L/2))] <- 1
  } else if (shape == 'iid') {
    thetas = matrix(rbinom(n=L^2, 1, prob=sparsity), nrow = L, ncol = L)
  }
  Xs = thetas * matrix(rnorm(L^2,mean=mu,sd=1),ncol=L) + (1-thetas) * matrix(rnorm(L^2,mean=0,sd=1),ncol=L)
  # Return the data and latent states
  return(list(Xs = Xs, thetas = thetas))
}

generate_3d_signal <- function(dim, mu, shape='sphere', radius, coord, decay_method='none', mix_method='add', FWHM = 1){
    # Generate a 3d signal.
    # dim: dimension of the grid.
    # mu: mean of the signal.
    # coord: a matrix, each row is 3d coordinates of the signal.
    # decay_method: decay method of the signal.
    # FWHM: full width at half maximum.
    # Returns a 3d signal.
    signal = array(0, dim = dim)
    mask = array(0, dim = dim)
    if (shape=='sphere'){
        for (i in 1:nrow(coord)){
            coords = coord[i,]
            for (x in floor(coords[1]-radius):ceiling(coords[1]+radius)){
                for (y in floor(coords[2]-radius):ceiling(coords[2]+radius)){
                    for (z in floor(coords[3]-radius):ceiling(coords[3]+radius)){
                        if (((x-coords[1])^2 + (y-coords[2])^2 + (z-coords[3])^2 <= radius^2)&
                            x>=1 & x<=dim[1] & y>=1 & y<=dim[2] & z>=1 & z<=dim[3]){
                            mask[x,y,z] = 1
                            if (decay_method=='none'){
                                if (mix_method=='add') signal[x,y,z] = signal[x,y,z] + mu
                                else if (mix_method=='max') signal[x,y,z] = max(signal[x,y,z], mu)                            
                            } else if (decay_method=='gaussian'){
                                sigma = FWHM / sqrt(8 * log(2))
                                if (mix_method=='add') signal[x,y,z] = signal[x,y,z] + mu * dnorm(sqrt((x-coords[1])^2 + (y-coords[2])^2 + (z-coords[3])^2),sd=sigma)/dnorm(0,sd=sigma)
                                else if (mix_method=='max') signal[x,y,z] = max(signal[x,y,z], mu * dnorm(sqrt((x-coords[1])^2 + (y-coords[2])^2 + (z-coords[3])^2),sd=sigma)/dnorm(0,sd=sigma))
                            } else {
                                stop('decay_method not supported.')
                            }
                        }
                    }
                }
            }
        }
    } else if (shape=='cube'){
        for (i in 1:nrow(coord)){
            coords = coord[i,]
            for (x in floor(coords[1]-radius):ceiling(coords[1]+radius)){
                for (y in floor(coords[2]-radius):ceiling(coords[2]+radius)){
                    for (z in floor(coords[3]-radius):ceiling(coords[3]+radius)){
                        if (abs(x-coords[1]) <= radius & abs(y-coords[2]) <= radius & abs(z-coords[3]) <= radius
                            &(x>=1 & x<=dim[1] & y>=1 & y<=dim[2] & z>=1 & z<=dim[3])){
                            mask[x,y,z] = 1
                            if (decay_method=='none'){
                                if (mix_method=='add') signal[x,y,z] = signal[x,y,z] + mu
                                else if (mix_method=='max') signal[x,y,z] = max(signal[x,y,z], mu)                            
                            } else if (decay_method=='gaussian'){
                                sigma = FWHM / sqrt(8 * log(2))
                                if (mix_method=='add') signal[x,y,z] = signal[x,y,z] + mu * dnorm(sqrt((x-coords[1])^2 + (y-coords[2])^2 + (z-coords[3])^2),sd=sigma)/dnorm(0,sd=sigma)
                                else if (mix_method=='max') signal[x,y,z] = max(signal[x,y,z], mu * dnorm(sqrt((x-coords[1])^2 + (y-coords[2])^2 + (z-coords[3])^2),sd=sigma)/dnorm(0,sd=sigma))
                            } else {
                                stop('decay_method not supported.')
                            }
                        }
                    }
                }
            }
        }
    }else {
        stop('shape not supported.')
    }

    return(list(signal = signal, mask = mask))
}


# Function to calculate PC set-wise theta.
# theta: individual theta, 0 as null, 1 as non-null.
# u-1: num. of true signals under the null.
# Returns set-wise theta.
merge_PC_theta <- function(theta, u) {
    return(ifelse(sum(theta) >= u, 1, 0))
}

# Function to calculate False Discovery Proportion (FDP) given theta and decision.
# theta: individual theta, 0 as null, 1 as non-null.
# decision: 0 as non-rejection, 1 as rejection.
# Returns FDP.
cal_FDP <- function(theta, decision) {
    return(sum(decision * (1-theta)) / max(sum(decision), 1))
}

# Function to calculate Missed Set Rate (MSR), i.e. the proportion of missed discoveries.
# theta: individual theta, 0 as null, 1 as non-null.
# decision: 0 as non-rejection, 1 as rejection.
# Returns MSR.
cal_MSR <- function(theta, decision) {
    return(sum(theta * (1-decision)) / max(sum(theta), 1))
}

# Function to perform Benjamini-Hochberg (BH) method for multiple testing correction.
# p_values: p-values.
# alpha: FDR.
# Returns decisions vector.
BH <- function(p_values, alpha) {
    p_order <- order(p_values) 
    m <- length(p_values)
    rejected <- rep(0, m)
    
    for (i in m:1) {
        if (p_values[p_order][i] <= alpha * i / m) {
            rejected[p_order][1:i] <- 1
            break
        }
    }
    
    return(rejected)
}

# ==============================================================================
# Wavelet Denoising Function
# ==============================================================================

#' Apply 3D wavelet denoising to data
#' 
#' This function applies hard thresholding wavelet denoising to 3D data arrays.
#' The denoising procedure is consistent across simulations and real data analysis.
#' 
#' @param z_map A 3D array containing the data to be denoised
#' @param expand_to Optional. If provided, the data will be expanded to these dimensions before denoising.
#'                  This is useful when the data dimensions are not powers of 2.
#' @param data_mask Optional. A 3D array of the same dimensions as z_map indicating which voxels to preserve.
#'                  If provided, the mask will be applied after denoising.
#' @param wf Wavelet filter to use. Default is "d8" (Daubechies 8).
#' @param J Number of decomposition levels. Default is 6.
#' @param verbose Logical. If TRUE, prints progress messages. Default is FALSE.
#' 
#' @return A 3D array of the same dimensions as the input (or original dimensions if expanded),
#'         containing the denoised data.
#' 
#' @details
#' The denoising procedure:
#' 1. Optionally expands data to specified dimensions (for real data with non-power-of-2 dimensions)
#' 2. Performs 3D discrete wavelet transform using dwt.3d
#' 3. Applies hard thresholding to detail coefficients (where the 4th character of coefficient name <= 1)
#' 4. Threshold is calculated as sqrt(2 * log(n)) * lambda, where lambda is estimated from median absolute deviation
#' 5. Reconstructs the signal using idwt.3d
#' 6. Optionally applies mask and crops back to original dimensions
apply_wavelet_denoising_3d <- function(z_map, expand_to = NULL, data_mask = NULL, 
                                        wf = "d8", target_level = 1, J = 6, verbose = FALSE) {
    
    if (verbose) cat("Applying wavelet denoising...\n")
    
    # Hard threshold function
    hard_threshold <- function(x, thresh) {
        ifelse(abs(x) > thresh, x, 0)
    }
    
    # Store original dimensions
    orig_dims <- dim(z_map)
    
    # Expand data if needed (for real data with non-power-of-2 dimensions)
    if (!is.null(expand_to)) {
        if (verbose) cat(paste("Expanding data from", paste(orig_dims, collapse='x'), "to", paste(expand_to, collapse='x'), "...\n"))
        data_raw <- array(0, dim = expand_to)
        data_raw[1:orig_dims[1], 1:orig_dims[2], 1:orig_dims[3]] <- z_map
    } else {
        data_raw <- z_map
    }
    
    # Calculate threshold
    # For expanded data, use only non-zero voxels; for original data, use all
    if (!is.null(expand_to)) {
        n <- length(data_raw[data_raw != 0])
    } else {
        n <- length(data_raw)
    }
    thresh_value <- sqrt(2 * log(n))
    
    if (verbose) cat(paste("Using threshold:", round(thresh_value, 4), "(n =", n, ")\n"))
    
    # Wavelet decomposition
    if (verbose) cat(paste("Performing ", J, "-level wavelet decomposition with filter '", wf, "'...\n", sep = ""))
    wt <- dwt.3d(data_raw, wf = wf, J = J)
    
    # Apply thresholding to detail coefficients
    # Only threshold coefficients where the 4th character of the name is <= 1
    if (verbose) cat("Applying hard thresholding to detail coefficients...\n")
    for (name in names(wt)) {
        level <- as.numeric(sub(".*([0-9]+)$", "\\1", name))
        if (!is.na(level) && level %in% target_level) {

            non_zero <- wt[[name]][wt[[name]] != 0] 
            if (length(non_zero) > 0) {
                sigma_hat <- median(abs(non_zero)) / 0.6745
                thr <- thresh_value * sigma_hat
                wt[[name]] <- hard_threshold(wt[[name]], thr)
            }
        }
    }
    
    # Inverse transform
    if (verbose) cat("Reconstructing denoised signal...\n")
    z_map_denoised <- idwt.3d(wt)
    
    # Apply mask if provided (for real data)
    if (!is.null(expand_to)) {
        z_map_denoised[data_raw == 0] <- 0
        z_map_denoised <- z_map_denoised[1:orig_dims[1], 1:orig_dims[2], 1:orig_dims[3]]
    }
    
    # Apply additional mask if provided
    if (!is.null(data_mask)) {
        z_map_denoised[data_mask == 0] <- 0
    }
    
    if (verbose) cat("Wavelet denoising complete.\n")
    
    return(z_map_denoised)
}
