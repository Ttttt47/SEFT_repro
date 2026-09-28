.bh_decision <- function(p_values, alpha) {
    p_values[!is.finite(p_values)] <- 1
    p_values <- pmin(pmax(p_values, 0), 1)
    order_idx <- order(p_values)
    m <- length(p_values)
    rejected <- integer(m)
    eligible <- which(p_values[order_idx] <= alpha * seq_len(m) / m)
    if (length(eligible)) rejected[order_idx[seq_len(max(eligible))]] <- 1L
    rejected
}

.e_bh_decision <- function(e_values, alpha) {
    e_values[is.na(e_values) | e_values < 0] <- 0
    order_idx <- order(e_values, decreasing = TRUE)
    m <- length(e_values)
    eligible <- which(e_values[order_idx] >= m / (alpha * seq_len(m)))
    rejected <- integer(m)
    if (length(eligible)) rejected[order_idx[seq_len(max(eligible))]] <- 1L
    rejected
}

.pc_evalue <- function(scores, region_mask, u) {
    log_scores <- matrix(c(scores$log_R[region_mask], scores$log_R_til[region_mask]), nrow = 2L, byrow = TRUE)
    offset <- suppressWarnings(max(log_scores))
    if (!is.finite(offset)) return(NA_real_)
    cal_ELIS_cpp(log_scores - offset, as.integer(u), method = "Sym_miny", til_mth = "neglog", signmin_flag = FALSE)
}

.regional_inference <- function(score_results, z, region_map, region_ids, labels, pc_levels, alpha, simes) {
    label_lookup <- setNames(labels$label, labels$region_id)
    rows <- vector("list", length(pc_levels) * length(region_ids) * (length(score_results) + as.integer(simes)))
    row_index <- 1L
    for (pc_level in pc_levels) {
        n_voxels <- vapply(region_ids, function(id) sum(region_map == id), integer(1))
        u_values <- pmax(1L, as.integer(ceiling(n_voxels * pc_level)))
        for (model in names(score_results)) {
            e_values <- vapply(seq_along(region_ids), function(i) .pc_evalue(score_results[[model]], region_map == region_ids[[i]], u_values[[i]]), numeric(1))
            decisions <- .e_bh_decision(e_values, alpha)
            p_values <- ifelse(is.na(e_values), 1, pmin(1, 1 / e_values))
            method <- unname(c(
                co = "SEFT-CO", ising = "SEFT-Ising",
                `fdr-smoothing` = "SEFT-FDR smoothing",
                deepfdr = "SEFT-DeepFDR", fchmrf = "SEFT-fcHMRF"
            )[[model]])
            for (i in seq_along(region_ids)) {
                rows[[row_index]] <- data.frame(
                    pc_level = pc_level, method = method, working_model = model,
                    model_identifier = score_results[[model]]$model_identifier,
                    region_id = region_ids[[i]], region_label = unname(label_lookup[as.character(region_ids[[i]])]),
                    n_voxels = n_voxels[[i]], u = u_values[[i]], e_value = e_values[[i]],
                    pc_p_value = p_values[[i]], significant = decisions[[i]], stringsAsFactors = FALSE
                )
                row_index <- row_index + 1L
            }
        }
        if (simes) {
            p_values <- vapply(seq_along(region_ids), function(i) {
                voxel_p <- 2 * stats::pnorm(-abs(z[region_map == region_ids[[i]]]))
                min(1, max(0, Simes_PC_test_cpp(voxel_p, u_values[[i]])))
            }, numeric(1))
            decisions <- .bh_decision(p_values, alpha)
            for (i in seq_along(region_ids)) {
                rows[[row_index]] <- data.frame(
                    pc_level = pc_level, method = "BHe", working_model = "none", model_identifier = "bhe_simes",
                    region_id = region_ids[[i]], region_label = unname(label_lookup[as.character(region_ids[[i]])]),
                    n_voxels = n_voxels[[i]], u = u_values[[i]],
                    e_value = if (p_values[[i]] > 0) 1 / p_values[[i]] else Inf,
                    pc_p_value = p_values[[i]], significant = decisions[[i]], stringsAsFactors = FALSE
                )
                row_index <- row_index + 1L
            }
        }
    }
    do.call(rbind, rows[seq_len(row_index - 1L)])
}

.summarise_regions <- function(regions, alpha, seed) {
    groups <- interaction(regions$pc_level, regions$method, drop = TRUE, lex.order = TRUE)
    pieces <- split(regions, groups)
    result <- lapply(pieces, function(x) data.frame(
        pc_level = x$pc_level[[1L]], method = x$method[[1L]],
        total_regions = nrow(x), discovered_regions = sum(x$significant == 1L),
        alpha = alpha, seed = as.integer(seed), stringsAsFactors = FALSE
    ))
    output <- do.call(rbind, result)
    rownames(output) <- NULL
    output[order(output$pc_level, match(output$method, c("SEFT-CO", "SEFT-Ising", "SEFT-FDR smoothing", "SEFT-DeepFDR", "SEFT-fcHMRF", "BHe"))), , drop = FALSE]
}
