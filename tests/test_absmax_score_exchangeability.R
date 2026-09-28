#!/usr/bin/env Rscript

file_arg <- sub(
    "^--file=", "",
    grep("^--file=", commandArgs(FALSE), value = TRUE)[[1]]
)
project_root <- normalizePath(file.path(dirname(file_arg), ".."))
setwd(project_root)

Sys.setenv(
    SEFT_WORKING_MODEL_CPP = file.path(
        project_root, "src", "working_models",
        "working_model_scores.cpp"
    ),
    SEFT_ML_WORKING_MODEL_SCRIPT = file.path(
        project_root, "python", "working_models",
        "ml_working_model.py"
    ),
    SEFT_ML_PYTHON = Sys.getenv("SEFT_ML_PYTHON", unset = Sys.which("python3")),
    SEFT_FDRS_ABSMAX_SCRIPT = file.path(
        project_root, "python", "working_models",
        "fdr_smoothing_absmax.py"
    ),
    SEFT_FDRS_PYTHON = Sys.getenv("SEFT_ML_PYTHON", unset = Sys.which("python3")),
    OMP_NUM_THREADS = "1",
    MKL_NUM_THREADS = "1"
)

source(file.path(
    project_root, "R", "working_models",
    "working_model_scores.R"
))
source(file.path(
    project_root, "R", "working_models",
    "ml_working_model_scores.R"
))

set.seed(91)

small <- c(4L, 4L, 4L)
x <- array(rnorm(prod(small)), dim = small)
x_til <- array(rnorm(prod(small)), dim = small)
fdrs <- audit_pairwise_exchangeability(
    cal_fdr_smoothing_scores_3d_absmax,
    x,
    x_til,
    swap_fraction = 0.3,
    seed = 0L,
    tolerance = 1e-10,
    pr_table_size = 1024L
)
stopifnot(fdrs$passed)
fdrs_axis <- audit_pairwise_exchangeability(
    cal_fdr_smoothing_scores_3d_absmax,
    x,
    x_til,
    swap_fraction = 0.3,
    seed = 0L,
    tolerance = 1e-10,
    pr_table_size = 1024L,
    trail_mode = "axis"
)
stopifnot(fdrs_axis$passed)
fdrs_axis_persistent <- audit_pairwise_exchangeability(
    cal_fdr_smoothing_scores_3d_absmax,
    x,
    x_til,
    swap_fraction = 0.3,
    seed = 0L,
    tolerance = 1e-10,
    pr_table_size = 1024L,
    trail_mode = "axis",
    solver_mode = "persistent"
)
stopifnot(fdrs_axis_persistent$passed)
fdrs_grid_primal_dual <- audit_pairwise_exchangeability(
    cal_fdr_smoothing_scores_3d_absmax,
    x,
    x_til,
    swap_fraction = 0.3,
    seed = 0L,
    tolerance = 1e-10,
    pr_table_size = 1024L,
    solver_mode = "grid_primal_dual",
    grid_maxsteps = 400L,
    grid_converge = 1e-5
)
stopifnot(fdrs_grid_primal_dual$passed)

ml_dims <- c(12L, 12L, 12L)
x <- array(rnorm(prod(ml_dims)), dim = ml_dims)
x_til <- array(rnorm(prod(ml_dims)), dim = ml_dims)
deep <- audit_pairwise_exchangeability(
    cal_deepfdr_scores_3d_absmax,
    x,
    x_til,
    swap_fraction = 0.2,
    seed = 0L,
    tolerance = 1e-10,
    device = "cpu",
    kde_grid_size = 4096L
)
print(list(deepfdr_exchangeability = deep))
stopifnot(deep$passed)

fc <- audit_pairwise_exchangeability(
    cal_fchmrf_scores_3d_absmax,
    x,
    x_til,
    swap_fraction = 0.2,
    seed = 0L,
    tolerance = 1e-10,
    appearance_mode = "background",
    kde_grid_size = 4096L
)
print(list(fchmrf_exchangeability = fc))
stopifnot(fc$passed)

print(list(
    fdr_smoothing = fdrs,
    fdr_smoothing_axis = fdrs_axis,
    fdr_smoothing_axis_persistent = fdrs_axis_persistent,
    fdr_smoothing_grid_primal_dual = fdrs_grid_primal_dual,
    deepfdr = deep,
    fchmrf = fc
))
cat("ABSMAX_SCORE_EXCHANGEABILITY_OK\n")
