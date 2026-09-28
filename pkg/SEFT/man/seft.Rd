\name{seft}
\alias{seft}
\title{Run SEFT inference}
\usage{
seft(
  zmap = NULL,
  tstat = NULL,
  atlas,
  atlas_labels = NULL,
  mask = NULL,
  df = NULL,
  design = NULL,
  working_model = "co",
  alpha = 0.1,
  pc_levels = c(0.1, 0.2, 0.3),
  denoise = c("none", "wavelet"),
  bandwidth = 5,
  neighbor_range = 10L,
  lambda = 0.5,
  score_clip = 0.99,
  simes = FALSE,
  seed = 1L,
  out_dir = NULL,
  prefix = "seft",
  keep_scores = FALSE,
  verbose = interactive()
)
}
\arguments{
\item{zmap}{Signed Z map as a NIfTI path, RNifti image, or numeric three-dimensional array.}
\item{tstat}{Student t statistic map. Exactly one of \code{zmap} and \code{tstat} is required.}
\item{atlas}{Integer-valued region atlas.}
\item{atlas_labels}{Optional region-label file, data frame, or named character vector.}
\item{mask}{Optional analysis mask.}
\item{df}{Residual degrees of freedom for \code{tstat}.}
\item{design}{Numeric design matrix or FSL VEST matrix path used to derive residual degrees of freedom.}
\item{working_model}{One or more of \code{"co"}, \code{"ising"}, \code{"fdr-smoothing"}, \code{"deepfdr"}, and \code{"fchmrf"}. The last three use an optional external environment.}
\item{alpha}{Multiple-testing level.}
\item{pc_levels}{Partial-conjunction proportions.}
\item{denoise}{Either \code{"none"} or \code{"wavelet"}.}
\item{bandwidth}{CO spatial bandwidth in voxels.}
\item{neighbor_range}{CO neighbourhood radius in voxels.}
\item{lambda}{CO sparsity tuning parameter.}
\item{score_clip}{CO score clipping constant.}
\item{simes}{Whether to include BHe+Simes.}
\item{seed}{Random seed.}
\item{out_dir}{Optional output directory.}
\item{prefix}{Output filename prefix.}
\item{keep_scores}{Whether to retain voxelwise scores.}
\item{verbose}{Whether to report progress.}
}
\value{An object of class \code{seft_result}.}
\description{Runs region-level spatial e-value inference using compiled CO or Ising working models.}
\examples{
dims <- c(4, 4, 4)
z <- array(rnorm(prod(dims)), dims)
atlas <- array(rep(1:2, each = prod(dims) / 2), dims)
result <- seft(zmap = z, atlas = atlas, neighbor_range = 1, seed = 1)
print(result)
}
