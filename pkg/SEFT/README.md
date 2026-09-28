# SEFT

This directory contains an internal review build of the SEFT R package. It provides region-level inference from a signed Z map or t-statistic map. CO and Ising are built in; FDR smoothing, DeepFDR and fcHMRF are optional. The package is not a public release.

Package code is MIT-licensed. Bundled AAL3 atlas files are third-party material under the GNU GPL; see the [publisher](https://www.gin.cnrs.fr/en/tools/aal/) and `inst/NOTICE`.

## Installation

SEFT requires R 4.1 or later, a C++17 compiler, Rcpp, RcppArmadillo and RNifti. Install the R dependencies first if they are not already available:

```r
install.packages(c("Rcpp", "RcppArmadillo", "RNifti"), repos = "https://cloud.r-project.org")
```

From the root of the reproduction repository, run:

```bash
R CMD INSTALL --clean pkg/SEFT
```

`R CMD INSTALL` does not download missing dependencies; it stops and lists the unavailable packages. Once the three R dependencies are installed, the base package can be installed and used offline. Wavelet denoising is disabled by default and additionally requires the optional R package `waveslim`. The analysis examples below enable it explicitly with `denoise = "wavelet"`; install it with `install.packages("waveslim")`, or omit that argument to run without denoising.

## R interface

`seft()` is the main user function. It takes a voxelwise statistic map and an integer-valued atlas, fits one or more working models, performs regional partial-conjunction testing and returns all methods in a common result object.

### Minimal example

```r
library(SEFT)

result <- seft(
  zmap = "contrast_signed_z.nii.gz",
  atlas = "AAL3v1.nii.gz",
  atlas_labels = "AAL3v1.nii.txt",
  working_model = "co",
  denoise = "wavelet",
  out_dir = "seft_output"
)
```

### Statistic and atlas inputs

Supply exactly one statistic input:

- `zmap`: a signed Z-statistic map.
- `tstat`: a t-statistic map together with either residual `df` or a numeric/FSL VEST `design` matrix; SEFT converts it to signed Z internally.

| Input | Accepted values | Requirements |
|---|---|---|
| `zmap` | NIfTI path, `RNifti` image or numeric 3-D array | Signed voxelwise Z statistics; mutually exclusive with `tstat` |
| `tstat` | NIfTI path, `RNifti` image or numeric 3-D array | Signed voxelwise t statistics; requires exactly one of `df` and `design` |
| `atlas` | NIfTI path, `RNifti` image or numeric 3-D array | Non-negative integer region identifiers; 0 is background |
| `mask` | NIfTI path, `RNifti` image, numeric array or logical array | Optional; positive or `TRUE` voxels are retained |
| `atlas_labels` | File path, data frame or named character vector | Optional mapping from atlas identifiers to display labels |
| `df` | One positive number | Residual degrees of freedom for `tstat` |
| `design` | Numeric matrix or matrix-file path | Used to calculate residual degrees of freedom as rows minus matrix rank |

The statistic map, atlas and optional mask must have identical dimensions. NIfTI inputs must also have matching affines. Non-finite statistic values and voxels outside the atlas or mask are excluded. Empty atlas regions are omitted; atlas identifiers do not need to be consecutive.

For a signed Z-map stored as NIfTI:

```r
result <- seft(
  zmap = "contrast_signed_z.nii.gz",
  atlas = "AAL3v1.nii.gz",
  mask = "analysis_mask.nii.gz",
  atlas_labels = "AAL3v1.nii.txt",
  denoise = "wavelet"
)
```

For a t-statistic map with known residual degrees of freedom:

```r
result <- seft(
  tstat = "contrast_tstat.nii.gz",
  df = 307,
  atlas = "AAL3v1.nii.gz"
)
```

For in-memory arrays and a named label vector:

```r
z <- array(rnorm(20 * 20 * 20), dim = c(20, 20, 20))
atlas <- array(0L, dim = dim(z))
atlas[1:10, , ] <- 1L
atlas[11:20, , ] <- 2L
labels <- c(`1` = "Region A", `2` = "Region B")
result <- seft(zmap = z, atlas = atlas, atlas_labels = labels)
```

An `RNifti` image can be passed without converting it to an array:

```r
z_image <- RNifti::readNifti("contrast_signed_z.nii.gz")
atlas_image <- RNifti::readNifti("AAL3v1.nii.gz")
result <- seft(zmap = z_image, atlas = atlas_image)
```

### Working models and comparator

| Value | Method | Availability |
|---|---|---|
| `"co"` | SEFT-CO | Built in; default |
| `"ising"` | SEFT-Ising | Built in |
| `"fdr-smoothing"` | SEFT-FDR smoothing | Optional environment |
| `"deepfdr"` | SEFT-DeepFDR | Optional environment |
| `"fchmrf"` | SEFT-fcHMRF | Optional environment |

Pass a character vector such as `working_model = c("co", "ising")` to compare models using the same input and mirror statistics. `simes = TRUE` additionally runs BHe+Simes; BHe is a regional comparator and is therefore not a `working_model` value.

### Main arguments

| Argument | Default | Meaning |
|---|---:|---|
| `alpha` | `0.1` | Regional multiple-testing level |
| `pc_levels` | `c(0.1, 0.2, 0.3)` | Partial-conjunction proportions tested in every atlas region |
| `denoise` | `"none"` | No denoising, or `"wavelet"` when the optional `waveslim` package is installed |
| `bandwidth` | `5` | CO spatial bandwidth in voxels |
| `neighbor_range` | `10` | CO neighbourhood radius in voxels |
| `lambda` | `0.5` | CO sparsity tuning parameter |
| `score_clip` | `0.99` | CO score clipping constant |
| `simes` | `FALSE` | Whether to add BHe+Simes |
| `seed` | `1` | Seed shared by mirror generation and working-model fitting |
| `out_dir` | `NULL` | Return results only; when set, also write tables, maps, RDS and PDF output |
| `keep_scores` | `FALSE` | Retain voxelwise working-model scores in the returned object |

The CO tuning arguments do not alter Ising or the optional working models, which retain their model-specific fitting settings.

### Returned result and files

`seft()` returns an object of class `seft_result` with the following components:

| Component | Contents |
|---|---|
| `regions` | One row per atlas region, method and PC level |
| `summary` | Number of tested and selected regions for every method and PC level |
| `models` | Model identifier and fitting metadata; voxelwise scores are included when `keep_scores = TRUE` |
| `metadata` | Statistic type, residual df, image dimensions, affine, mask size, region count and resolved parameters |
| `output_files` | Paths written under `out_dir`, or `NULL` when no output directory was requested |

The main `result$regions` columns are:

| Column | Meaning |
|---|---|
| `pc_level` | Requested partial-conjunction proportion |
| `method` | Display name such as `SEFT-CO`, `SEFT-Ising` or `BHe` |
| `working_model` | Working-model key used by `seft()` |
| `region_id`, `region_label` | Atlas identifier and resolved label |
| `n_voxels` | Number of analyzed voxels in the region |
| `u` | Partial-conjunction order, `ceiling(n_voxels * pc_level)` |
| `e_value` | Regional e-value |
| `pc_p_value` | Regional partial-conjunction p-value or reciprocal e-value used for reporting |
| `significant` | `1` when selected at `alpha`, otherwise `0` |

For example:

```r
print(result)
head(result$regions)
result$summary
result$metadata[c("dimensions", "n_regions", "working_models", "seed")]
```

When `out_dir` is supplied, SEFT writes regional and significant-region TSV files, a summary TSV, the signed Z map, one regional discovery NIfTI per method and PC level, the complete result as RDS, and an available-methods PDF. Only methods actually present in `result` are displayed.

```text
seft_output/
├── seft_result.rds
├── tables/
│   ├── seft_region_results.tsv
│   ├── seft_significant_regions.tsv
│   └── seft_summary.tsv
├── maps/
│   ├── seft_signed_z.nii.gz
│   └── seft_pc*_regions.nii.gz
└── figures/
    └── seft_available_methods.pdf
```

The discovery NIfTI files preserve the atlas identifier for selected regions and contain 0 elsewhere. Change the filename stem with `prefix`, for example `prefix = "contrast1"`. Existing files with the same names are replaced.

For example, the built-in comparison is:

```r
result <- seft(
  zmap = "contrast_signed_z.nii.gz",
  atlas = "AAL3v1.nii.gz",
  working_model = c("co", "ising"),
  simes = TRUE,
  denoise = "wavelet",
  seed = 1
)
plot(result, file = "seft_available_methods.pdf")
```

### Plotting regional results

`plot(result)` displays one row for each method present in the result and one column for each atlas region. A coloured cell gives the highest tested PC level at which that region was selected; light grey means that the region was not selected. Regions are ordered by atlas identifier, and recognized AAL3 identifiers receive anatomical group headings. Methods that were not run are not shown.

Draw on the current graphics device:

```r
plot(result)
```

Write a PDF and set its title:

```r
plot(
  result,
  file = "seft_available_methods.pdf",
  main = "Regional findings"
)
```

An optional TFCE coverage row can be added with a data frame containing `region_id` and `tfce_coverage`. Coverage must be between 0 and 1; missing atlas regions are displayed as zero coverage.

```r
tfce <- data.frame(
  region_id = c(1L, 2L),
  tfce_coverage = c(0.42, 0.78)
)
plot(result, file = "seft_with_tfce.pdf", tfce_coverage = tfce)
```

`plot()` returns the original result invisibly, so it can be used inside a pipeline without changing the stored inference result.

## Optional working models

The base package does not install Python, PyTorch or external repositories. FDR smoothing, DeepFDR and fcHMRF additionally require the R package `jsonlite` and their optional Python environment. Install the pinned environment only when one of these models is needed:

```bash
Rscript -e 'install.packages("jsonlite", repos="https://cloud.r-project.org")'
SEFT_SETUP="$(Rscript -e 'cat(system.file("optional", "setup_working_models.sh", package = "SEFT"))')"
bash "$SEFT_SETUP"
```

The setup command installs all three optional backends by default on Linux. To install only selected backends, set `SEFT_OPTIONAL_MODELS`, for example `SEFT_OPTIONAL_MODELS=deepfdr bash "$SEFT_SETUP"` or `SEFT_OPTIONAL_MODELS=fdr-smoothing,fchmrf bash "$SEFT_SETUP"`. This initial setup normally requires internet access for conda packages and the pinned external repositories; the configured models can subsequently run offline. After setup, use `working_model = "fdr-smoothing"`, `"deepfdr"` or `"fchmrf"` in the same `seft()` call. DeepFDR uses a visible GPU when available and falls back to CPU. If an optional method has not been installed, `seft()` stops before fitting and prints the setup command.

`bash "$SEFT_SETUP" --check-paths` verifies the bundled resources and prints the install paths without downloading anything.

## Public demo

The package includes the public signed Z map from NeuroVault image 790848, resampled to the bundled AAL3 grid. Its source, license, checksums and preparation are documented in `inst/extdata/neurovault_790848/PROVENANCE.md`.

A pre-generated comparison figure is installed at `system.file("extdata", "neurovault_790848", "seft_available_methods.pdf", package = "SEFT")`. It contains only SEFT-CO, SEFT-Ising and BHe, the methods used in the bundled demo run.

```r
library(SEFT)

demo <- system.file("extdata", "neurovault_790848", package = "SEFT")
result <- seft(
  zmap = file.path(demo, "speech_vs_reversed_zmap_aal3_2mm.nii.gz"),
  atlas = file.path(demo, "AAL3v1.nii.gz"),
  atlas_labels = file.path(demo, "AAL3v1.nii.txt"),
  working_model = "co",
  denoise = "wavelet",
  seed = 1
)
print(result)
plot(result, file = "seft_demo.pdf")
```

To compare the built-in methods, use `working_model = c("co", "ising")` and `simes = TRUE`. Optional methods can be added to the same vector after their environment is installed. The resulting plot contains only methods that were actually run.

## Command line

Locate the installed script once and invoke it like any Rscript executable:

```bash
SEFT_BIN="$(Rscript -e 'cat(system.file("scripts", "seft", package = "SEFT"))')"
"$SEFT_BIN" --zmap contrast_signed_z.nii.gz --atlas AAL3v1.nii.gz --working-model co --denoise --out-dir seft_output
```

Wavelet denoising is disabled unless `--denoise` is present. Use `--working-model co,ising --simes` for the built-in comparison or add any installed optional model to the comma-separated list. Run `"$SEFT_BIN" --help` for the optional inference parameters.

## Tests

Build and check the source package with `R CMD build pkg/SEFT` followed by `R CMD check SEFT_0.0.0.9000.tar.gz`. The complete public-demo integration test is `Rscript pkg/SEFT/tests/integration/run-neurovault-demo.R`; it runs CO, Ising, BHe, NIfTI output and the regional PDF and therefore takes longer than the package check.
