# SEFT reproducibility code

This repository accompanies *Structure-Adaptive E-Value Filter for Detecting Regional Signals in Brain Imaging*. It reproduces the ADNI3 application and the simulations; downloaded data and generated results are not included.

## Paper-to-code guide

| Manuscript component | Reproduction entry point |
|---|---|
| Section 3.1: ADNI sample and FSL-VBM | `real_data/adni/run_pipeline.py` |
| Sections 3.2–3.7: diagnostics, SEFT, comparisons, sensitivity and stability | `real_data/adni/stages/` and `real_data/adni/figures/` |
| Section 4 and Appendix B: SEFT and working-model constructions | `R/core/`, `R/working_models/`, `src/working_models/` |
| Appendix E.1: two-dimensional simulations | `simulations/sim_2d/` |
| Appendix E.2: IID/GRF three-dimensional simulations | `simulations/sim_3d/` |
| Appendix E.3: Ising data-generating model | `simulations/sim_ising/run_ising_parallel.R` |

## Software

The reference environment uses R 4.3, Python 3.11 and FSL 6.0.7.22.

```bash
micromamba create -f environment.yml
micromamba activate seft-repro
export FSLDIR=/path/to/fsl
bash scripts/setup_working_models.sh
export SEFT_ML_PYTHON="$PWD/.envs/working-models/bin/python"
export SEFT_ML_VENDOR_ROOT="$PWD/.vendor"
```

The maintained workflow uses the following R packages: `Rcpp`, `RcppArmadillo`, `data.table`, `ggplot2`, `cowplot`, `glue`, `jsonlite`, `RNifti`, `oro.nifti`, `waveslim`, `optparse`, `MASS`, `digest`, and `neuRosim`. Legacy figure scripts additionally use `ComplexHeatmap`, `circlize`, `plot3D`, and `ciftiTools`. Python requirements are recorded in `environment.yml`.

The working-model setup pins smoothfdr 0.9.5 (`c5b693d`), DeepFDR (`44294ac`), and fcHMRF-LIS (`f32e855`). DeepFDR uses one worker per visible GPU and falls back to one CPU worker; other methods default to 64 CPU workers.

## Run SEFT directly from a Z-map

The ADNI pipeline is not required when a signed three-dimensional Z-map is already available. The Z-map, integer-valued atlas and optional mask must have the same NIfTI shape and affine. The shortest call is:

```bash
python tools/seft_fsl/seft_fsl_run \
  --zmap /path/to/contrast_signed_z.nii.gz \
  --atlas real_data/input/atlas/AAL3v1.nii.gz \
  --working-model co \
  --denoise \
  --out-dir outputs/seft_example \
  --prefix example
```

Key options:

| Option | Choices and default |
|---|---|
| Statistic input | Use exactly one of `--zmap`; or `--tstat` together with `--design-mat`. |
| Atlas inputs | `--atlas` is required; `--atlas-labels` and `--mask` are optional. |
| `--working-model` | `co` (default), `fdr-smoothing`, `deepfdr`, `fchmrf`, or `ising`; `covariate-adaptive` and `fdrs` are accepted aliases. |
| Testing | `--alpha 0.1`; `--pc-levels 0.01,0.05,0.1,0.2,0.3,0.4,0.5`; add `--simes` for the BH+Simes comparison. |
| Preprocessing | Add `--denoise` to apply wavelet denoising; omitted by default. |
| CO score | `--bandwidth 5`, `--neighbor-range 10`, `--lambda 0.5`, and `--score-clip-c 0.99`; the other models retain their pinned implementation defaults. |
| Reproducibility and diagnostics | `--seed 1`; add `--save-internals` to retain score maps. |

All five choices use the same regional-testing and output interface. FDR smoothing, DeepFDR and fcHMRF require `scripts/setup_working_models.sh`; DeepFDR uses a visible GPU when available. The command writes maps under `outputs/seft_example/maps/`, region-level tables and model-fit metadata under `outputs/seft_example/tables/`, and the resolved inputs and parameters to `outputs/seft_example/run_metadata.json`.

### A public Z-map demo

For a small end-to-end test that requires no private imaging data, the repository includes an unthresholded, group-level signed Z-map from [NeuroVault image 790848](https://neurovault.org/images/790848/), listening to speech versus reversed speech. The original public 4 mm image and a reproducibly resampled AAL3-compatible 2 mm image are stored in `examples/neurovault_790848/`; source, CC0 licensing, checksums and processing details are recorded in its [provenance note](examples/neurovault_790848/PROVENANCE.md).

From the repository root, run the same SEFT command used for any signed Z-map:

```bash
python tools/seft_fsl/seft_fsl_run \
  --zmap examples/neurovault_790848/speech_vs_reversed_zmap_aal3_2mm.nii.gz \
  --atlas real_data/input/atlas/AAL3v1.nii.gz \
  --working-model co \
  --denoise \
  --out-dir outputs/seft_example \
  --prefix example
```

On completion, `outputs/seft_example/tables/example_summary.tsv` reports the number of regions selected at each PC level, while `outputs/seft_example/maps/` contains the signed Z-map and regional discovery maps. To reproduce the grid conversion from the bundled NeuroVault source image, run `python examples/neurovault_790848/prepare_demo.py`.

### Local R package

The compact package in `pkg/SEFT` provides the same regional inference through one R function without requiring the full reproduction environment. It requires R 4.1 or later, a C++17 compiler, `Rcpp`, `RcppArmadillo` and `RNifti`; `R CMD INSTALL` lists any missing packages but does not download them. CO and Ising are compiled into the package, while wavelet denoising and the external working models use optional dependencies described in the package README. Install it with `R CMD INSTALL --clean pkg/SEFT`, then run the bundled public demo as follows:

```r
library(SEFT)
demo <- system.file("extdata", "neurovault_790848", package = "SEFT")
result <- seft(
  zmap = file.path(demo, "speech_vs_reversed_zmap_aal3_2mm.nii.gz"),
  atlas = file.path(demo, "AAL3v1.nii.gz"),
  working_model = "co",
  denoise = "wavelet",
  out_dir = "outputs/seft_r_demo"
)
print(result)
```

Use `working_model = c("co", "ising")` and `simes = TRUE` to compare the built-in methods. See [`pkg/SEFT/README.md`](pkg/SEFT/README.md) for t-statistic input, optional working models and command-line use.

## ADNI3 application

ADNI data require approval from the [ADNI data portal](https://adni.loni.usc.edu/). Download the full ADNI3 3T T1-weighted scan pool and metadata under the [ADNI Data Use Agreement](https://adni.loni.usc.edu/wp-content/themes/adni_2023/documents/ADNI_Data_Use_Agreement.pdf). The pipeline reads participant and Image IDs from the user's authorized download rather than a repository-provided list; an incomplete scan download can change the matched sample.

Prepare this layout:

```text
ADNI_DATA/
├── downloads/*.zip
├── All_Subjects_Key_MRI*.csv
├── All_Subjects_DXSUM*.csv
├── All_Subjects_PTDEMOG*.csv
└── All_Subjects_Study_Entry*.csv
```

ADNI appends a download-date or version suffix to these CSV filenames. The workflow matches the stable prefixes shown above, so no renaming is needed; keep exactly one matching file for each prefix in `ADNI_DATA`. The four tables provide scan metadata, visit-level diagnosis, sex/education, and study-entry age/date, respectively. Run:

```bash
python real_data/adni/run_pipeline.py \
  --data-dir /path/to/ADNI_DATA \
  --work-dir outputs/adni3 \
  --fsldir "$FSLDIR" \
  --seft-env "$CONDA_PREFIX" \
  --workers 64
```

The restartable stages are:

```text
index → cohort → convert → vbm → primary → working-models
      → stability → figures → validate
```

`cohort` selects the earliest qualifying ADNI3 3T T1 visit with a same-visit CN/MCI/dementia diagnosis and complete age, sex and education, preferring standard acquisitions. Dementia participants anchor one-to-one CN/MCI triplets matched exactly on sex and scanner/model/protocol family, then by age and education distance. The selection audit is written to `outputs/adni3/tables/` and `outputs/adni3/provenance/`.

The primary analysis uses sigma=3 mm FSL-VBM maps, a covariate-adjusted three-group GLM, 5,000 requested TFCE permutations, AAL3, wavelet denoising, bandwidth 5 voxels, neighborhood 10, lambda 0.5, alpha 0.1, and PC level `c=0.20`. PC levels 0.10/0.30, alpha 0.05, and Harvard–Oxford are sensitivity analyses. The weaker CN–MCI result at `c=0.10` is exploratory.

### Atlases

The repository includes AAL3v1 from the SPM12 distribution of the Automated Anatomical Labeling 3 atlas:

- `real_data/input/atlas/AAL3v1.nii.gz`: integer atlas image;
- `real_data/input/atlas/AAL3v1.nii.txt`: region-code-to-name dictionary.

The Harvard–Oxford 25% 2-mm sensitivity atlas is generated automatically from the cortical and subcortical atlases distributed with FSL. Its merged image, labels and provenance are written below `outputs/adni3/atlas/`.

## Simulations

IID/GRF experiment (Appendix E.2):

```bash
python simulations/sim_3d/run_all.py \
  --workers 64 --gpu-workers 0 --output-dir outputs/sim_3d
```

Use `--smoke` for a small resumable check. The formal grid uses `L=64`, `mu=2:0.5:5`, 10/20/30 clusters and 100 repetitions.

Ising experiment (Appendix E.3):

```bash
Rscript simulations/sim_ising/validate_ising_reweight.R
Rscript simulations/sim_ising/run_ising_parallel.R \
  --workers 64 --output-dir outputs/sim_ising
Rscript simulations/sim_ising/make_figures_ising.R \
  outputs/sim_ising outputs/sim_ising/figures
```

The older two-dimensional and no-denoise workflows remain in their existing simulation directories.

## Checks

Routine acceptance does not rerun the full VBM, TFCE or 100-repetition grids:

```bash
python -m pytest tests/test_adni_archive_index.py \
  tests/test_adni_workflow_guards.py tests/test_fsl_wrapper.py \
  tests/test_randomise_fragment_merge.py tests/test_neurovault_demo.py
Rscript simulations/sim_ising/validate_ising_reweight.R
Rscript tests/test_working_model_cpp_equivalence.R
Rscript tests/test_absmax_score_exchangeability.R
Rscript tests/test_working_model_dispatch.R
```

## Historical MDD example

The MDD analysis is retained only as a historical example: `Rscript real_data/real_data_analysis_pipeline.R run_mdd_paper`.

## License

SEFT code is [MIT-licensed](LICENSE). The bundled AAL3 atlas and labels (also included in the R package) are third-party material under the GNU GPL; see the [AAL3 publisher](https://www.gin.cnrs.fr/en/tools/aal/). The NeuroVault demo map is [CC0](pkg/SEFT/inst/extdata/neurovault_790848/PROVENANCE.md).
