# Bundled NeuroVault Z-map demo

The bundled `AAL3v1.nii.gz` atlas and `AAL3v1.nii.txt` labels are from the [AAL3 publisher](https://www.gin.cnrs.fr/en/tools/aal/) and are under the GNU GPL, separately from the NeuroVault map's CC0 license and the package code's MIT license.

The source image is NeuroVault image [790848](https://neurovault.org/images/790848/), with persistent identifier [neurovault.image:790848](https://identifiers.org/neurovault.image:790848). NeuroVault describes it as an unthresholded, group-level fMRI-BOLD Z-map in MNI space for listening to speech versus reversed speech. It belongs to NeuroVault collection [13137](https://neurovault.org/collections/13137/), “Group activation maps for an fMRI language localizer (listening to speech vs reversed speech).”

NeuroVault makes its public data available under [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/). The source file was downloaded from `https://neurovault.org/media/images/13137/speechRev_Zmap.nii.gz` and is redistributed here as `source_speechRev_Zmap_4mm.nii.gz`. For NeuroVault itself, cite Gorgolewski et al. (2015), [doi:10.3389/fninf.2015.00008](https://doi.org/10.3389/fninf.2015.00008).

Source SHA-256:

```text
f9121c3414d2ecb5f0ec73749c6552ca19fd86bffa2b63e13400eae0b6727477
```

The ready-to-run file `speech_vs_reversed_zmap_aal3_2mm.nii.gz` is derived deterministically by `prepare_demo.py`: the signed Z-map is linearly resampled from its original 4 mm MNI grid to the repository's 2 mm AAL3 grid, stored as float32, non-finite values are replaced by zero, and voxels outside the AAL3 support are set to zero. The values are not thresholded or otherwise transformed. Its SHA-256 is `ae7cb8f1585209f9707ade0ab31138d62b665802d98f15dae67509f529514bfe`. This small public map is provided only to exercise the SEFT interface; it is not part of the paper's ADNI analysis.
