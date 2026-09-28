from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "real_data/adni/fsl/randomise_parallel_limited.sh"


def executable(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


def test_fragment_merge_accounting_and_cleanup(tmp_path: Path) -> None:
    """Exercise the real wrapper with a tiny fake FSL command surface."""
    fsl_bin = tmp_path / "fsl/bin"
    fsl_bin.mkdir(parents=True)
    executable(
        fsl_bin / "randomise",
        """#!/usr/bin/env bash
set -euo pipefail
root=''
query=0
seed=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    -o) root=$2; shift 2 ;;
    -Q) query=1; shift ;;
    --seed=*) seed=${1#--seed=}; shift ;;
    *) shift ;;
  esac
done
if [[ $query -eq 1 ]]; then
  printf '5000 1 %s 100\n' "$root"
  exit 0
fi
printf 'fragment %s\n' "$seed" > "${root}_tfce_corrp_tstat1.nii.gz"
printf 'raw %s\n' "$seed" > "${root}_tstat1.nii.gz"
""",
    )
    executable(
        fsl_bin / "fsl_sub",
        """#!/usr/bin/env bash
set -euo pipefail
task=''
last=''
while [[ $# -gt 0 ]]; do
  last=$1
  if [[ $1 == -t ]]; then task=$2; shift 2; continue; fi
  case "$1" in -T|-N|-l|-x|-j) shift 2 ;; *) shift ;; esac
done
if [[ -n $task ]]; then bash "$task" >/dev/null; else bash "$last" >/dev/null; fi
printf 'FAKE_JOB_ID\n'
""",
    )
    executable(
        fsl_bin / "imglob",
        """#!/usr/bin/env bash
set -euo pipefail
for pattern in "$@"; do
  [[ $pattern == -* ]] && continue
  for path in $pattern; do [[ -e $path ]] && printf '%s ' "$path"; done
done
printf '\n'
""",
    )
    executable(
        fsl_bin / "fslmaths",
        """#!/usr/bin/env bash
set -euo pipefail
for last in "$@"; do :; done
printf 'merged\n' > "$last"
""",
    )
    executable(fsl_bin / "sleep", "#!/usr/bin/env bash\nexit 0\n")

    output_root = tmp_path / "result/three_group_sigma3"
    output_root.parent.mkdir()
    env = os.environ.copy()
    env["FSLDIR"] = str(tmp_path / "fsl")
    env["PATH"] = f"{fsl_bin}:{env.get('PATH', '')}"
    subprocess.run(
        [
            "bash", str(WRAPPER), "--threads", "2", "--requested-time", "1", "--",
            "-i", str(tmp_path / "input.nii.gz"), "-o", str(output_root),
            "-d", str(tmp_path / "design.mat"), "-t", str(tmp_path / "design.con"),
            "-n", "5000", "-T",
        ],
        check=True,
        env=env,
        capture_output=True,
        text=True,
    )

    metadata = json.loads(Path(f"{output_root}_permutation_metadata.json").read_text())
    assert metadata == {
        "requested_permutations": 5000,
        "fragments_per_contrast": 50,
        "permutations_per_fragment": 100,
        "rounded_permutations_per_contrast": 5000,
        "effective_permutation_denominator": 4951,
    }
    assert Path(f"{output_root}_tfce_corrp_tstat1.nii.gz").is_file()
    assert Path(f"{output_root}_tstat1.nii.gz").is_file()
    assert not list(output_root.parent.glob(f"{output_root.name}_SEED*_tfce_corrp_tstat1.nii.gz"))
    assert Path(f"{output_root}_logs").is_dir()
    assert Path(f"{output_root}_randomise.status").read_text().strip() == "COMPLETE"
    assert Path(f"{output_root}_submission.log").is_file()
