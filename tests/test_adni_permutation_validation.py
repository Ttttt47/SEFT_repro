from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "real_data/adni/stages/validate_results.py"
spec = importlib.util.spec_from_file_location("validate_results", SCRIPT)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def test_permutation_accounting_accepts_nondefault_request() -> None:
    row = {
        "requested_permutations": 1200,
        "fragments_per_contrast": 12,
        "permutations_per_fragment": 100,
        "rounded_permutations_per_contrast": 1200,
        "effective_permutation_denominator": 1189,
    }
    assert module.permutation_accounting_errors(row, 1200) == []
    assert "primary analysis used 5000" in module.permutation_accounting_errors(row, 5000)[0]


def test_permutation_accounting_rejects_bad_fragment_denominator() -> None:
    row = {
        "requested_permutations": 1200,
        "fragments_per_contrast": 12,
        "permutations_per_fragment": 100,
        "rounded_permutations_per_contrast": 1200,
        "effective_permutation_denominator": 1190,
    }
    assert "effective permutation denominator" in module.permutation_accounting_errors(row, 1200)[0]
    row["rounded_permutations_per_contrast"] = 1199
    assert any("fragment grid" in message for message in module.permutation_accounting_errors(row, 1200))
