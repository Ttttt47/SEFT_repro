#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPRO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
ENVIRONMENT_PATH=${SEFT_ML_ENV:-${REPRO_ROOT}/.envs/working-models}
VENDOR_ROOT=${SEFT_ML_VENDOR_ROOT:-${REPRO_ROOT}/.vendor}
MAMBA_BIN=${MAMBA_BIN:-$(command -v micromamba || command -v mamba || true)}
[[ -n ${MAMBA_BIN} ]] || { echo "Set MAMBA_BIN to micromamba or mamba" >&2; exit 2; }

"${MAMBA_BIN}" create -y -p "${ENVIRONMENT_PATH}" -c conda-forge \
    python=3.11 pip numpy scipy pandas nibabel nilearn scikit-image matplotlib tqdm \
    pybind11 ninja cmake gsl r-base=4.3 r-rcpp r-rcpparmadillo r-data.table \
    r-ggplot2 r-cowplot r-glue r-rnifti r-waveslim r-jsonlite

PYTHON=${ENVIRONMENT_PATH}/bin/python
"${PYTHON}" -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
"${PYTHON}" -m pip install torchinfo
mkdir -p "${VENDOR_ROOT}"

clone_pin() {
    local url=$1 name=$2 revision=$3
    if [[ ! -d ${VENDOR_ROOT}/${name}/.git ]]; then
        git clone "${url}" "${VENDOR_ROOT}/${name}"
    fi
    git -C "${VENDOR_ROOT}/${name}" fetch --all --tags
    git -C "${VENDOR_ROOT}/${name}" checkout --detach "${revision}"
}

clone_pin https://github.com/tansey/smoothfdr.git smoothfdr c5b693d0a66e83c9387433b33c0eab481bd4a763
clone_pin https://github.com/tansey/gfl.git gfl 30046ea13e63e2d28000e8b8fb4d2e7c92302a8c
clone_pin https://github.com/kimtae55/DeepFDR.git DeepFDR 44294ac4742f1e2b1634158cfb869353c3708ea9
clone_pin https://github.com/kimtae55/fcHMRF-LIS.git fcHMRF-LIS f32e8550d40a83448525d811376f942873ec994a
"${PYTHON}" -m pip install -e "${VENDOR_ROOT}/gfl"
"${PYTHON}" -m pip install -e "${VENDOR_ROOT}/smoothfdr" --no-deps

pushd "${VENDOR_ROOT}/fcHMRF-LIS/src" >/dev/null
env MAX_JOBS=${MAX_JOBS:-8} "${PYTHON}" setup.py build_ext --inplace
popd >/dev/null

suffix=$("${PYTHON}" - <<'PY'
import sysconfig
print(sysconfig.get_config_var("EXT_SUFFIX"))
PY
)
read -r -a includes <<< "$("${PYTHON}" -m pybind11 --includes)"
${CXX:-g++} -O3 -DNDEBUG -std=c++17 -shared -fPIC "${includes[@]}" \
    "${REPRO_ROOT}/src/working_models/absmax_backend.cpp" \
    -o "${REPRO_ROOT}/python/working_models/absmax_backend${suffix}"

BUILD_DIR=$(mktemp -d)
trap 'rm -rf "${BUILD_DIR}"' EXIT
${CC:-gcc} -O3 -DNDEBUG -fPIC -fopenmp -I"${VENDOR_ROOT}/gfl/cpp/include" \
    -I"${ENVIRONMENT_PATH}/include" -c "${VENDOR_ROOT}/gfl/cpp/src/graph_fl.c" -o "${BUILD_DIR}/graph_fl.o"
${CC:-gcc} -O3 -DNDEBUG -fPIC -fopenmp -I"${VENDOR_ROOT}/gfl/cpp/include" \
    -I"${ENVIRONMENT_PATH}/include" -c "${VENDOR_ROOT}/gfl/cpp/src/tf_dp.c" -o "${BUILD_DIR}/tf_dp.o"
${CXX:-g++} -O3 -DNDEBUG -std=c++17 -shared -fPIC -fopenmp "${includes[@]}" \
    -I"${VENDOR_ROOT}/gfl/cpp/include" -I"${ENVIRONMENT_PATH}/include" \
    "${REPRO_ROOT}/src/working_models/persistent_gfl_backend.cpp" \
    "${BUILD_DIR}/graph_fl.o" "${BUILD_DIR}/tf_dp.o" \
    -o "${REPRO_ROOT}/python/working_models/persistent_gfl_backend${suffix}"

cat <<EOF
Environment ready.
export SEFT_ML_ENV=${ENVIRONMENT_PATH}
export SEFT_ML_PYTHON=${PYTHON}
export SEFT_ML_VENDOR_ROOT=${VENDOR_ROOT}
EOF
