#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PACKAGE_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
OPTIONAL_ROOT=${SEFT_OPTIONAL_ROOT:-$(Rscript --vanilla -e 'cat(file.path(tools::R_user_dir("SEFT", which = "data"), "working-models"))')}
ENVIRONMENT_PATH=${SEFT_ML_ENV:-${OPTIONAL_ROOT}/env}
VENDOR_ROOT=${SEFT_ML_VENDOR_ROOT:-${OPTIONAL_ROOT}/vendor}
PYTHON_ROOT=${OPTIONAL_ROOT}/python
MODELS=${SEFT_OPTIONAL_MODELS:-fdr-smoothing,deepfdr,fchmrf}
for asset in optional/src/absmax_backend.cpp optional/src/persistent_gfl_backend.cpp optional/python/ml_working_model.py; do
    [[ -f ${PACKAGE_ROOT}/${asset} ]] || { echo "Missing packaged optional-model asset: ${asset}" >&2; exit 2; }
done
if [[ ${1:-} == --check-paths ]]; then
    printf '%s\n%s\n' "${PACKAGE_ROOT}" "${OPTIONAL_ROOT}"
    exit 0
fi
MAMBA_BIN=${MAMBA_BIN:-$(command -v micromamba || command -v mamba || true)}
[[ -n ${MAMBA_BIN} ]] || { echo "Set MAMBA_BIN to micromamba or mamba." >&2; exit 2; }

has_model() {
    [[ ,${MODELS}, == *,$1,* ]]
}

for model in ${MODELS//,/ }; do
    case ${model} in
        fdr-smoothing|deepfdr|fchmrf) ;;
        *) echo "Unknown model in SEFT_OPTIONAL_MODELS: ${model}" >&2; exit 2 ;;
    esac
done

mkdir -p "${OPTIONAL_ROOT}" "${VENDOR_ROOT}" "${PYTHON_ROOT}"
"${MAMBA_BIN}" create -y -p "${ENVIRONMENT_PATH}" -c conda-forge python=3.11 pip numpy scipy pandas nibabel nilearn scikit-image matplotlib tqdm pybind11 ninja cmake gsl c-compiler cxx-compiler make

PYTHON=${ENVIRONMENT_PATH}/bin/python
if has_model deepfdr || has_model fchmrf; then
    "${PYTHON}" -m pip install torch==2.5.1 torchinfo --index-url https://download.pytorch.org/whl/cu121
fi

clone_pin() {
    local url=$1 name=$2 revision=$3
    if [[ ! -d ${VENDOR_ROOT}/${name}/.git ]]; then
        git clone "${url}" "${VENDOR_ROOT}/${name}"
    fi
    git -C "${VENDOR_ROOT}/${name}" fetch --all --tags
    git -C "${VENDOR_ROOT}/${name}" checkout --detach "${revision}"
}

if has_model fdr-smoothing; then
    clone_pin https://github.com/tansey/smoothfdr.git smoothfdr c5b693d0a66e83c9387433b33c0eab481bd4a763
    clone_pin https://github.com/tansey/gfl.git gfl 30046ea13e63e2d28000e8b8fb4d2e7c92302a8c
    "${PYTHON}" -m pip install -e "${VENDOR_ROOT}/gfl"
    "${PYTHON}" -m pip install -e "${VENDOR_ROOT}/smoothfdr" --no-deps
fi
if has_model deepfdr; then
    clone_pin https://github.com/kimtae55/DeepFDR.git DeepFDR 44294ac4742f1e2b1634158cfb869353c3708ea9
fi
if has_model fchmrf; then
    clone_pin https://github.com/kimtae55/fcHMRF-LIS.git fcHMRF-LIS f32e8550d40a83448525d811376f942873ec994a
fi

if has_model fchmrf; then
    pushd "${VENDOR_ROOT}/fcHMRF-LIS/src" >/dev/null
    env MAX_JOBS=${MAX_JOBS:-8} "${PYTHON}" setup.py build_ext --inplace
    popd >/dev/null
fi

if has_model fdr-smoothing; then
    cp "${PACKAGE_ROOT}/optional/python/fdr_smoothing_absmax.py" "${PYTHON_ROOT}/"
fi
if has_model deepfdr || has_model fchmrf; then
    cp "${PACKAGE_ROOT}/optional/python/absmax_models.py" "${PYTHON_ROOT}/"
    cp "${PACKAGE_ROOT}/optional/python/ml_working_model.py" "${PYTHON_ROOT}/"
fi

SUFFIX=$("${PYTHON}" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')
read -r -a INCLUDES <<< "$("${PYTHON}" -m pybind11 --includes)"
CXX_BIN=$(find -L "${ENVIRONMENT_PATH}/bin" -maxdepth 1 -name '*-c++' -type f -print -quit)
CC_BIN=$(find -L "${ENVIRONMENT_PATH}/bin" -maxdepth 1 -name '*-cc' -type f -print -quit)
[[ -x ${CXX_BIN} && -x ${CC_BIN} ]] || { echo "The optional environment does not contain C/C++ compilers." >&2; exit 2; }
"${CXX_BIN}" -O3 -DNDEBUG -std=c++17 -shared -fPIC "${INCLUDES[@]}" "${PACKAGE_ROOT}/optional/src/absmax_backend.cpp" -o "${PYTHON_ROOT}/absmax_backend${SUFFIX}"

if has_model fdr-smoothing; then
    BUILD_DIR=$(mktemp -d)
    trap 'rm -rf "${BUILD_DIR}"' EXIT
    "${CC_BIN}" -O3 -DNDEBUG -fPIC -fopenmp -I"${VENDOR_ROOT}/gfl/cpp/include" -I"${ENVIRONMENT_PATH}/include" -c "${VENDOR_ROOT}/gfl/cpp/src/graph_fl.c" -o "${BUILD_DIR}/graph_fl.o"
    "${CC_BIN}" -O3 -DNDEBUG -fPIC -fopenmp -I"${VENDOR_ROOT}/gfl/cpp/include" -I"${ENVIRONMENT_PATH}/include" -c "${VENDOR_ROOT}/gfl/cpp/src/tf_dp.c" -o "${BUILD_DIR}/tf_dp.o"
    "${CXX_BIN}" -O3 -DNDEBUG -std=c++17 -shared -fPIC -fopenmp "${INCLUDES[@]}" -I"${VENDOR_ROOT}/gfl/cpp/include" -I"${ENVIRONMENT_PATH}/include" "${PACKAGE_ROOT}/optional/src/persistent_gfl_backend.cpp" "${BUILD_DIR}/graph_fl.o" "${BUILD_DIR}/tf_dp.o" -o "${PYTHON_ROOT}/persistent_gfl_backend${SUFFIX}"
fi

printf 'Optional SEFT working models are ready.\nSEFT_OPTIONAL_ROOT=%s\nSEFT_ML_PYTHON=%s\nSEFT_ML_VENDOR_ROOT=%s\n' "${OPTIONAL_ROOT}" "${PYTHON}" "${VENDOR_ROOT}"
