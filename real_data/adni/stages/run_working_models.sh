#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPRO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
WORK="${REPRO_ROOT}/outputs/adni3"
WORKERS=64
GPU_WORKERS=0
RSCRIPT=${RSCRIPT:-Rscript}
PYTHON=${SEFT_ML_PYTHON:-python3}
SEED=20260723
GPU_IDS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --work-dir) WORK=$2; shift 2 ;;
        --workers) WORKERS=$2; shift 2 ;;
        --gpu-workers) GPU_WORKERS=$2; shift 2 ;;
        --rscript) RSCRIPT=$2; shift 2 ;;
        --python) PYTHON=$2; shift 2 ;;
        --seed) SEED=$2; shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done

for value in "${WORKERS}" "${GPU_WORKERS}"; do
    [[ ${value} =~ ^[0-9]+$ ]] || { echo "Worker counts must be nonnegative integers" >&2; exit 2; }
done
[[ ${WORKERS} -ge 1 ]] || { echo "--workers must be positive" >&2; exit 2; }

OUTPUT="${WORK}/results/absmax_working_models_real_seed20260723"
RUNNER="${SCRIPT_DIR}/run_working_model_task.R"
RECOMPUTE="${SCRIPT_DIR}/recompute_working_model_pc.R"
mkdir -p "${OUTPUT}"/{logs,status,tables,maps,scores,figures}
printf 'RUNNING\n' > "${OUTPUT}/RUN_STATE"
{
    echo "started_at=$(date -Is)"
    echo "seed=${SEED}"
    echo "contrasts=CN_gt_Dementia,MCI_gt_Dementia,CN_gt_MCI"
    echo "methods=fdrs_absmax,deepfdr_absmax,fchmrf_absmax,ising_absmax"
    echo "pc_levels=0.01,0.05,0.1,0.2,0.3,0.4,0.5"
    echo "regional_alpha=0.1"
    echo "mirror=iid_N(0,1); padding=exchangeable_iid_N(0,1)_pair"
} > "${OUTPUT}/run_config.txt"

run_one() {
    local method=$1 contrast=$2 gpu=${3:-}
    local task="${method}__${contrast}"
    local status="${OUTPUT}/status/${task}.status"
    local table="${OUTPUT}/tables/${task}_regions.tsv"
    if [[ -s ${table} && -s ${status} && $(head -n 1 "${status}") == COMPLETE ]]; then
        return 0
    fi
    printf 'RUNNING\n' > "${status}"
    local extra=()
    if [[ ${method} == ising_absmax ]]; then
        extra=(--ising-iter-max 200 --ising-sweep-b 20 --ising-sweep-r 40 \
               --ising-burnin-lis 100 --ising-sweep-lis 250 --ising-n-chains 2)
    fi
    local command=("${RSCRIPT}" "${RUNNER}" --method "${method}" --contrast "${contrast}" \
                   --seed "${SEED}" --alpha 0.1 --pc-levels 0.01,0.05,0.1,0.2,0.3,0.4,0.5 \
                   --work-dir "${WORK}" --output-dir "${OUTPUT}" "${extra[@]}")
    if [[ -n ${gpu} ]]; then
        CUDA_VISIBLE_DEVICES=${gpu} SEFT_ML_PYTHON=${PYTHON} "${command[@]}" \
            > "${OUTPUT}/logs/${task}.log" 2>&1
    else
        SEFT_ML_PYTHON=${PYTHON} "${command[@]}" > "${OUTPUT}/logs/${task}.log" 2>&1
    fi
    printf 'COMPLETE\n' > "${status}"
}

run_limited() {
    local limit=$1 method=$2 gpu_mode=$3
    shift 3
    local active=0 index=0 failures=0 contrast gpu
    for contrast in "$@"; do
        gpu=""
        if [[ ${gpu_mode} == yes && ${#GPU_IDS[@]} -gt 0 ]]; then
            gpu=${GPU_IDS[$((index % ${#GPU_IDS[@]}))]}
        fi
        run_one "${method}" "${contrast}" "${gpu}" &
        active=$((active + 1)); index=$((index + 1))
        if [[ ${active} -ge ${limit} ]]; then
            wait -n || failures=$((failures + 1))
            active=$((active - 1))
        fi
    done
    while [[ ${active} -gt 0 ]]; do
        wait -n || failures=$((failures + 1))
        active=$((active - 1))
    done
    [[ ${failures} -eq 0 ]]
}

contrasts=(CN_gt_Dementia MCI_gt_Dementia CN_gt_MCI)
run_limited "${WORKERS}" fdrs_absmax no "${contrasts[@]}"
run_limited "${WORKERS}" ising_absmax no "${contrasts[@]}"

GPU_IDS=()
if [[ -n ${CUDA_VISIBLE_DEVICES:-} ]]; then
    IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
elif command -v nvidia-smi >/dev/null 2>&1; then
    mapfile -t GPU_IDS < <(nvidia-smi --query-gpu=index --format=csv,noheader)
fi
if [[ ${GPU_WORKERS} -eq 0 ]]; then GPU_WORKERS=${#GPU_IDS[@]}; fi
GPU_WORKERS=$((GPU_WORKERS < 1 ? 1 : GPU_WORKERS))
while [[ ${#GPU_IDS[@]} -lt ${GPU_WORKERS} && ${#GPU_IDS[@]} -gt 0 ]]; do
    GPU_IDS+=("${#GPU_IDS[@]}")
done
run_limited "${GPU_WORKERS}" deepfdr_absmax yes "${contrasts[@]}"
run_limited "${WORKERS}" fchmrf_absmax no "${contrasts[@]}"

"${RSCRIPT}" "${RECOMPUTE}" "${OUTPUT}" aal3 "${WORK}" > "${OUTPUT}/logs/recompute_aal3.log" 2>&1
if [[ -s ${WORK}/atlas/harvard_oxford_thr25_2mm.nii.gz ]]; then
    "${RSCRIPT}" "${RECOMPUTE}" "${OUTPUT}" harvard_oxford "${WORK}" \
        > "${OUTPUT}/logs/recompute_harvard_oxford.log" 2>&1
fi
printf 'COMPLETE\n' > "${OUTPUT}/RUN_STATE"
date -Is > "${OUTPUT}/COMPLETED_AT"
