#!/usr/bin/env bash
set -euo pipefail

THREADS=64
REQUESTED_TIME=30

usage() {
  cat <<'USAGE'
Usage: randomise_parallel_limited.sh [--threads N] [--requested-time MIN] -- <randomise options>

Runs FSL randomise in parallel fragments with a local fsl_sub array limit.
Output filenames follow randomise_parallel/randomise conventions.
USAGE
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --threads)
      THREADS="$2"
      shift 2
      ;;
    --requested-time)
      REQUESTED_TIME="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    *)
      break
      ;;
  esac
done

if ! [[ "$THREADS" =~ ^[0-9]+$ ]] || [ "$THREADS" -lt 1 ]; then
  echo "ERROR: --threads must be a positive integer; got ${THREADS}" >&2
  exit 1
fi

if ! [[ "$REQUESTED_TIME" =~ ^[0-9]+$ ]] || [ "$REQUESTED_TIME" -lt 1 ]; then
  echo "ERROR: --requested-time must be a positive integer; got ${REQUESTED_TIME}" >&2
  exit 1
fi

if [ "$#" -le 3 ]; then
  usage >&2
  exit 1
fi

if [ -z "${FSLDIR:-}" ]; then
  echo "ERROR: FSLDIR is not set." >&2
  exit 1
fi

quote_args() {
  local quoted=""
  local arg
  for arg in "$@"; do
    printf -v quoted '%s %q' "$quoted" "$arg"
  done
  printf '%s' "$quoted"
}

RANDOMISE_OUTPUT=$("${FSLDIR}/bin/randomise" "$@" -Q) || {
  echo "ERROR: randomise could not initialise with the command line given." >&2
  exit 1
}

read -r PERMS CONTRASTS ROOTNAME PERMS_PER_SLOT _ <<< "$RANDOMISE_OUTPUT"
if [ -z "${PERMS:-}" ] || [ -z "${CONTRASTS:-}" ] || [ -z "${ROOTNAME:-}" ] || [ -z "${PERMS_PER_SLOT:-}" ]; then
  echo "ERROR: could not parse randomise -Q output: ${RANDOMISE_OUTPUT}" >&2
  exit 1
fi

if [ "$PERMS_PER_SLOT" -lt 1 ]; then
  echo "ERROR: randomise reported invalid permutations per slot: ${PERMS_PER_SLOT}" >&2
  exit 1
fi

BASENAME=$(basename "$ROOTNAME")
DIRNAME=$(dirname "$ROOTNAME")
LOGDIR="${DIRNAME}/${BASENAME}_logs"
mkdir -p "$LOGDIR"
STATUS_FILE="${ROOTNAME}_randomise.status"
SUBMISSION_FILE="${ROOTNAME}_submission.log"
printf 'SUBMITTING\n' > "$STATUS_FILE"

SLOTS_PER_CONTRAST=$((PERMS / PERMS_PER_SLOT))
if [ "$SLOTS_PER_CONTRAST" -lt 1 ]; then
  SLOTS_PER_CONTRAST=1
fi

PERMS_PER_CONTRAST=$((PERMS_PER_SLOT * SLOTS_PER_CONTRAST))
REQUESTED_SLOTS=$((CONTRASTS * SLOTS_PER_CONTRAST))
EFFECTIVE_PERMS=$((PERMS_PER_CONTRAST - SLOTS_PER_CONTRAST + 1))

echo "Generating ${REQUESTED_SLOTS} fragments for ${CONTRASTS} contrasts with ${PERMS_PER_SLOT} permutations per fragment."
echo "Local fsl_sub array limit: ${THREADS}; requested time per fragment: ${REQUESTED_TIME} minutes."
echo "Total permutations per contrast after fragment rounding: ${PERMS_PER_CONTRAST}."

TASK_FILE="${DIRNAME}/${BASENAME}.generate"
DEFRAG_FILE="${DIRNAME}/${BASENAME}.defragment"
rm -f "$TASK_FILE" "$DEFRAG_FILE"

RANDOMISE_ARGS=$(quote_args "$@")
RANDOMISE_BIN="${FSLDIR}/bin/randomise"

CURRENT_SEED=1
while [ "$CURRENT_SEED" -le "$SLOTS_PER_CONTRAST" ]; do
  SLEEPTIME=$CURRENT_SEED
  CURRENT_CONTRAST=1
  while [ "$CURRENT_CONTRAST" -le "$CONTRASTS" ]; do
    {
      printf 'FSLOUTPUTTYPE=NIFTI_GZ; sleep %q; ' "$SLEEPTIME"
      printf '%q%s' "$RANDOMISE_BIN" "$RANDOMISE_ARGS"
      printf ' -n %q -o %q --seed=%q' "$PERMS_PER_SLOT" "${ROOTNAME}_SEED${CURRENT_SEED}" "$CURRENT_SEED"
      if [ "$CONTRASTS" -ne 1 ]; then
        printf ' --skipTo=%q' "$CURRENT_CONTRAST"
      fi
      printf '\n'
    } >> "$TASK_FILE"
    CURRENT_CONTRAST=$((CURRENT_CONTRAST + 1))
  done
  CURRENT_SEED=$((CURRENT_SEED + 1))
done
chmod +x "$TASK_FILE"

GENERATE_ID=$("${FSLDIR}/bin/fsl_sub" \
  -T "$REQUESTED_TIME" \
  -N "${BASENAME}.generate" \
  -n \
  -l "$LOGDIR" \
  -t "$TASK_FILE" \
  -x "$THREADS")

cat > "$DEFRAG_FILE" <<DEFRAG
#!/usr/bin/env bash
set -euo pipefail
STATUS_FILE="$STATUS_FILE"
printf 'MERGING\n' > "\$STATUS_FILE"
trap 'printf "FAILED\\n" > "\$STATUS_FILE"' ERR

echo "Merging stat images"
for FIRSTSEED in \$(imglob -extension "${ROOTNAME}_SEED1_"*_p_* "${ROOTNAME}_SEED1_"*_corrp_*); do
  ADDCOMMAND="\$FIRSTSEED"
  ACTIVESEED=1
  if [ -e "\$FIRSTSEED" ]; then
    while [ "\$ACTIVESEED" -lt "$SLOTS_PER_CONTRAST" ]; do
      ACTIVESEED=\$((ACTIVESEED + 1))
      NEXTSEED="\${FIRSTSEED/_SEED1_/_SEED\${ACTIVESEED}_}"
      if [ ! -e "\$NEXTSEED" ]; then
        echo "ERROR: missing randomise fragment \$NEXTSEED" >&2
        exit 1
      fi
      ADDCOMMAND="\$ADDCOMMAND -add \$NEXTSEED"
    done
    echo "\$ADDCOMMAND"
    fslmaths \$ADDCOMMAND -mul "$PERMS_PER_SLOT" -div "$EFFECTIVE_PERMS" "\${FIRSTSEED/_SEED1/}"
  fi
done

echo "Merging text files"
for FIRSTSEED in "${ROOTNAME}_SEED1_"*perm_*.txt "${ROOTNAME}_SEED1_"*_p_*.txt "${ROOTNAME}_SEED1_"*_corrp_*.txt; do
  ACTIVESEED=1
  if [ -e "\$FIRSTSEED" ]; then
    while [ "\$ACTIVESEED" -le "$SLOTS_PER_CONTRAST" ]; do
      if [ "\$ACTIVESEED" -eq 1 ]; then
        cat "\${FIRSTSEED/_SEED1_/_SEED\${ACTIVESEED}_}" >> "\${FIRSTSEED/_SEED1/}"
      else
        tail -n +2 "\${FIRSTSEED/_SEED1_/_SEED\${ACTIVESEED}_}" >> "\${FIRSTSEED/_SEED1/}"
      fi
      ACTIVESEED=\$((ACTIVESEED + 1))
    done
  fi
done

echo "Renaming raw stats"
for TYPE in _ _tfce_; do
  for FIRSTSEED in \$(imglob -extension "${ROOTNAME}_SEED1\${TYPE}tstat"* "${ROOTNAME}_SEED1\${TYPE}fstat"*); do
    if [ -e "\$FIRSTSEED" ]; then
      cp "\$FIRSTSEED" "\${FIRSTSEED/_SEED1/}"
    fi
  done
done

ACTIVESEED=1
while [ "\$ACTIVESEED" -le "$SLOTS_PER_CONTRAST" ]; do
  rm -f "${ROOTNAME}_SEED\${ACTIVESEED}"*_p_*
  rm -f "${ROOTNAME}_SEED\${ACTIVESEED}"*_corrp_*
  rm -f \$(imglob -extensions "${ROOTNAME}_SEED\${ACTIVESEED}_"?stat*)
  rm -f "${ROOTNAME}_SEED\${ACTIVESEED}_"*perm_*.txt "${ROOTNAME}_SEED\${ACTIVESEED}_"*_p_*.txt "${ROOTNAME}_SEED\${ACTIVESEED}_"*_corrp_*.txt
  ACTIVESEED=\$((ACTIVESEED + 1))
done

cat > "${ROOTNAME}_permutation_metadata.json" <<JSON
{
  "requested_permutations": $PERMS,
  "fragments_per_contrast": $SLOTS_PER_CONTRAST,
  "permutations_per_fragment": $PERMS_PER_SLOT,
  "rounded_permutations_per_contrast": $PERMS_PER_CONTRAST,
  "effective_permutation_denominator": $EFFECTIVE_PERMS
}
JSON

printf 'COMPLETE\n' > "\$STATUS_FILE"
trap - ERR
echo "Done"
DEFRAG
chmod +x "$DEFRAG_FILE"

DEFRAGMENT_TIME=20
if [ "$REQUESTED_SLOTS" -ge 150 ]; then
  DEFRAGMENT_TIME=40
fi

printf 'SUBMITTED generate_job=%s\n' "$GENERATE_ID" > "$STATUS_FILE"
"${FSLDIR}/bin/fsl_sub" \
  -j "$GENERATE_ID" \
  -T "$DEFRAGMENT_TIME" \
  -l "$LOGDIR" \
  -N "${BASENAME}.defragment" \
  "$DEFRAG_FILE" > "$SUBMISSION_FILE"
