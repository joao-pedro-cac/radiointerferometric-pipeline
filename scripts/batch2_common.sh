# Shared batch-2 definitions, sourced by exp_*.sh. Not executable on its own.
#
# Schemes chosen from the v3 pilot conclusions: model stage dominates
# (modelf16 vs allf16), vis quantization looks safe (visf16), dirty/PSF look
# safe (dirtyf16), compounding (allf16/allbf16). f32 already shown invisible,
# allf16raw already shown NaN — both excluded from batch 2.
B2_SCHEMES="f64=float64:float64:float64:float64:65503 allf16=float16:float16:float16:float16:65503 allbf16=bfloat16:bfloat16:bfloat16:bfloat16:65503 modelf16=float64:float64:float64:float16:65503 visf16=float16:float64:float64:float64:65503 dirtyf16=float64:float16:float64:float64:65503"

# spec list -> dataset-name prefixes (what study_sweep_v3.sh expects in SKIES)
b2_prefixes() {
  local out=""
  for spec in $1; do
    out="$out ${spec%%:*}"
  done
  echo "$out"
}
