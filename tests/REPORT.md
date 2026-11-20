# Test report — v2 library gate

Run: `cd code && $VENV_PATH/bin/python -m pytest tests/ -q` (pytest 8.4.2, numpy 1.26.4)
Status: **43 passed, 0 failed** (2026-11-20, incl. anti-BFP global-scale test and the probe gate)

| Area | Test | Analytic ground truth | Expected | Result |
|---|---|---|---|---|
| metrics | `test_cross_correlation_*` | cc(x,x)=1, cc(x,ax+b)=1, cc(x,-x)=-1 | exact ±1e-12 | pass |
| metrics | `test_flux_conservation_exact` | model=truth → 1, 0.5·truth → 0.5 | exact | pass |
| metrics | `test_residual_rms_*` | const c → |c|; N(0,σ) → σ ±2% | analytic | pass |
| metrics | `test_residual_mad_gaussian_noise` | 1.4826·MAD(N(0,σ)) = σ ±2% | analytic | pass |
| metrics | `test_achieved_dr_definition` | peak / residual_mad | definitional | pass |
| metrics | `test_snr_db_*` | identical → >100 dB, monotone in MSE | definitional | pass |
| quantize | f32/f16/bf16 round-trip bounds | rel err ≤ 2⁻²³ / 2⁻¹⁰ / 2⁻⁷; f16,bf16 values exactly representable (low bits zero) | IEEE-754 | pass |
| quantize | `test_bfloat16_round_to_nearest_even` | 0x…0001/0x7FFF/0x8000 → down, 0x8001 → up | RNE bit-exact | pass |
| quantize | rescale | f16 output ≤ 65504, finite, scale = 65503/max | range guard | pass |
| quantize | `test_f16_global_scale_not_block_floating_point` | ONE global scalar scale (65503/amax); outputs exactly f16-representable; subnormal-region errors >10× f16 eps (anti-BFP signature) | global-scale proof | pass |
| operators | to_native/to_display | exact inverses | identity | pass |
| operators | phase-centre point source | vis ≡ flux; dirty/psf peak = flux ±1e-4 | analytic | pass |
| operators | beam kernel | peak = 1, sum = beam_area_px | analytic | pass |
| operators | restore delta | restored peak = flux, sum = flux·beam_area | analytic | pass |
| operators | synth_beam on known gaussian | recovers (bmaj,bmin) ±5% | analytic | pass |
| operators | vis2dirty∘dirty2vis round-trip | corr with I⊛psf > 0.9 (wgridding-limited) | consistency | pass |
| sky | render point / gauss | sum = flux; gauss peak = I/(2πσxσy) | analytic | pass |
| sky | sample_field | realized DR == dr exactly; positions in FOV square | analytic | pass |
| sky | WCS round-trip | radec→pix→radec identity | analytic | pass |
| clean | hogbom recovers point source | model at source pixel, sum ≈ flux | synthetic PSF | pass |
| clean | clean_loop converges | flux recovered ±5%, residual at noise level | synthetic + noise | pass |
| probe | `test_log2_hist_*` | known |x| → exact per-octave bin counts; zeros dropped, out-of-range clipped into edge bins (counts sum = #nonzero); complex → modulus | analytic | pass |
| probe | `test_array_stats_*` | max/min-nonzero/median/MAD exact on known array; f16/bf16 limit fractions exact; constants = 65504, 2⁻²⁴, 2⁻¹³³, f32max | analytic | pass |
| probe | collector round-trip | `to_arrays`/`from_arrays` preserve labels, edges, hists, stats, scalars across heterogeneous snap keys | identity | pass |
| probe | pipeline integration | `run_pipeline(probe=True)` returns snaps init/k*/final; hist sums = #nonzero of returned arrays; per-cycle scalars res_peak/res_mad/n_components/s_*; scales dict returned | synthetic ring uv | pass |
| probe | cast arrays | with |vis|~1e6 and q-vis=f16 the rescale triggers (s_vis<1), `vis_cast` appears, modulus ≤ 65504·√2 (complex casts re/im separately), no bf16-range overflow | f16 envelope | pass |

Notes:
- Högbom's `3×MAD` threshold dominates when the dirty image is noise-dominated;
  `pf` dominates for PSF-dominated (noiseless) images. Tests pin both regimes.
- ducc0 wgridding is not an exact adjoint: round-trip correlation is ~0.9–0.98,
  hence the consistency (not exactness) bound above; the sharp invariant is the
  phase-centre delta test.
