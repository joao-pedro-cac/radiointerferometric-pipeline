#!/usr/bin/env python3
"""Build the config sweep for the quantization experiment.

Axes (Cartesian product):
  datasets     : every subdir of --datasets-root containing obs_I.xds
  quantization : one-channel-at-a-time factorial (16 configs) --
                 all-f64 baseline; each of 4 channels -> {f32,f16,bf16};
                 all-channels -> {f32,f16,bf16}
  vis_fraction : --vis-fractions

Deconv params FROZEN across all configs (frozen-deconv design). rng_seed is
read per-dataset from that dataset's .osm filename (inert given first-N vis
slicing, but kept for provenance). One JSON per cell + manifest.csv.
"""
import argparse, csv, json, re
from pathlib import Path

CHANNELS = ["visibilities", "dirty_image", "psf", "clean_model"]
GROUND, TEST = "float64", ["float32", "float16", "bfloat16"]
CH = {"visibilities": "vis", "dirty_image": "dirty", "psf": "psf", "clean_model": "model"}
DT = {"float64": "f64", "float32": "f32", "float16": "f16", "bfloat16": "bf16"}

# Matches "<point|gaussian>_dr<digits>_s<digits>_..." e.g. "point_dr20_s1234_..." -> seed 1234
SEED_RE = re.compile(r"^(?:point|gaussian)_dr\d+_s(\d+)")


def quant_configs():
    base = {c: GROUND for c in CHANNELS}
    out = [("baseline-f64", dict(base))]
    for dt in TEST:
        for ch in CHANNELS:                       # single-channel: one channel -> dt, rest f64
            q = dict(base); q[ch] = dt
            out.append((f"{CH[ch]}-{DT[dt]}", q))
        # out.append((f"all-{DT[dt]}", {c: dt for c in CHANNELS}))   # all-channels -> dt
    return out                                     # 1 + 3*(4+1) = 16


def find_seed(dataset_name: str, fallback: int) -> int:
    """Pull the rng seed straight out of the dataset directory name, e.g.
    'point_dr20_s1234' -> 1234. Falls back to `fallback` (the --seed CLI
    value) and prints a warning if the name doesn't match the expected shape.
    """
    m = SEED_RE.match(dataset_name)
    if not m:
        print(f"WARN: {dataset_name}: directory name doesn't match "
              f"'<point|gaussian>_dr<N>_s<seed>...', using fallback seed {fallback}", flush=True)
        return fallback
    return int(m.group(1))


def discover(root):
    ds = []
    for d in sorted(Path(root).iterdir()):
        if (d / "obs_I.xds").exists():
            ds.append(d.name)
            if not (d / "truth_model.fits").exists():
                print(f"WARN: {d.name}: obs_I.xds present, truth_model.fits missing", flush=True)
    return ds


def build(args, dataset, qname, qmap, vf, seed):
    label = f"{dataset}__q-{qname}__vf{int(round(vf*100)):03d}"
    results_dir = Path(args.results_root).resolve() / label
    sigma = (args.sigma_map or {}).get(dataset, args.sigma)
    cfg = {
        "attributes": {
            "telescope": args.telescope,
            "sky_model": dataset.split("_")[0],           # point / gaussian, derived
            "vis_fraction": vf,
            "rng_seed": seed,
            "noise_floor": sigma,                         # sigma-as-input, by hand
            "clean_filename" : f"{label}.fits"
        },
        "file_paths": {
            "data_path": str(Path(args.datasets_root).resolve()),
            "dataset_name": dataset,
            "output_file_path": str(results_dir),         # unique -> no timestamp collision
        },
        "computation_parameters": {
            "gridding_epsilon": args.gridding_epsilon,
            "clean_variant": "hogbom",
            "clean_gamma": args.clean_gamma,
            "clean_peak_fraction": args.clean_pf,
            "clean_max_iterations": args.clean_maxiter,
            "feedback_loop_max_iterations": args.major_maxiter,
            "safe_float16_max": args.safe_float16_max,
            "experiment_commentary": f"{label} | frozen-deconv q={qname} vf={vf}",
            "enable_log": args.enable_log,
        },
        "quantization": {
            "coordinates": "float64",                     # kept for later, not yet wired
            "frequencies": "float64",                     #  "
            **qmap,
        },
    }
    return label, cfg, results_dir


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets-root", required=True)
    p.add_argument("--out-dir", required=True, help="where to write the .json configs")
    p.add_argument("--results-root", required=True, help="base for per-config output_file_path")
    p.add_argument("--vis-fractions", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0])
    p.add_argument("--telescope", default="meerkat")
    p.add_argument("--sigma", type=float, default=1e-4, help="noise_floor for all datasets unless --sigma-map")
    p.add_argument("--sigma-map", type=Path, default=None, help="JSON {dataset: sigma} for per-dataset noise")
    p.add_argument("--seed", type=int, default=42,
                   help="fallback rng_seed used only if a dataset's .osm file "
                        "is missing or its seed can't be parsed")
    p.add_argument("--gridding-epsilon", type=float, default=1e-5)
    p.add_argument("--clean-gamma", type=float, default=0.025)
    p.add_argument("--clean-pf", type=float, default=0.0075)
    p.add_argument("--clean-maxiter", type=int, default=10000)
    p.add_argument("--major-maxiter", type=int, default=5)
    p.add_argument("--safe-float16-max", type=float, default=65503.0)
    p.add_argument("--enable-log", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--dry-run", action="store_true", help="count only")
    args = p.parse_args()
    args.sigma_map = json.loads(args.sigma_map.read_text()) if args.sigma_map else None

    datasets = discover(args.datasets_root)
    if not datasets:
        raise SystemExit(f"no obs_I.xds datasets under {args.datasets_root}")

    seed_map = {name: find_seed(name, args.seed) for name in datasets}

    qcfgs, vfs = quant_configs(), args.vis_fractions
    total = len(datasets) * len(qcfgs) * len(vfs)
    print(f"{len(datasets)} datasets x {len(qcfgs)} quant x {len(vfs)} vf = {total} configs")
    if args.dry_run:
        return

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for dataset in datasets:
        seed = seed_map[dataset]
        for qname, qmap in qcfgs:
            for vf in vfs:
                label, cfg, rdir = build(args, dataset, qname, qmap, vf, seed)
                rdir.mkdir(parents=True, exist_ok=True)     # script chdir()s here
                (out_dir / f"{label}.json").write_text(json.dumps(cfg, indent=4))
                rows.append({"label": label, "dataset": dataset,
                             "sky": cfg["attributes"]["sky_model"], "quant": qname,
                             "vis_fraction": vf, "rng_seed": seed,
                             "config": str(out_dir / f"{label}.json"),
                             "output": str(rdir)})
    with open(out_dir / "manifest.csv", "w", newline="") as fd:
        w = csv.DictWriter(fd, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"wrote {len(rows)} configs + manifest.csv -> {out_dir}")


if __name__ == "__main__":
    main()













































# #!/usr/bin/env python3
# """Build the config sweep for the quantization experiment.

# Axes (Cartesian product):
#   datasets     : every subdir of --datasets-root containing obs_I.xds
#   quantization : one-channel-at-a-time factorial (16 configs) --
#                  all-f64 baseline; each of 4 channels -> {f32,f16,bf16};
#                  all-channels -> {f32,f16,bf16}
#   vis_fraction : --vis-fractions

# Deconv params FROZEN across all configs (frozen-deconv design). rng_seed fixed
# (inert given first-N vis slicing). One JSON per cell + manifest.csv.
# """
# import argparse, csv, json
# from pathlib import Path

# CHANNELS = ["visibilities", "dirty_image", "psf", "clean_model"]
# GROUND, TEST = "float64", ["float32", "float16", "bfloat16"]
# CH = {"visibilities": "vis", "dirty_image": "dirty", "psf": "psf", "clean_model": "model"}
# DT = {"float64": "f64", "float32": "f32", "float16": "f16", "bfloat16": "bf16"}


# def quant_configs():
#     base = {c: GROUND for c in CHANNELS}
#     out = [("baseline-f64", dict(base))]
#     for dt in TEST:
#         for ch in CHANNELS:                       # single-channel: one channel -> dt, rest f64
#             q = dict(base); q[ch] = dt
#             out.append((f"{CH[ch]}-{DT[dt]}", q))
#         # out.append((f"all-{DT[dt]}", {c: dt for c in CHANNELS}))   # all-channels -> dt
#     return out                                     # 1 + 3*(4+1) = 16


# def discover(root):
#     ds = []
#     for d in sorted(Path(root).iterdir()):
#         if (d / "obs_I.xds").exists():
#             ds.append(d.name)
#             if not (d / "truth_model.fits").exists():
#                 print(f"WARN: {d.name}: obs_I.xds present, truth_model.fits missing", flush=True)
#     return ds


# def build(args, dataset, qname, qmap, vf):
#     label = f"{dataset}__q-{qname}__vf{int(round(vf*100)):03d}"
#     results_dir = Path(args.results_root).resolve() / label
#     sigma = (args.sigma_map or {}).get(dataset, args.sigma)
#     cfg = {
#         "attributes": {
#             "telescope": args.telescope,
#             "sky_model": dataset.split("_")[0],           # point / gaussian, derived
#             "vis_fraction": vf,
#             "rng_seed": args.seed,
#             "noise_floor": sigma,                         # sigma-as-input, by hand
#             "clean_filename" : f"{label}.fits"
#         },
#         "file_paths": {
#             "data_path": str(Path(args.datasets_root).resolve()),
#             "dataset_name": dataset,
#             "output_file_path": str(results_dir),         # unique -> no timestamp collision
#         },
#         "computation_parameters": {
#             "gridding_epsilon": args.gridding_epsilon,
#             "clean_variant": "hogbom",
#             "clean_gamma": args.clean_gamma,
#             "clean_peak_fraction": args.clean_pf,
#             "clean_max_iterations": args.clean_maxiter,
#             "feedback_loop_max_iterations": args.major_maxiter,
#             "safe_float16_max": args.safe_float16_max,
#             "experiment_commentary": f"{label} | frozen-deconv q={qname} vf={vf}",
#             "enable_log": args.enable_log,
#         },
#         "quantization": {
#             "coordinates": "float64",                     # kept for later, not yet wired
#             "frequencies": "float64",                     #  "
#             **qmap,
#         },
#     }
#     return label, cfg, results_dir


# def main():
#     p = argparse.ArgumentParser(description=__doc__,
#                                 formatter_class=argparse.RawDescriptionHelpFormatter)
#     p.add_argument("--datasets-root", required=True)
#     p.add_argument("--out-dir", required=True, help="where to write the .json configs")
#     p.add_argument("--results-root", required=True, help="base for per-config output_file_path")
#     p.add_argument("--vis-fractions", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0])
#     p.add_argument("--telescope", default="meerkat")
#     p.add_argument("--sigma", type=float, default=1e-4, help="noise_floor for all datasets unless --sigma-map")
#     p.add_argument("--sigma-map", type=Path, default=None, help="JSON {dataset: sigma} for per-dataset noise")
#     p.add_argument("--seed", type=int, default=42)
#     p.add_argument("--gridding-epsilon", type=float, default=1e-5)
#     p.add_argument("--clean-gamma", type=float, default=0.025)
#     p.add_argument("--clean-pf", type=float, default=0.0075)
#     p.add_argument("--clean-maxiter", type=int, default=10000)
#     p.add_argument("--major-maxiter", type=int, default=5)
#     p.add_argument("--safe-float16-max", type=float, default=65503.0)
#     p.add_argument("--enable-log", action=argparse.BooleanOptionalAction, default=True)
#     p.add_argument("--dry-run", action="store_true", help="count only")
#     args = p.parse_args()
#     args.sigma_map = json.loads(args.sigma_map.read_text()) if args.sigma_map else None

#     datasets = discover(args.datasets_root)
#     if not datasets:
#         raise SystemExit(f"no obs_I.xds datasets under {args.datasets_root}")
#     qcfgs, vfs = quant_configs(), args.vis_fractions
#     total = len(datasets) * len(qcfgs) * len(vfs)
#     print(f"{len(datasets)} datasets x {len(qcfgs)} quant x {len(vfs)} vf = {total} configs")
#     if args.dry_run:
#         return

#     out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
#     rows = []
#     for dataset in datasets:
#         for qname, qmap in qcfgs:
#             for vf in vfs:
#                 label, cfg, rdir = build(args, dataset, qname, qmap, vf)
#                 rdir.mkdir(parents=True, exist_ok=True)     # script chdir()s here
#                 (out_dir / f"{label}.json").write_text(json.dumps(cfg, indent=4))
#                 rows.append({"label": label, "dataset": dataset,
#                              "sky": cfg["attributes"]["sky_model"], "quant": qname,
#                              "vis_fraction": vf, "config": str(out_dir / f"{label}.json"),
#                              "output": str(rdir)})
#     with open(out_dir / "manifest.csv", "w", newline="") as fd:
#         w = csv.DictWriter(fd, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
#     print(f"wrote {len(rows)} configs + manifest.csv -> {out_dir}")


# if __name__ == "__main__":
#     main()
