"""hemoflow command-line interface.

Subcommands:
  list-plugins        list all registered viscosity/drag/lift/migration/geometry/bc plugins
  inspect-geometry    show a geometry family's metadata and build a host-safe MeshSpec + graded mesh
  solve               run a FEniCSx Euler-Euler solve (requires the dolfinx environment)
  build-web-bundle    build the compact site/data corpus bundle from a corpus_flowfields dir
"""
from __future__ import annotations

import argparse
import json
import sys


def _cmd_list_plugins(args) -> int:
    import hemoflow
    plugins = hemoflow.list_plugins()
    for kind in ("viscosity", "drag", "lift", "migration", "geometry", "bc"):
        names = plugins.get(kind, [])
        print(f"[{kind}] ({len(names)})")
        for n in names:
            extra = ""
            if kind == "geometry":
                meta = hemoflow.geometry_metadata(n)
                extra = f"  -- {meta.get('name','')} | {meta.get('topology','')}"
            print(f"  {n}{extra}")
    return 0


def _cmd_inspect_geometry(args) -> int:
    import numpy as np
    import hemoflow
    from hemoflow.geometries import builders, mesher
    fam = args.family.upper()
    meta = hemoflow.geometry_metadata(fam)
    if not meta:
        print(f"unknown geometry family {fam!r}", file=sys.stderr)
        return 2
    print(f"family   : {fam}")
    print(f"name     : {meta.get('name')}")
    print(f"topology : {meta.get('topology')}")
    print(f"desc     : {meta.get('description')}")
    params = {p["name"]: p["default"] for p in meta.get("params", [])}
    if args.param:
        for kv in args.param:
            k, v = kv.split("=", 1)
            params[k] = float(v) if "." in v or "e" in v.lower() else int(v)
    print(f"params   : {json.dumps(params)}")
    print("param spec:")
    for p in meta.get("params", []):
        print(f"  {p['name']:<16} default={p['default']:<8} min={p['min']} max={p['max']}"
              + (" [int]" if p.get("int") else ""))
    spec = builders.build_geometry(fam, params, level=1)
    nodes, cells, tagged, min_wall_dy = mesher.graded_mesh_arrays(fam, params, refine=args.refine)
    areas = mesher.cell_signed_areas(nodes, cells)
    tags = sorted({t[2] for t in tagged})
    print(f"MeshSpec : blocks_2d={len(spec.blocks_2d)} verts_2d={len(spec.vertices_2d)} "
          f"inlet_height={spec.inlet_height} outlet_heights={spec.outlet_heights}")
    print(f"graded   : refine={args.refine} nodes={len(nodes)} cells={len(cells)} "
          f"min_cell_area={float(areas.min()):.3e} min_wall_dy={min_wall_dy:.3e}")
    print(f"bc tags  : INLET={mesher.INLET} OUTLET={mesher.OUTLET} WALL={mesher.WALL} -> found {tags}")
    return 0


def _cmd_solve(args) -> int:
    import hemoflow
    try:
        import dolfinx  # noqa: F401
    except Exception:
        print("solve requires the dolfinx environment (dolfinx/dolfinx:v0.9.0 Docker image or "
              "conda-forge dolfinx). Install it, then re-run: hemoflow solve ...", file=sys.stderr)
        return 3
    from hemoflow.solver.stationary import EulerEulerSolver
    cfg = hemoflow.EulerEulerConfig.v4_wu(re_p=args.re, d_hat=args.d_hat, phi_in=args.phi_in,
                                          k_star=args.k_star)
    solver = EulerEulerSolver(cfg)
    res = solver.solve(args.family, geometry_params={p["name"]: p["default"]
                                                      for p in hemoflow.geometry_metadata(args.family).get("params", [])})
    print(json.dumps(res.summary(), indent=2, default=str))
    if args.out:
        res.to_npz(args.out)
        print(f"wrote {args.out}")
    return 0


def _cmd_build_web_bundle(args) -> int:
    import importlib.util
    from pathlib import Path
    # load scripts/build_web_bundle.py from the repo root (parent of the hemoflow package)
    here = Path(__file__).resolve().parent.parent
    script = here / "scripts" / "build_web_bundle.py"
    if not script.exists():
        print(f"build_web_bundle.py not found at {script}", file=sys.stderr)
        return 2
    spec = importlib.util.spec_from_file_location("build_web_bundle", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.main(["--src", args.src, "--out", args.out])


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="hemoflow", description="hemoflow 2D Euler-Euler blood-flow library CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("list-plugins", help="list all registered plugins")
    sp.set_defaults(func=_cmd_list_plugins)

    sp = sub.add_parser("inspect-geometry", help="inspect a geometry family (G1-G13, KG)")
    sp.add_argument("family", help="geometry family id, e.g. G2")
    sp.add_argument("--refine", type=float, default=0.5, help="graded mesh refine factor")
    sp.add_argument("--param", action="append", default=[], help="override param, e.g. E=3.0")
    sp.set_defaults(func=_cmd_inspect_geometry)

    sp = sub.add_parser("solve", help="run a FEniCSx Euler-Euler solve (needs dolfinx)")
    sp.add_argument("family", help="geometry family id")
    sp.add_argument("--re", type=float, default=10.0)
    sp.add_argument("--d-hat", dest="d_hat", type=float, default=0.063)
    sp.add_argument("--phi-in", dest="phi_in", type=float, default=0.30)
    sp.add_argument("--k-star", dest="k_star", type=float, default=0.0)
    sp.add_argument("--out", default=None, help="write FlowResult .npz here")
    sp.set_defaults(func=_cmd_solve)

    sp = sub.add_parser("build-web-bundle", help="build the site/data corpus bundle")
    sp.add_argument("--src", required=True, help="corpus_flowfields dir (G1-G13, V1/V2/V4)")
    sp.add_argument("--out", required=True, help="output site/data dir")
    sp.set_defaults(func=_cmd_build_web_bundle)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
