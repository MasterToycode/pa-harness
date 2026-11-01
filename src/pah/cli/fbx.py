"""`pah fbx` — the format bridge the part API forces on us, on the command line.

Why a command and not just a library function
---------------------------------------------
The one thing that can go wrong here is *setup*: Blender absent, `PAH_BLENDER`
pointing at a directory, a model with no mesh objects in it. All three fail before
any geometry exists, and none of them are visible from inside a `pah build` run
that has already spent a paid generation call. `pah fbx doctor` answers the setup
question in one line; `to` and `split` let a person check a specific file without
writing Python.

    pah fbx doctor                          # which Blender, or the fix
    pah fbx to out.fbx body.stl insert.stl  # PAH parts -> one FBX (what the API wants)
    pah fbx split dir/ decomposed.fbx       # a decomposition -> one STL per object

What it refuses
---------------
`to` an empty source list, a source that does not exist, or an input with no mesh
objects: all three would otherwise produce a file the part API rejects server-side,
with an error that names nothing on this end.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

EXIT_OK = 0
EXIT_NO = 1
EXIT_CRASH = 3


def add_parser(groups: argparse._SubParsersAction) -> None:
    fbx = groups.add_parser("fbx", help="convert to/from FBX via headless Blender")
    commands = fbx.add_subparsers(dest="command", required=True)

    doctor = commands.add_parser("doctor", help="is Blender installed and wired up?")
    doctor.add_argument("--json", action="store_true", help="machine-readable output")
    doctor.set_defaults(handler=_doctor)

    to = commands.add_parser("to", help="PAH meshes -> one multi-object FBX")
    to.add_argument("out", help="the .fbx to write")
    to.add_argument("sources", nargs="+", help=".stl/.glb/.obj/.ply inputs, one per part")
    to.add_argument("--json", action="store_true", help="machine-readable output")
    to.set_defaults(handler=_to)

    split = commands.add_parser("split", help="a multi-object FBX -> one STL per object")
    split.add_argument("out_dir", help="directory to write <object>.stl into")
    split.add_argument("source", help="the .fbx/.glb/.obj to split")
    split.add_argument("--json", action="store_true", help="machine-readable output")
    split.set_defaults(handler=_split)


def _doctor(args: argparse.Namespace) -> int:
    from pah.tools.fbx import describe  # noqa: PLC0415

    row = describe()
    if args.json:
        print(json.dumps(row, ensure_ascii=False, indent=1))
    elif row["blender"] == "ok":
        print(f"  Blender: {row['path']}")
        print("  FBX conversion is available.")
    else:
        print("  Blender: MISSING")
        for line in row["detail"].split(". "):
            if line.strip():
                print(f"    {line.strip()}.")
    return EXIT_OK if row["blender"] == "ok" else EXIT_NO


def _to(args: argparse.Namespace) -> int:
    from pah.tools.fbx import to_fbx  # noqa: PLC0415

    out = Path(args.out)
    names = to_fbx([Path(source) for source in args.sources], out)
    payload = {
        "out": str(out),
        "bytes": out.stat().st_size,
        "objects": names,
        "sources": list(args.sources),
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=1))
    else:
        print(f"  wrote {out}  ({out.stat().st_size / 1024:.0f} KB)")
        print(f"  objects: {', '.join(names)}")
        print("  ready to submit as the part API's File/Url field.")
    return EXIT_OK


def _split(args: argparse.Namespace) -> int:
    from pah.tools.fbx import from_fbx  # noqa: PLC0415

    parts = from_fbx(Path(args.source), Path(args.out_dir))
    payload = {"out_dir": str(args.out_dir), "source": args.source, "parts": parts}
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=1))
    else:
        print(f"  {args.source} -> {len(parts)} part(s) in {args.out_dir}")
        for name, path in sorted(parts.items()):
            size = Path(path).stat().st_size / 1024
            print(f"    {name:24s} {size:8.0f} KB  {path}")
    return EXIT_OK
