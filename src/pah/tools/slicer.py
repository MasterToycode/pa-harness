"""Slicer dry run: turn "the mesh looks printable" into "a slicer accepted it".

Why this module exists
----------------------
The audit (`docs/remediation-plan.md`, P2-4) named the gap: `printable` was decided
by mesh-level checks — watertight, wall thickness, optionally overhang — and none of
those is a manufacturing process. A file can pass all of them and still be
unsliceable (zero-area features, layers with no extrusion, a wall thinner than one
line width). The only tool that knows is a slicer, so the export gate must be able to
run one and must be able to say, in the manifest, that it did.

The hard case, and the rule that follows
----------------------------------------
A slicer is a large external program and this machine may not have one. The tempting
shortcut — "no slicer found, so nothing to object to" — turns an unasked question into
a pass, which is exactly the class of bug the audit is about. So `find_slicer` raises
`SlicerMissing` with a message that says what was *not* verified and where to get a
slicer; `slice_mesh` never invents a number, and a caller that cannot slice must
export `printable: false`. **Missing is not passing.**

What a passed dry run still does not prove
------------------------------------------
A slicer accepting a G-code file means the geometry was expressible as toolpaths with
the chosen profile. It does not mean the part is strong, dimensionally correct, or
prints on *your* printer: the profile decides line width, supports, temperatures and
speeds, and those are physical assumptions. This module therefore reports **evidence**
(`backend`, `seconds_estimate`, `filament_mm`, `layers`, `support_used`) and never a
verdict about manufacturability beyond "this slicer, with this profile, produced
G-code and reported no empty layers".

Parsing rules
-------------
Every field is optional, and a field whose comment is absent stays `None` rather than
being estimated. Three comment dialects are recognised because three slicers are
supported:

* PrusaSlicer / OrcaSlicer: `; estimated printing time (normal mode) = 1h 2m 3s`,
  `; filament used [mm] = 1234.5`, `; total layers count = 42`,
  `; support_material = 0`;
* CuraEngine: `;TIME:4020`, `;Filament used: 1.234m`, `;LAYER_COUNT:42`,
  and `"support_enable": "True"` inside a `;SETTING_3 {...}` line.

`parse_slicer_output` is a pure function over text so the tests never need a slicer
installed; the dialect knowledge is testable in isolation, which is the only way it
stays correct across slicer versions.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess  # noqa: S404 - running a slicer is this module's job
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "SliceResult",
    "Slicer",
    "SlicerError",
    "SlicerMissing",
    "find_slicer",
    "parse_slicer_output",
    "slice_mesh",
]

#: Executables to look for on `PATH`, per backend, in preference order. PrusaSlicer
#: and OrcaSlicer use the `-console`/CLI binary because the GUI binary does not take
#: `--export-gcode` reliably; CuraEngine is the headless engine Cura ships.
_CANDIDATES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("prusaslicer", ("prusa-slicer-console", "prusa-slicer", "PrusaSlicer")),
    ("orcaslicer", ("orca-slicer", "OrcaSlicer")),
    ("curaengine", ("CuraEngine", "cura-engine")),
)

#: Where the Windows installers put things. `shutil.which` does not search these, and
#: on Windows that is the normal case rather than the exception.
_WINDOWS_PATHS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "prusaslicer",
        (
            r"C:\Program Files\Prusa3D\PrusaSlicer\prusa-slicer-console.exe",
            r"C:\Program Files\Prusa3D\PrusaSlicer\prusa-slicer.exe",
        ),
    ),
    (
        "orcaslicer",
        (
            r"C:\Program Files\OrcaSlicer\orca-slicer.exe",
            r"C:\Program Files\OrcaSlicer\OrcaSlicer.exe",
        ),
    ),
    (
        "curaengine",
        (
            r"C:\Program Files\UltiMaker Cura\CuraEngine.exe",
            r"C:\Program Files\Cura\CuraEngine.exe",
        ),
    ),
)

#: What to tell someone whose machine has no slicer. Concrete, because "install a
#: slicer" is not actionable and a fabricated pass is worse than a stop.
_INSTALL_HINT = (
    "refusing to claim manufacturing evidence. Install PrusaSlicer "
    "(https://www.prusa3d.com/prusaslicer/), OrcaSlicer "
    "(https://github.com/SoftFever/OrcaSlicer) or CuraEngine, or set PAH_SLICER to "
    "the executable. Without one, `printable` stays false: the mesh layer was checked, "
    "the manufacturing layer was not."
)


class SlicerMissing(RuntimeError):
    """No slicer could be found, so no slicing evidence can be produced."""


class SlicerError(RuntimeError):
    """A slicer was found but this request cannot be handed to it (e.g. no profile)."""


@dataclass(frozen=True)
class Slicer:
    """One discovered slicer: the command prefix and which comment dialect it speaks.

    `argv` is usually just `(path,)`, but `$PAH_SLICER` may carry fixed arguments
    (`"C:/Python/python.exe" "fake_slicer.py"`), which is how a wrapper around an
    engine that needs a profile is installed — and how the tests drive a fake slicer
    without writing platform-specific launchers.
    """

    path: str
    backend: str
    argv: tuple[str, ...] = ()

    @property
    def command(self) -> list[str]:
        return list(self.argv) if self.argv else [self.path]


@dataclass(frozen=True)
class SliceResult:
    """The evidence from one dry run. `None` means "the slicer did not report it".

    `ok` is deliberately conservative: a zero exit code is not enough, because the
    audit's question is "did it produce a usable toolpath and say so". Time, filament
    and layer count are required (they are what the manifest reports); support usage
    is optional because a slicer may omit the flag and a missing flag is not a defect.
    """

    ok: bool
    backend: str
    gcode_path: str | None = None
    seconds_estimate: int | None = None
    filament_mm: float | None = None
    filament_g: float | None = None
    support_used: bool | None = None
    layers: int | None = None
    empty_layers: int = 0
    problems: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        """The manifest row: JSON-ready, `problems` as a list for a reader."""
        return {
            "ok": self.ok,
            "backend": self.backend,
            "gcode_path": self.gcode_path,
            "seconds_estimate": self.seconds_estimate,
            "filament_mm": self.filament_mm,
            "filament_g": self.filament_g,
            "support_used": self.support_used,
            "layers": self.layers,
            "empty_layers": self.empty_layers,
            "problems": list(self.problems),
        }


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
def find_slicer(*, environ: dict[str, str] | None = None) -> Slicer:
    """The slicer to use: `$PAH_SLICER`, then PATH, then the Windows install paths.

    `$PAH_SLICER` is checked first so a machine with several slicers (or a test with a
    fake one) can pin the choice without editing `PATH`. A value that points at
    nothing is an error, not a fall-through: someone who set the variable meant it,
    and silently using a different slicer would make two runs incomparable.
    """
    env = environ if environ is not None else os.environ
    explicit = str(env.get("PAH_SLICER", "") or "").strip()
    if explicit:
        if Path(explicit).exists():
            # A plain path wins over splitting: an install directory may contain
            # spaces, and quoting it should not be mandatory for the common case.
            return Slicer(path=explicit, backend=_backend_for(Path(explicit).name))
        argv = _split_command(explicit)
        if not argv or not Path(argv[0]).exists():
            raise SlicerMissing(
                f"PAH_SLICER points at {explicit!r}, which does not exist; " + _INSTALL_HINT
            )
        return Slicer(path=argv[0], backend=_backend_for(Path(argv[0]).name), argv=tuple(argv))

    for backend, names in _CANDIDATES:
        for name in names:
            found = shutil.which(name)
            if found:
                return Slicer(path=found, backend=backend)

    for backend, candidates in _WINDOWS_PATHS:
        for candidate in candidates:
            if Path(candidate).exists():
                return Slicer(path=candidate, backend=backend)

    raise SlicerMissing(
        "no slicer found on this machine (looked for PrusaSlicer, OrcaSlicer and "
        "CuraEngine on PATH, in the Windows install directories, and at $PAH_SLICER); "
        + _INSTALL_HINT
    )


def _split_command(value: str) -> tuple[str, ...]:
    """Split a quoted command line, keeping Windows backslashes intact.

    `shlex.split` is not used on purpose: with `posix=True` it treats `\\` as an escape
    and mangles `C:\\Python\\python.exe`; with `posix=False` it leaves the quotes in the
    tokens. This splitter only knows quotes and whitespace, which is all a slicer
    command needs.
    """
    parts: list[str] = []
    current: list[str] = []
    quoted = False
    for char in value:
        if char == '"':
            quoted = not quoted
        elif char.isspace() and not quoted:
            if current:
                parts.append("".join(current))
                current = []
        else:
            current.append(char)
    if quoted:
        raise SlicerMissing(f"PAH_SLICER has an unterminated quote: {value!r}")
    if current:
        parts.append("".join(current))
    return tuple(parts)


def _backend_for(executable_name: str) -> str:
    """Which comment dialect the executable speaks, guessed from its name."""
    lowered = executable_name.lower()
    if "prusa" in lowered:
        return "prusaslicer"
    if "orca" in lowered:
        return "orcaslicer"
    if "cura" in lowered:
        return "curaengine"
    return "custom"


# --------------------------------------------------------------------------- #
# Parsing (pure: the tests drive this without a slicer)
# --------------------------------------------------------------------------- #
_TIME_COMMENT = re.compile(
    r"estimated printing time(?:\s*\([^)]*\))?\s*=\s*(?P<value>[^\r\n;]+)", re.IGNORECASE
)
_TIME_DURATION = re.compile(
    r"(?:(?P<d>\d+)\s*d(?:ays?)?)?\s*(?:(?P<h>\d+)\s*h(?:ours?)?)?\s*"
    r"(?:(?P<m>\d+)\s*m(?:in(?:utes?)?)?)?\s*(?:(?P<s>\d+)\s*s(?:ec(?:onds?)?)?)?",
    re.IGNORECASE,
)
_CURA_TIME = re.compile(r"^;TIME:(?P<seconds>\d+)\s*$", re.MULTILINE)
_FILAMENT_MM = re.compile(
    r"^;\s*(?:total\s+)?filament used\s*\[mm\]\s*=\s*(?P<value>[0-9.]+)", re.MULTILINE | re.IGNORECASE
)
_FILAMENT_G = re.compile(
    r"^;\s*(?:total\s+)?filament used\s*\[g\]\s*=\s*(?P<value>[0-9.]+)", re.MULTILINE | re.IGNORECASE
)
_CURA_FILAMENT = re.compile(
    r"^;Filament used:\s*(?P<value>[0-9.]+)\s*m\s*$", re.MULTILINE | re.IGNORECASE
)
_LAYERS = re.compile(
    r"^;\s*total layers count\s*=\s*(?P<value>\d+)", re.MULTILINE | re.IGNORECASE
)
_CURA_LAYERS = re.compile(r"^;LAYER_COUNT:(?P<value>\d+)\s*$", re.MULTILINE)
_SUPPORT_FLAG = re.compile(
    r"^;\s*support_material\s*=\s*(?P<value>\S+)", re.MULTILINE | re.IGNORECASE
)
_CURA_SUPPORT = re.compile(
    r'"support_enable"\s*:\s*"?(?P<value>true|false)"?', re.IGNORECASE
)
_EMPTY_LAYER = re.compile(r"empty\s+layer", re.IGNORECASE)


def parse_slicer_output(text: str) -> dict:
    """Pull the evidence out of slicer output (G-code comments or console text).

    Missing fields stay `None`. The one default is `empty_layers: 0`, because slicers
    emit a warning line when a layer is empty and silence *is* the normal case — an
    `empty_layers` of 0 therefore means "no empty-layer warning was printed", which is
    as strong a claim as G-code supports.
    """
    parsed: dict = {
        "seconds_estimate": None,
        "filament_mm": None,
        "filament_g": None,
        "support_used": None,
        "layers": None,
        "empty_layers": 0,
    }
    if not text:
        return parsed

    match = _TIME_COMMENT.search(text)
    if match:
        parsed["seconds_estimate"] = _duration_seconds(match.group("value"))
    if parsed["seconds_estimate"] is None:
        cura_time = _CURA_TIME.search(text)
        if cura_time:
            parsed["seconds_estimate"] = int(cura_time.group("seconds"))

    match = _FILAMENT_MM.search(text)
    if match:
        parsed["filament_mm"] = float(match.group("value"))
    if parsed["filament_mm"] is None:
        cura_filament = _CURA_FILAMENT.search(text)
        if cura_filament:
            # Cura reports metres; the manifest is millimetres throughout.
            parsed["filament_mm"] = round(float(cura_filament.group("value")) * 1000.0, 3)
    match = _FILAMENT_G.search(text)
    if match:
        parsed["filament_g"] = float(match.group("value"))

    match = _LAYERS.search(text)
    if match:
        parsed["layers"] = int(match.group("value"))
    else:
        cura_layers = _CURA_LAYERS.search(text)
        if cura_layers:
            parsed["layers"] = int(cura_layers.group("value"))

    match = _SUPPORT_FLAG.search(text)
    if match:
        parsed["support_used"] = _as_bool(match.group("value"))
    if parsed["support_used"] is None:
        match = _CURA_SUPPORT.search(text)
        if match:
            parsed["support_used"] = match.group("value").lower() == "true"

    parsed["empty_layers"] = len(_EMPTY_LAYER.findall(text))
    return parsed


def _duration_seconds(value: str) -> int | None:
    """`1h 2m 3s` / `23m 45s` / `45s` / `2d 3h` -> seconds, or `None` if unreadable."""
    match = _TIME_DURATION.search(value.strip())
    if not match or not any(match.groupdict().values()):
        return None
    days = int(match.group("d") or 0)
    hours = int(match.group("h") or 0)
    minutes = int(match.group("m") or 0)
    seconds = int(match.group("s") or 0)
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def _as_bool(value: str) -> bool | None:
    """`1`/`0`, `true`/`false`, `nil`/`none` -> tri-state, never a guess."""
    lowered = str(value).strip().strip('"').lower()
    if lowered in ("1", "true", "yes", "on"):
        return True
    if lowered in ("0", "false", "no", "off"):
        return False
    return None


# --------------------------------------------------------------------------- #
# The dry run
# --------------------------------------------------------------------------- #
#: How many characters of the slicer's own message go into a problem line. Enough to
#: identify the failure, short enough that a manifest stays readable.
_EXCERPT = 300


def slice_mesh(
    stl_path: str | Path,
    out_dir: str | Path,
    *,
    profile: str | Path | None = None,
    timeout_s: int = 600,
    slicer: Slicer | None = None,
) -> SliceResult:
    """Slice one STL and return the evidence, never raising for a slicer's failure.

    A missing slicer raises `SlicerMissing` (the caller has nothing to report on); a
    slicer that ran and failed is a `SliceResult(ok=False)` with its own words in
    `problems`. That split is what lets `pah export --slice` write
    `printable: false, problems: ["no slicer ..."]` instead of crashing.

    `profile` is a slicer settings file: PrusaSlicer/OrcaSlicer load it with `--load`,
    CuraEngine with `-j`. Leaving it out means the slicer's defaults, which are a real
    choice — a G-code file sliced with defaults is evidence about *that* profile, and
    the manifest records only the backend, not a profile it never had.
    """
    chosen = slicer or find_slicer()
    stl = Path(stl_path)
    if not stl.exists():
        raise FileNotFoundError(f"no STL to slice at {stl}")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    gcode = out / f"{stl.stem}.gcode"

    argv = _command(chosen, stl, gcode, profile)
    try:
        completed = subprocess.run(  # noqa: S603 - argv is built here, not shell-interpolated
            argv,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
            check=False,
        )
    except FileNotFoundError as error:
        # The path was found and then vanished, or is not executable at all. Both mean
        # "this machine cannot slice", which is `SlicerMissing`, not a failed slice.
        raise SlicerMissing(f"cannot execute slicer {chosen.path!r}: {error}") from error
    except subprocess.TimeoutExpired:
        return SliceResult(
            ok=False,
            backend=chosen.backend,
            gcode_path=None,
            problems=(f"the slicer did not finish within {timeout_s}s and was killed",),
        )

    text = ""
    if gcode.exists():
        text = gcode.read_text("utf-8", errors="replace")
    else:
        # Some builds print the G-code to stdout when `--output` is ignored. Parsing
        # stdout costs nothing and must not be confused with *finding* a file, which is
        # why `gcode_path` stays `None` unless the file is really there.
        text = completed.stdout or ""

    parsed = parse_slicer_output(
        "\n".join(part for part in (text, completed.stdout or "", completed.stderr or "") if part)
    )
    problems = _problems(completed, gcode, parsed)
    return SliceResult(
        ok=not problems,
        backend=chosen.backend,
        gcode_path=str(gcode) if gcode.exists() and gcode.stat().st_size > 0 else None,
        seconds_estimate=parsed["seconds_estimate"],
        filament_mm=parsed["filament_mm"],
        filament_g=parsed["filament_g"],
        support_used=parsed["support_used"],
        layers=parsed["layers"],
        empty_layers=parsed["empty_layers"],
        problems=tuple(problems),
    )


def _command(slicer: Slicer, stl: Path, gcode: Path, profile: str | Path | None) -> list[str]:
    """The CLI invocation for one backend. One place, so a flag change is one edit."""
    prefix = slicer.command
    if slicer.backend == "curaengine":
        argv = [*prefix, "slice", "-l", str(stl), "-o", str(gcode)]
        if profile is not None:
            argv.extend(["-j", str(profile)])
        return argv
    # PrusaSlicer and OrcaSlicer share the PrusaSlicer CLI; `--export-gcode` is the
    # headless slicing switch and `--output` names the G-code file.
    argv = [*prefix, "--export-gcode", "--output", str(gcode)]
    if profile is not None:
        argv.extend(["--load", str(profile)])
    argv.append(str(stl))
    return argv


def _problems(completed: subprocess.CompletedProcess, gcode: Path, parsed: dict) -> list[str]:
    """Everything that stops the output from counting as slicing evidence.

    Each entry is a sentence a person can act on, and the required fields are required
    on purpose: the audit asks for time and material in the manifest (C4), so a slicer
    that reports neither has not produced the evidence the manifest cites.
    """
    problems: list[str] = []
    produced = gcode.exists() and gcode.stat().st_size > 0
    if completed.returncode != 0:
        problems.append(
            f"the slicer exited {completed.returncode}: {_excerpt(completed.stderr or completed.stdout)}"
        )
    if not produced:
        problems.append("the slicer produced no G-code")
    if parsed["seconds_estimate"] is None:
        problems.append("the output reports no estimated printing time")
    if parsed["filament_mm"] is None and parsed["filament_g"] is None:
        problems.append("the output reports no filament usage")
    if parsed["layers"] is None:
        problems.append("the output reports no layer count")
    if parsed["empty_layers"]:
        problems.append(f"the slicer reported {parsed['empty_layers']} empty layer(s)")
    # Slicer diagnostics are evidence too: an error line is not cancelled by a zero
    # exit code (`--export-gcode` has been observed to exit 0 after a repair warning).
    for line in (completed.stderr or "").splitlines():
        if re.search(r"(?i)\berror\b", line) and line.strip():
            problems.append(f"slicer said: {line.strip()[:200]}")
            break
    return problems


def _excerpt(text: str) -> str:
    """The last non-empty line of a slicer's message, bounded for a manifest."""
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    return (lines[-1] if lines else "(no message)")[:_EXCERPT]
