"""What each manufacturing process can actually make.

Why a table and not scattered constants
---------------------------------------
The printability verifier used to carry the process knowledge itself: one overhang
angle per process, and a single `min_wall_mm` from the design constraints that meant the
same thing for FDM and for CNC. That makes two different claims share one number — a
0.5 mm wall is unprintable on an FDM nozzle and ordinary in resin or in aluminium — and
the audit's P2-1 asks for the numbers to live where the process is named.

The table is the whole of it. `min_feature_mm` is the smallest free-standing feature
the process resolves (a peg, a rib); `tolerance_mm` is what a fit can be trusted to;
`supports` says whether a downward face needs support at all, which is why the overhang
angle is `None` for a process that has no such concept.

`bridge_mm` is a *span*, not an angle
-------------------------------------
Every additive process except resin can print a horizontal ceiling over a narrow
cavity without support: the first layer after the gap is extruded as a bridge, and
it holds because both ends are anchored. So "is this ceiling unsupported?" is two
questions, not one — how shallow is the face (the overhang angle) **and** how wide
is the opening it has to cross (the bridge span). The angle alone condemns a
0.3 mm-wide step inside a 6.4 mm bore, which any FDM machine prints as part of the
wall; measured 2026-11-02 on a three-part decomposition, 6.148 mm2 per part, and it
made every decomposed piece "unprintable" for a feature no operator can act on.

`bridge_mm` is therefore the widest unsupported span the process crosses without
support. `None` means the question does not arise: a powder bed already holds the
part up, and a subtractive process has no layers to sag.

The rule the verifier follows
-----------------------------
The effective wall floor is **the stricter of** the task's `FitConstraints.min_wall_mm`
("this part must have at least 1.2 mm") and the process's own floor ("this process
cannot make less than 0.4 mm"). A design that asks for less than the process can make is
not made more printable by asking; one that asks for more is held to its own word.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["PROCESS_PROFILES", "ProcessProfile", "effective_min_wall_mm", "profile_for"]


@dataclass(frozen=True)
class ProcessProfile:
    """One process's manufacturing envelope. Every number is millimetres or degrees."""

    name: str
    #: Thinnest wall the process makes reliably.
    min_wall_mm: float
    #: Smallest free-standing feature (peg diameter, rib width).
    min_feature_mm: float
    #: Steepest downward face that still prints without support, measured from the
    #: horizontal. `None` means the concept does not apply (subtractive processes).
    overhang_deg: float | None
    #: Widest unsupported **span** the process crosses without support, in mm: a
    #: horizontal ceiling over a cavity narrower than this is bridged by the first
    #: layer, so it is not an overhang. `None` = the question does not arise (powder
    #: bed holds the part; subtractive has no layers). Not an angle.
    bridge_mm: float | None
    layer_mm: float
    #: What a fit may be trusted to, per side.
    tolerance_mm: float
    #: Whether a downward face needs support at all.
    supports: bool
    note: str = ""


#: The five processes the design plane can name. Numbers are conservative hobby-shop
#: values, not machine specs: the point is that they *differ per process* and are
#: printed in the report, so a conclusion can be read against the profile it used.
PROCESS_PROFILES: dict[str, ProcessProfile] = {
    "fdm": ProcessProfile(
        name="fdm",
        min_wall_mm=1.2,
        min_feature_mm=1.2,
        overhang_deg=45.0,
        # 8 mm is a conservative hobby-shop bridge on a 0.4 mm nozzle: a 6.4 mm bore's
        # ceiling prints, a 10 mm span is where supports start to earn their keep.
        bridge_mm=8.0,
        layer_mm=0.2,
        tolerance_mm=0.2,
        supports=True,
        note="0.4 mm nozzle, three perimeters as the wall floor",
    ),
    "sla": ProcessProfile(
        name="sla",
        min_wall_mm=0.4,
        min_feature_mm=0.4,
        overhang_deg=30.0,
        # Resin does not bridge: a ceiling over any cavity is printed on supports or
        # it sags into the vat. Zero, not None — the check still runs, it just never
        # forgives a span.
        bridge_mm=0.0,
        layer_mm=0.05,
        tolerance_mm=0.1,
        supports=True,
        note="resin holds finer walls but droops on shallow overhangs",
    ),
    "sls": ProcessProfile(
        name="sls",
        min_wall_mm=0.7,
        min_feature_mm=0.7,
        overhang_deg=0.0,
        bridge_mm=None,
        layer_mm=0.1,
        tolerance_mm=0.15,
        supports=False,
        note="powder bed supports the part: no overhang limit",
    ),
    "mjf": ProcessProfile(
        name="mjf",
        min_wall_mm=0.9,
        min_feature_mm=0.9,
        overhang_deg=0.0,
        bridge_mm=None,
        layer_mm=0.08,
        tolerance_mm=0.15,
        supports=False,
        note="powder bed like SLS, but coarser walls",
    ),
    "cnc": ProcessProfile(
        name="cnc",
        min_wall_mm=0.8,
        min_feature_mm=1.0,
        overhang_deg=None,
        bridge_mm=None,
        layer_mm=0.0,
        tolerance_mm=0.05,
        supports=False,
        note="subtractive: no layers, no overhangs; internal corners need a tool radius",
    ),
}


def profile_for(name: str) -> ProcessProfile | None:
    """The profile for a process name, or `None` when the name is unknown.

    `None` is the capability boundary the audit asked for: a verifier that cannot find
    the profile must say *not evaluated* rather than fall back to FDM's numbers and
    pretend the conclusion applies to a process nobody described.
    """
    return PROCESS_PROFILES.get(str(name or "").strip().lower())


def effective_min_wall_mm(constraints, profile: ProcessProfile) -> float:
    """The stricter of what the design asks and what the process can make."""
    return max(float(constraints.min_wall_mm), float(profile.min_wall_mm))
