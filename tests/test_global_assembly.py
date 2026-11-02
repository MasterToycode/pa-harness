"""The global interference verifier and the assembly-sequence search.

What these tests are for
------------------------
The 2026-11-02 audit found two blind spots in the N-part path, and each test here is
the evidence that the hole is covered rather than a claim that it is:

* **A pair no interface joins is invisible to the pairwise cascade.** Three parts in
  an L: `part_a` and `part_c` overlap by 3 mm (600 mm3) while both declared joints are
  clean. Every two-part shadow the pairwise path can build is clear, and the global
  verifier is the only thing that reports the overlap — that is the audit's A1
  counter-example, with the numbers it produces.
* **A clear seat is not a clear order.** A block sealed inside a closed shell does not
  interfere with anything, and has no straight-line way out; `find_sequence` must say
  "not found" and must say in words that a failed search is not a proof.
* **Both modules carry their own proof.** The verifier ships a `known_good` /
  `known_bad` pair built from `trimesh` primitives (never by the materialiser, which
  would be marking its own homework), and the search is run on a chain the *real*
  pipeline built — decompose, pair, machine — so a success there is about the
  geometry this project actually produces.

The geometry is deliberately small: three slabs of a 60 x 40 x 12 plate, two
40 x 40 x 10 plates, one 40 mm shell. The boolean count grows with the square of the
part count, and a fixture that takes a minute to build is a fixture that stops being
run.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import trimesh
from trimesh.creation import box

from pah.design.api import DESIGN_CODES, DesignVerifierRegistry, DesignView
from pah.design.assembly_sequence import find_sequence
from pah.design.decompose import (
    file_hash,
    parts_from_decomposition,
    split_slabs,
    store_parts,
)
from pah.design.materialize import materialize
from pah.design.pairing import propose_interfaces, spec_from_parts
from pah.design.pairs import pair_spec
from pah.design.spec import DesignSpec, FitConstraints, InterfaceSpec, PartSpec
from pah.tools.verifiers import _design_fixtures as design_fixtures
from pah.tools.verifiers.global_interference import (
    CODE_GLOBAL_INTERFERENCE,
    GlobalInterference,
    register,
)
from pah.tools.verifiers.interface_interference import InterfaceInterference

#: The blank the three-part chain is cut from: the same plate the two-part acceptance
#: case uses, so a regression here shows up as a difference from a known solid.
PLATE = (60.0, 40.0, 12.0)

#: A nominal connector for the hand-built specs below. The sequence search reads the
#: parts' envelopes and the interface graph, never these numbers; a fixture that
#: declared a fit it did not build would be inviting a reader to check the wrong thing.
_PEG = {
    "d_peg_mm": 5.8,
    "d_hole_mm": 6.0,
    "engage_mm": 3.0,
    "approach_mm": 6.0,
}


# --------------------------------------------------------------------------- #
# The A1 counter-example: an overlap no interface can see
# --------------------------------------------------------------------------- #
#: Three boxes in an L. `part_a` is the foot, `part_b` sits on it, `part_c` is beside
#: `part_b`. `part_c` is 3 mm too low: it bites into `part_a`, which no interface joins
#: to it, while both declared joints (`a`-`b`, `b`-`c`) touch along a face with zero
#: shared volume.
_L_PARTS = (
    ("part_a", (40.0, 20.0, 10.0), (0.0, 0.0, 0.0)),
    ("part_b", (20.0, 20.0, 10.0), (0.0, 0.0, 10.0)),
    ("part_c", (20.0, 20.0, 10.0), (20.0, 0.0, 7.0)),
)
#: The overlap the audit report asks for: 3 mm deep over a 10 x 20 mm patch, i.e.
#: 600 mm3 — measured, not assumed (see the test's assertion).
_L_OVERLAP_MM3 = 600.0


def l_assembly() -> tuple[DesignSpec, dict[str, trimesh.Trimesh]]:
    """The L, as a spec plus the world-frame solids its poses describe."""
    spec = DesignSpec(
        spec_id="l-blind-spot",
        requirement="three boxes in an L: two clear joints, one 3 mm hidden overlap",
        parts=tuple(
            PartSpec(
                id=part_id,
                role="body" if part_id == "part_a" else "part",
                size_mm=size_mm,
                pose_mm=pose_mm,
            )
            for part_id, size_mm, pose_mm in _L_PARTS
        ),
        interfaces=(
            InterfaceSpec(
                id="iface_a_b", kind="peg_hole", a="part_a", b="part_b", params=_PEG
            ),
            InterfaceSpec(
                id="iface_b_c", kind="peg_hole", a="part_b", b="part_c", params=_PEG
            ),
        ),
    )
    meshes: dict[str, trimesh.Trimesh] = {}
    for part_id, size_mm, pose_mm in _L_PARTS:
        mesh = box(extents=size_mm)
        mesh.apply_translation(pose_mm)
        meshes[part_id] = mesh
    return spec, meshes


def test_a_pair_no_interface_joins_is_invisible_to_the_pairwise_check():
    """The audit's A1: what the per-joint cascade cannot see, and why.

    Two-part shadows are built per interface, so the pair (`part_a`, `part_c`) never
    appears in one — not because those two are far apart, but because no interface
    joins them. The claim is proven twice over: every shadow the pairwise path can
    build is clean, and the global check reports the 600 mm3 the pairwise path cannot.
    """
    spec, meshes = l_assembly()

    # The structural reason the blind spot exists: no shadow ever holds both parts.
    shadows = [pair_spec(spec, interface.id) for interface in spec.interfaces]
    assert len(shadows) == 2
    assert not any(
        {"part_a", "part_c"} <= {part.id for part in shadow.parts} for shadow in shadows
    )
    # ... and the pairwise interference verifier, run on every shadow it can build,
    # finds nothing wrong: both joints are seated with zero shared volume.
    for shadow in shadows:
        pair_view = DesignView(
            spec=shadow,
            root=None,
            meshes={part.id: meshes[part.id] for part in shadow.parts},
        )
        assert InterfaceInterference().check(pair_view) == []

    certificates = GlobalInterference().check(
        DesignView(spec=spec, root=None, meshes=dict(meshes))
    )

    assert [certificate.code for certificate in certificates] == [
        CODE_GLOBAL_INTERFERENCE
    ]
    certificate = certificates[0]
    assert certificate.family == "assembly"
    assert certificate.observed_by == "global_interference"
    assert certificate.locus["parts"] == ["part_a", "part_c"]
    # No interface to name: this is the pair the joint-level machinery cannot reach.
    assert "interface" not in certificate.locus
    assert certificate.magnitude["volume_mm3"] == pytest.approx(_L_OVERLAP_MM3, rel=1e-6)


# --------------------------------------------------------------------------- #
# The clean case, on geometry the real pipeline built
# --------------------------------------------------------------------------- #
def three_slab_chain(tmp_path: Path) -> tuple[DesignSpec, object]:
    """A 60 x 40 x 12 plate cut into three slabs, pegged back together for real.

    Decompose -> pair -> machine -> materialise, with no shortcuts: the solids come
    back with sockets cut and pegs grown, so a clean verdict here is a statement about
    the geometry this project ships rather than about boxes drawn by the test.
    """
    decomposition = split_slabs(box(extents=PLATE), count=3, axis=2)
    out = tmp_path / "run"
    stored = store_parts(decomposition.parts, out)
    parts = parts_from_decomposition(
        decomposition,
        source_hashes={key: file_hash(path) for key, path in stored.items()},
    )
    spec = spec_from_parts(
        spec_id="three-slab-chain",
        parts=parts,
        interfaces=propose_interfaces(parts).interfaces,
        # No print orientation declared, for the reason the CLI gives: the middle slab
        # carries a peg pointing down, so the assembly pose is never the pose a person
        # prints it in, and declaring one would manufacture an overhang finding about a
        # decision nobody made.
        constraints=FitConstraints(),
    )
    return spec, materialize(spec, out)


def test_a_clean_three_part_chain_earns_no_global_certificate(tmp_path):
    spec, design = three_slab_chain(tmp_path)

    # a chain, so the verifier really had pairs to look at
    assert len(spec.interfaces) == 2
    assert GlobalInterference().check(design.view()) == []


# --------------------------------------------------------------------------- #
# Assembly order
# --------------------------------------------------------------------------- #
def test_a_straight_chain_has_a_disassembly_order_whose_reverse_is_the_assembly(
    tmp_path,
):
    spec, design = three_slab_chain(tmp_path)

    result = find_sequence(spec, design.meshes)

    assert result.found is True, result.reason
    assert sorted(result.order) == ["part_00", "part_01", "part_02"]
    assert result.blocked_at is None
    assert result.remaining == ()
    # Two of the three parts had to be freed by a move; the last one has nothing left
    # to collide with and costs none.
    assert result.attempts >= 2
    # A straight stack can only go together from one end: the middle slab carries a
    # peg on one face and a socket on the other, so the two valid build orders are the
    # two ends-first ones. Anything else would be claimed by the search and false.
    assert result.assembly_order in (
        ("part_00", "part_01", "part_02"),
        ("part_02", "part_01", "part_00"),
    )
    assert "straight-line disassembly found" in result.reason
    assert "only pulls along an interface normal are modelled" in result.reason


def sealed_assembly() -> tuple[DesignSpec, dict[str, trimesh.Trimesh]]:
    """A 20 mm block inside a closed 40 mm shell, 0.5 mm clear on every side.

    The two solids do not share a cubic millimetre, so no interference verifier has
    anything to say — and the block cannot be pulled out in any direction, which is a
    fact no seat-level check can express.
    """
    shell = trimesh.boolean.difference(
        [box(extents=(40.0, 40.0, 40.0)), box(extents=(21.0, 21.0, 21.0))],
        engine="manifold",
    )
    spec = DesignSpec(
        spec_id="sealed-block",
        requirement="a 20 mm block inside a closed 40 mm shell, 0.5 mm clearance",
        parts=(
            PartSpec(id="inner_block", role="insert", size_mm=(20.0, 20.0, 20.0)),
            PartSpec(id="outer_shell", role="body", size_mm=(40.0, 40.0, 40.0)),
        ),
        # The spec needs an interface for its graph to connect; the search only reads
        # its normal, and both signs of it are tried.
        interfaces=(
            InterfaceSpec(
                id="iface_shell_block",
                kind="peg_hole",
                a="outer_shell",
                b="inner_block",
                params=_PEG,
            ),
        ),
    )
    return spec, {
        "inner_block": box(extents=(20.0, 20.0, 20.0)),
        "outer_shell": shell,
    }


def test_a_part_sealed_inside_a_closed_shell_is_reported_as_not_found():
    """The distinction the whole module exists for: "not found" is not "impossible"."""
    spec, meshes = sealed_assembly()

    # Nothing interferes, so no other verifier has a finding here: the assembly is
    # clear at rest and still cannot be put together.
    assert GlobalInterference().check(DesignView(spec=spec, meshes=dict(meshes))) == []

    result = find_sequence(spec, meshes)

    assert result.found is False
    assert result.order == ()
    assert result.attempts >= 1
    assert result.blocked_at in {"inner_block", "outer_shell"}
    assert set(result.remaining) == {"inner_block", "outer_shell"}
    # The words carry the claim: a failed heuristic search must never read as
    # "cannot be assembled".
    assert "no straight-line disassembly order found" in result.reason
    assert "not a proof of impossibility" in result.reason


def test_a_cut_short_search_says_the_budget_ran_out():
    """An exhausted move budget proves nothing, and the reason has to say which."""
    spec, meshes = sealed_assembly()

    result = find_sequence(spec, meshes, max_attempts=1)

    assert result.found is False
    assert result.attempts >= 1
    assert "cut short" in result.reason
    assert "not a proof of impossibility" in result.reason


def test_the_search_refuses_a_mesh_set_the_spec_does_not_describe():
    """A sweep over the wrong solid answers a question nobody asked."""
    spec, _ = l_assembly()

    with pytest.raises(KeyError, match="no mesh for part"):
        find_sequence(spec, {})


# --------------------------------------------------------------------------- #
# The §4.8 gate: the verifier against its own fixtures
# --------------------------------------------------------------------------- #
def test_the_verifier_passes_its_own_fixture_pair():
    """`known_good` clean, `known_bad` failing, and differing in one number only."""
    good, bad = GlobalInterference().fixtures()

    assert len(good) == 1 and len(bad) == 1
    assert GlobalInterference().check(good[0]) == []

    certificates = GlobalInterference().check(bad[0])
    assert [certificate.code for certificate in certificates] == [
        CODE_GLOBAL_INTERFERENCE
    ]
    # The overlap is the 3 mm the suite is about: 40 x 40 x 3 mm3, no sliver.
    assert certificates[0].magnitude["volume_mm3"] == pytest.approx(4800.0, rel=1e-6)
    # The one number the two fixtures disagree about, checked against the fixtures
    # themselves: equal envelopes, different poses.
    assert good[0].spec.parts[0].size_mm == bad[0].spec.parts[0].size_mm
    assert (
        good[0].spec.parts[1].pose_mm[2] - bad[0].spec.parts[1].pose_mm[2]
        == pytest.approx(8.0)
    )


def test_an_adjacent_seat_that_interferes_is_reported_with_its_interface():
    """The deliberate duplicate, and why leaving it out would be a false claim.

    `interface_interference` already reports this defect as `geometric_interference`.
    This verifier reports it again, with the interface id in its locus so a report can
    dedupe: skipping adjacent pairs would let "no global interference" be true while
    two solids occupy the same space at a seat.
    """
    view = design_fixtures.oversized_fixture()

    certificates = GlobalInterference().check(view)

    assert [certificate.code for certificate in certificates] == [
        CODE_GLOBAL_INTERFERENCE
    ]
    assert certificates[0].locus["interface"] == view.spec.interface.id
    joined = [view.spec.interface.a, view.spec.interface.b]
    assert certificates[0].locus["parts"] == joined
    # The same two solids, the same defect, from the joint's own point of view.
    assert InterfaceInterference().check(view)


def test_an_open_shell_is_unknown_rather_than_clear():
    """A boolean on an open shell has no volume. Raising lets the cascade file
    `environment`; returning [] would read as "every pair is clear"."""
    spec, meshes = l_assembly()
    opened = box(extents=(20.0, 20.0, 10.0))
    # Drop one face: the bounds stay put, so it is genuinely compared against part_a,
    # and the shell stops being closed.
    opened.update_faces([index != 0 for index in range(len(opened.faces))])
    opened.apply_translation((20.0, 0.0, 7.0))
    assert not opened.is_watertight
    meshes["part_c"] = opened

    with pytest.raises(ValueError, match="not watertight"):
        GlobalInterference().check(DesignView(spec=spec, root=None, meshes=dict(meshes)))


def test_the_verifier_joins_the_registry_once_the_contract_declares_its_code():
    """The wiring is two steps, and the order matters: declare the code, then register.

    Written to hold on either side of the contract edit, because the module lands
    before `pah.design.api.DESIGN_CODES` names the code: the registry's refusal is
    itself the property worth pinning (a verifier may not invent a code), and once the
    code is declared the same call must register *and* pass the fixture gate.
    """
    registry = DesignVerifierRegistry()

    if CODE_GLOBAL_INTERFERENCE not in DESIGN_CODES:
        with pytest.raises(ValueError, match="add it to DESIGN_CODES first"):
            register(registry)
        return

    register(registry)
    assert CODE_GLOBAL_INTERFERENCE in registry.ids()
    assert registry.sanity()[CODE_GLOBAL_INTERFERENCE] is True
