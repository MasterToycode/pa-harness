"""The adapter that turns a design-plane `Assembly` into evidence the loop can use.

`policy/loop.py:RepairLoop` knows exactly one thing about verification: it calls
`check(assembly)` and reads the `FailureCertificate`s that come back. Everything
else — building a view, materialising a spec that has no triangles yet, running
the registry, charging for the work — is this module's job. Three jobs, and each
one exists because leaving it out was a real failure mode:

1. **Make the triangles exist — and know what they mean.** A design-plane
   `Assembly` records `geom.hash` (from `design/api.py:geometry_hash`, spec-only)
   and possibly an empty `geom.path`, because operators are pure: they edit the
   spec and never build geometry (contract §2). The verifiers are forbidden to
   build anything — a verifier that could synthesise the geometry it judges would
   be marking its own homework. So the adapter materialises, once per distinct
   spec, and hands the view the solids plus the interface's `mate_axis` and
   `approach_mm`. Those two facts exist only in the builder's output: an assembly
   does not record them, so a view rebuilt from recorded STLs would measure a
   dovetail slide against a +Z axis and call it blocked. This is the only place in
   the design plane allowed to call the materialiser.
2. **Fail closed, per verifier.** A verifier that raises is not a pass and not a
   physical failure: it is *unknown*. The loop treats an `environment`-family
   certificate as such (it never counts one as repaired, and it never files one
   in a failure rate). So an exception becomes `registry.fail_closed_view(...)`
   and the cascade keeps going, or stops — `stop_on_hard` decides, and the
   contract's default is to keep going, because evidence collected after one
   broken tool is still evidence.
3. **Pay for it.** Verification costs geometry calls, and an "equal budget"
   comparison between two policies is only fair if the expensive tier is charged
   (contract §3). `spent` is a monotonically growing `Cost`; `RepairLoop` reads
   the delta between two checks and adds it to the round the checks made
   possible. A cascade that does not report would make the physics tier free, and
   `sim_calls` would stop meaning anything.

The §4.8 gate is *not* run here
-------------------------------
`registry.sanity()` is what disables a verifier whose own fixtures disagree with
it, and this adapter honours `registry.disabled()` — but it does not run the gate
itself, matching the frozen `DesignVerifierRegistry.check` (`design/api.py`), which
does not either. Running fixtures on every construction would make each episode
pay for the whole sanity pass, and it would let a test double that is supposed to
raise be disabled before the exception path could be exercised. The gate is a
step of its own: `pah design sanity` (contract §5) runs it once before a batch and
the disabled set travels with the registry.
"""

from __future__ import annotations

import importlib
from dataclasses import replace
from pathlib import Path

from pah.core.model import Assembly, Cost, FailureCertificate
from pah.design.api import (
    DESIGN_VERIFIER_IDS,
    DesignVerifierRegistry,
    DesignView,
    MaterializedDesign,
    Materializer,
    spec_of,
    view_of,
)

from .fit_clearance import FitClearance
from .insertion_path import InsertionPath
from .interface_interference import InterfaceInterference
from .printability import Printability

__all__ = ["DesignVerifierCascade", "default_registry"]

#: Each enabled verifier costs one call of its tier's currency, and the physics
#: tier is the only one paid in `sim_calls` (`VerifierSpec.cost_class`).
_ENVIRONMENT = "environment"

#: Verifiers that do not apply to a **decomposed** pair, with the reason a report
#: must show. One table, read by both the skip decision and the note, because the
#: audit (`docs/remediation-plan.md` P0-1) found the two drifting apart: the skip set
#: named both verifiers, only one of them wrote a note, and the missing one was then
#: counted as run — `validated: full` for a three-part run whose material check had
#: never executed.
_PAIR_SKIPS: dict[str, str] = {
    "insert_material": (
        "not evaluated: this check asks whether a *generated* insert is a solid "
        "rather than a shell, by comparing its volume to a parametric reference "
        "cylinder. A decomposed part has no such reference; running it would fail "
        "every pair with `excess_material`, which is a category error, not a finding"
    ),
    "physics_settle": (
        "not evaluated: the settle model assumes a peg-shaped insert in a bore. A "
        "decomposed part is a plate or a block, so the cavity guard does not describe "
        "the joint. The check that does — joint retention — is not built yet"
    ),
}


def default_registry() -> DesignVerifierRegistry:
    """Every design-plane verifier this checkout ships, registered and ungated.

    Registration is *not* proof: `sanity()` is the gate, and `pah design sanity`
    is what runs it. The order is `DESIGN_VERIFIER_IDS`, cheapest first, so the
    cascade's early-stop mode saves the tiers it is meant to save.
    """
    registry = DesignVerifierRegistry()
    registry.register(FitClearance())
    # Material before surfaces: a hollow shell has a perfect outer surface, so every
    # verifier below this one would agree that it fits.
    from pah.tools.verifiers.insert_material import InsertMaterial  # noqa: PLC0415

    registry.register(InsertMaterial())
    registry.register(InterfaceInterference())
    registry.register(InsertionPath())
    # Whole-assembly pass: the per-pair shadows cannot see two parts that share
    # no interface, and that is exactly the collision a decomposition produces.
    from pah.tools.verifiers.global_interference import GlobalInterference  # noqa: PLC0415

    registry.register(GlobalInterference())
    registry.register(Printability())
    physics = _optional_physics_settle()
    if physics is not None:
        registry.register(physics)
    return registry


def _optional_physics_settle():
    """The MuJoCo settle verifier, when this checkout has one.

    `physics_settle` is the expensive tier and ships separately, so a tree
    without it gets a four-verifier registry rather than an ImportError. Only a
    module that is genuinely absent is skipped: an ImportError raised *inside* a
    present `physics_settle` is a defect and must surface, which is what stops a
    typo from silently dropping the physics tier.
    """
    module_name = f"{__package__}.physics_settle"
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name != module_name:
            raise
        return None
    return module.PhysicsSettle()


class DesignVerifierCascade:
    """`LoopVerifiers` for the design plane: view, verify, charge, never lie.

    `root` is the episode directory used both to *find* geometry (the
    `root/geometry/<part_id>.stl` convention a materialiser writes,
    `design/api.py:DesignView`) and to write it when the assembly carries no
    paths yet. `materializer` defaults to `pah.design.materialize.materialize`,
    resolved lazily so this module imports without it; the parameter exists for
    the same reason the `Materializer` port does — a CAD kernel or a DiT adapter
    must be able to replace the trimesh builder without touching the loop.

    Known limitation: with `root=None` the cascade cannot materialise, so it
    reads the assembly's recorded STLs and the view falls back to the +Z / 6 mm
    insertion defaults. That is correct for every axial kind and wrong for a slide
    fit, so an episode that can carry a `root` should — `RepairLoop` and the
    runtime always do.
    """

    def __init__(
        self,
        registry: DesignVerifierRegistry,
        root: str | Path | None = None,
        *,
        stop_on_hard: bool = False,
        materializer: Materializer | None = None,
    ) -> None:
        self.registry = registry
        self.root = Path(root) if root is not None else None
        self.stop_on_hard = stop_on_hard
        self._materializer = materializer
        self._designs: dict[str, MaterializedDesign] = {}
        self._spent = Cost()
        self._notes: dict[str, str] = {}
        #: The last disassembly search, or `None` when the assembly was too small to
        #: need one (two parts are covered by `insertion_path`). The report reads it
        #: to say whether an order was found, which is a different claim from
        #: "nothing collides".
        self._sequence = None

    # -- the meter ---------------------------------------------------------
    @property
    def spent(self) -> Cost:
        """Everything this cascade has spent since construction.

        Monotone on purpose: `RepairLoop` charges the delta between two checks, so
        a meter that can go backwards would refund a round. Charging is per
        verifier *attempt* (a crash costs a call too) plus one per solid built by
        a materialisation; wall time is not metered because it is not
        deterministic, so `wall_s` stays 0 and a report compares policies on
        `geom_calls`/`sim_calls`.
        """
        return self._spent

    @property
    def sequence(self):
        """The last disassembly search (`SequenceResult`), or `None`.

        Kept beside `notes` rather than folded into them: a *found* order is a
        positive fact the report has to print, and `notes` only carries what could
        not be evaluated.
        """
        return self._sequence

    @property
    def notes(self) -> dict[str, str]:
        """What the last `check` could not evaluate, one line per verifier.

        A capability boundary is not a tool failure and not a pass. A verifier that
        declined to answer — the physics tier handed a non-convex insert it cannot
        collide, the printability tier handed no print orientation — records it in
        its own `last_note` instead of emitting a certificate, because an
        `environment`-family certificate would make the loop stop as if the tool
        were broken. The report prints these as `not evaluated`, which is a
        different claim from both "healthy" and "failed".
        """
        return dict(self._notes)

    # -- `LoopVerifiers` ---------------------------------------------------
    def check(self, assembly: Assembly) -> list[FailureCertificate]:
        """The evidence for `assembly`, cheapest verifier first.

        Two shapes, one entry point. A two-part, one-interface design goes through the
        original single-view path unchanged. A decomposition (N parts, or any part
        carrying a `source` mesh) is judged **one interface at a time**, each pair in
        its own two-part shadow (`design/pairs.py`) with its own insertion axis — the
        verifiers never learn that the assembly has more than two pieces, which is
        why six pieces of physics did not have to be rewritten for N parts.
        """
        self._notes = {}
        try:
            spec = spec_of(assembly)
        except Exception as error:
            # No spec, so no locus to name; "we could not describe what to look at"
            # is unknown, not healthy.
            return [self._environment(assembly, error)]

        if len(spec.interfaces) > 1 or any(part.source for part in spec.parts):
            return self._check_per_pair(assembly, spec)

        try:
            view = self._view(assembly)
        except Exception as error:
            # No view, so no spec to name in the locus; the failure happened
            # before any verifier ran. Still an environment certificate, because
            # "we could not build the thing to look at" is unknown, not healthy.
            return [self._environment(assembly, error)]

        certificates: list[FailureCertificate] = []
        notes: dict[str, str] = {}
        # A two-part assembly has exactly one pair, and `interface_interference`
        # already boolean-tests exactly those two solids at the same noise floor. The
        # whole-assembly pass would re-report that one pair with a second code, so it
        # is skipped here — silently, because writing "not evaluated" would be false:
        # the check *was* done, by the verifier that owns the pair.
        self._run_registry(view, certificates, notes, silent_skip={"global_interference"})
        self._notes = notes
        return certificates

    def _check_per_pair(
        self, assembly: Assembly, spec
    ) -> list[FailureCertificate]:
        """Every interface judged in its own two-part shadow, tagged by interface."""
        from pah.design.pairs import pair_view  # noqa: PLC0415

        try:
            design = self._materialise(spec)
        except Exception as error:
            return [self._environment(assembly, error)]

        certificates: list[FailureCertificate] = []
        notes: dict[str, str] = {}
        for interface in spec.interfaces:
            pair = pair_view(design, interface.id)
            # The shadow holds two parts, so a global pass inside it would be both a
            # duplicate of the adjacent pair and *blind* to the pairs that matter:
            # two parts that share no interface never appear in one shadow together.
            self._run_registry(
                pair.view,
                certificates,
                notes,
                tag=interface.id,
                skip=_PAIR_SKIPS,
                silent_skip={"global_interference"},
            )

        # The whole assembly, once. This is the pass the audit asked for: every pair,
        # including the ones no interface connects.
        whole = DesignView(
            spec=spec,
            root=design.root,
            meshes=dict(design.meshes),
            paths=dict(design.paths),
        )
        self._run_registry(
            whole,
            certificates,
            notes,
            only={"global_interference"},
            skip=_PAIR_SKIPS,
        )
        self._check_sequence(spec, design, certificates, notes)
        self._notes = notes
        return certificates

    def _check_sequence(self, spec, design, certificates, notes) -> None:
        """Can the assembly actually be taken apart one straight pull at a time?

        A `found=False` result is a **heuristic** failure, not a physical one: the
        search only tries straight-line pulls, so "no order found" is "we could not
        show it", never "impossible". It therefore lands in `notes` (which the report
        renders as not-evaluated) rather than as a blocking certificate — the audit's
        requirement for P1-1, and the same three-state discipline the rest of the
        harness uses.

        Two parts are skipped: the single joint's own `insertion_path` check is the
        same statement, measured on the actual pair, and inventing a second answer for
        it would only give the report two ways to disagree.
        """
        self._sequence = None
        if len(spec.parts) < 3:
            return
        from pah.design.assembly_sequence import find_sequence  # noqa: PLC0415

        try:
            result = find_sequence(spec, design.meshes)
        except Exception as error:  # noqa: BLE001 - fail closed, like the verifiers
            certificates.append(self._environment(design.assembly, error))
            return
        self._sequence = result
        if not result.found:
            notes["assembly_sequence"] = result.reason

    def _run_registry(
        self,
        view: DesignView,
        certificates: list[FailureCertificate],
        notes: dict[str, str],
        *,
        tag: str | None = None,
        skip: dict[str, str] | None = None,
        silent_skip: set[str] | None = None,
        only: set[str] | None = None,
    ) -> None:
        """Run every enabled verifier against `view`, appending what it found.

        `tag` names the interface a certificate belongs to. Without it, an N-part
        report says "clearance out of band" and a reader cannot tell which of four
        joints needs the repair — and the repair operators take an interface id.

        **Every verifier that does not run gets a note, from the same table that
        skipped it.** Three ways a check can be absent — a skipped pair check, a
        verifier its own sanity gate disabled, and (further down) an environment
        certificate — and all three must read as *not evaluated* downstream. The
        audit that produced `docs/remediation-plan.md` found the first two silently
        counted as run, so a three-part run reported `validated: full` while
        `insert_material` had never executed.

        `silent_skip` is the one exception, and it is deliberately narrow: a check
        listed there was *done* by another verifier on the same solids (the whole-
        assembly pass is subsumed by the pair check for a two-part design), so a note
        would be a false "not evaluated". `only` restricts the pass to named
        verifiers, for the one whole-assembly call.
        """
        skip = skip or {}
        silent = silent_skip or set()
        for verifier_id in self.registry.ids():
            if only is not None and verifier_id not in only:
                continue
            key = f"{tag}/{verifier_id}" if tag else verifier_id
            if verifier_id in silent:
                continue
            if verifier_id in self.registry.disabled():
                why = self.registry.why_disabled(verifier_id)
                notes[key] = (
                    "not evaluated: the sanity gate disabled this verifier"
                    + (f" ({why})" if why else "")
                )
                continue
            if verifier_id in skip:
                notes[key] = skip[verifier_id]
                continue
            verifier = self.registry.get(verifier_id)
            try:
                found = list(verifier.check(view))
            except Exception as error:
                self._charge(verifier, ran=False)
                certificates.append(
                    _tagged(self.registry.fail_closed_view(view, error), tag)
                )
                if self.stop_on_hard:
                    break
                continue
            self._charge(verifier, ran=True)
            note = str(getattr(verifier, "last_note", "") or "")
            if note:
                notes[key] = note
            certificates.extend(_tagged(cert, tag) for cert in found)
            if self.stop_on_hard and any(cert.family != _ENVIRONMENT for cert in found):
                break
        self._notes = notes
        return certificates

    # -- view --------------------------------------------------------------
    def _view(self, assembly: Assembly) -> DesignView:
        """The design view: built if this cascade already built it, else read, else built.

        Order matters for two different reasons. A design this cascade materialised
        is reused first: it is the only source that is guaranteed to carry the
        builder's `mate_axis`/`approach_mm` and the solids the last check saw.
        After that, `view_of` is preferred over rebuilding, because it is cheaper
        and it recovers the insertion facts from the assembly's own constraints
        (`design/api.py:view_of`), which is what a resumed episode relies on.
        Materialising is the fallback: when the recorded triangles cannot be read
        there is nothing to verify, and the spec is the document of record.
        """
        spec = spec_of(assembly)
        known = self._designs.get(spec.content_key())
        if known is not None:
            return known.view()
        # An empty `geom.path` means "this design was never materialised" (contract
        # §2). It must not be answered by whatever STL happens to sit at
        # `root/geometry/<part_id>.stl`: that file is the *previous* design's
        # triangles, and judging a repaired snapshot against them made every
        # repair look like it had changed nothing. Measured 2026-11-01: round 0
        # with `set_clearance(0.35)` still reported the old interference, so the
        # `pah run --policy rule` demo never left `exhausted`.
        unbuilt = any(
            not str((part.get("geom") or {}).get("path") or "") for part in assembly.parts
        )
        if unbuilt:
            return self._materialise(spec).view()
        view = view_of(assembly, self.root)
        try:
            for part_id in view.part_ids():
                view.mesh(part_id)
        except Exception:
            # A recorded path that cannot be read — a deleted or corrupt STL. The
            # spec is the document of record, so rebuild from it.
            return self._materialise(spec).view()
        return view

    def _materialise(self, spec) -> MaterializedDesign:
        """`spec` built into `root`, cached by content, charged per solid."""
        key = spec.content_key()
        cached = self._designs.get(key)
        if cached is not None:
            return cached
        if self.root is None:
            raise RuntimeError(
                f"assembly for spec {spec.spec_id!r} carries no geometry paths and "
                "this cascade was built without a root to materialise into"
            )
        materialize = self._materializer or _default_materializer()
        design = materialize(spec, self.root)
        self._designs[key] = design
        # One geometry call per materialisation, however many parts it built: the
        # charge is for "the adapter had to touch geometry", which is what makes an
        # expensive cascade visible against a cheap one (contract §3).
        self._spent = _plus(self._spent, geom=1)
        return design

    # -- internals ---------------------------------------------------------
    def _charge(self, verifier, *, ran: bool) -> None:
        """Bill one attempted verifier call to the currency of its tier.

        A cheap or medium verifier is one geometry call. The expensive tier is
        billed what it says it spent — `last_sim_calls` — because a physics
        verifier that refused the case (a mesh it cannot collide, a missing MuJoCo)
        ran no simulation, and charging a full `sim_calls` for that would make a
        broken tool look as expensive as a working one while hiding the real cost
        of the ones that did run. A verifier that raised cannot be asked, so its
        attempt is billed one call.
        """
        if verifier.spec.cost_class != "expensive":
            self._spent = _plus(self._spent, geom=1)
            return
        calls = 1
        if ran:
            reported = getattr(verifier, "last_sim_calls", None)
            if isinstance(reported, int) and reported >= 0:
                calls = reported
        if calls:
            self._spent = _plus(self._spent, sim=calls)

    def _environment(self, assembly: Assembly, error: BaseException) -> FailureCertificate:
        """The certificate for a failure that happened before a view existed.

        Deliberately the same shape as
        `design/api.py:DesignVerifierRegistry.fail_closed_view`, spelled out here
        because that helper needs a view and this failure is precisely the case
        where there is none. Recorded so the report can tell *which* assembly
        could not be looked at.
        """
        return FailureCertificate(
            code="verifier_error",
            family=_ENVIRONMENT,
            locus={"assembly": assembly.id, "version": assembly.version},
            magnitude={"error": f"{type(error).__name__}: {error}"},
            observed_by="design_cascade",
        )


def _default_materializer() -> Materializer:
    """T1's materialiser, imported on first use.

    Lazy because the design plane must import (and its verifiers be gateable)
    before the geometry builder exists; a plain module-level import would make
    every verifier test depend on it.
    """
    from pah.design.materialize import materialize

    return materialize


def _tagged(cert: FailureCertificate, tag: str | None) -> FailureCertificate:
    """`cert` with the interface it came from written into its locus."""
    if not tag:
        return cert
    locus = dict(cert.locus or {})
    locus["interface"] = tag
    return replace(cert, locus=locus)


def _plus(base: Cost, *, geom: int = 0, sim: int = 0) -> Cost:
    return Cost(
        geom_calls=base.geom_calls + geom,
        sim_calls=base.sim_calls + sim,
        wall_s=base.wall_s,
        llm_calls=base.llm_calls,
        tokens=base.tokens,
    )

