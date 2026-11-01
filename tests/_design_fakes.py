"""Contract-faithful stand-ins for the design plane's three teammates (T1/T2/T3).

Why hand-written doubles instead of `unittest.mock`
---------------------------------------------------
The runtime is composed from four *protocols*: a `Materializer`, a `LoopVerifiers`
with a `spent` meter, a `LoopRegistry` and a `LoopPolicy`. A mock that patches method
names would keep passing after the runtime called the wrong protocol; these doubles
implement the contract's own names (`docs/contracts/v2-interfaces.md` §2) so a runtime
change that stops satisfying it breaks the test instead of the demo.

What they deliberately do *not* do
----------------------------------
* `FakeMaterializer` builds real STL files, because the runtime's report asserts they
  exist, but the geometry is two boxes — no connector is modelled. The verifiers here
  look at the *spec*, exactly like `fit_clearance` does, so the loop still sees a
  repair that changes evidence for a reason.
* `FakeCascade` materialises on demand when `geom.path` is empty, like T2's, and keeps
  the same `spent` contract: one geometry call per verifier attempt plus one per solid
  built. A runtime that stopped hand-ing the cascade a pure assembly would still pass,
  but a cascade that never materialised would leave `out/geometry` empty and the run
  would exit 1 — which is the property the CLI tests need.
* `FakeOperators` edits the spec and returns a pure `Assembly`, never touching disk:
  that is the contract the real operators follow, and a double that wrote files would
  hide a runtime that wrongly expected them to.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import trimesh

from pah.core.model import Cost, FailureCertificate, RepairPlan
from pah.design.api import (
    DESIGN_CODES,
    MaterializedDesign,
    assembly_from_spec,
    geometry_hash,
    spec_of,
)
from pah.design.spec import DesignSpec, FitConstraints, InterfaceSpec, PartSpec
from pah.tools.verifiers.base import VerifierSpec

__all__ = [
    "BandCascade",
    "ClearancePolicy",
    "FakeDesignRegistry",
    "FakeMaterializer",
    "FakeOperators",
    "design_spec",
    "install",
    "plus",
]


def plus(base: Cost, *, geom: int = 0, sim: int = 0) -> Cost:
    """`Cost` addition, the same shape `policy/loop.py` uses for its meter."""
    return Cost(
        geom_calls=base.geom_calls + geom,
        sim_calls=base.sim_calls + sim,
        wall_s=base.wall_s,
        llm_calls=base.llm_calls,
        tokens=base.tokens,
    )


def design_spec(
    *,
    clearance_mm: float = 0.0,
    band: tuple[float, float] = (0.20, 0.60),
    spec_id: str = "fake-design",
) -> DesignSpec:
    """A valid peg-hole spec whose designed clearance is exactly `clearance_mm`.

    Clearance is derived (`(d_hole - d_peg)/2`), so the diameter is what the helper
    sets; a test that reads `spec.clearance_mm` gets the number it asked for.
    """
    d_peg = 10.0
    body = PartSpec(id="body", role="body", size_mm=(40.0, 30.0, 10.0))
    insert = PartSpec(
        id="insert", role="insert", size_mm=(12.0, 12.0, 18.0), pose_mm=(0.0, 0.0, -2.0)
    )
    return DesignSpec(
        spec_id=spec_id,
        requirement="a fake peg that exists only to exercise the runtime",
        parts=(body, insert),
        interfaces=(
            InterfaceSpec(
                id="iface_0",
                kind="peg_hole",
                a="body",
                b="insert",
                params={
                    "d_peg_mm": d_peg,
                    "d_hole_mm": d_peg + 2.0 * clearance_mm,
                    "engage_mm": 8.0,
                    "approach_mm": 6.0,
                },
            ),
        ),
        constraints=FitConstraints(min_clearance_mm=band[0], max_clearance_mm=band[1]),
        provenance={"via": "tests/_design_fakes.py"},
    )


class FakeMaterializer:
    """T1's port: two boxes written as STLs, hashes from `geometry_hash`."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, spec: DesignSpec, out_dir: Path) -> MaterializedDesign:
        root = Path(out_dir)
        self.calls += 1
        geometry = root / "geometry"
        geometry.mkdir(parents=True, exist_ok=True)
        meshes: dict[str, trimesh.Trimesh] = {}
        paths: dict[str, str] = {}
        hashes: dict[str, str] = {}
        for part in spec.parts:
            mesh = trimesh.creation.box(extents=list(part.size_mm))
            target = geometry / f"{part.id}.stl"
            target.write_bytes(mesh.export(file_type="stl"))
            meshes[part.id] = mesh
            paths[part.id] = str(target)
            hashes[part.id] = geometry_hash(spec, part.id)
        (root / "spec.json").write_text(json.dumps(spec.model_dump(), indent=2), "utf-8")
        assembly = assembly_from_spec(
            spec=spec, geometry_paths=paths, geometry_hashes=hashes, version=1
        )
        return MaterializedDesign(
            spec=spec, assembly=assembly, meshes=meshes, paths=paths, root=root
        )


def _certificate(code: str, spec: DesignSpec, clearance_mm: float) -> FailureCertificate:
    return FailureCertificate(
        code=code,
        family="geometry",
        locus={"spec": spec.spec_id, "revision": spec.revision},
        magnitude={"clearance_mm": round(clearance_mm, 6)},
        observed_by="fake_fit",
    )


class BandCascade:
    """T2's adapter: materialise when needed, then judge the clearance against the band.

    `mode="band"` tells the truth about the spec; `mode="always_fail"` is a liar that
    exists only to exhaust a budget without cycling, so a budget test can observe
    exactly `rounds` rounds instead of an early cycle stop.
    """

    def __init__(
        self,
        registry: Any,
        root: str | Path,
        *,
        mode: str = "band",
        materializer: FakeMaterializer | None = None,
    ) -> None:
        self.registry = registry
        self.root = Path(root)
        self.mode = mode
        self._materializer = materializer or FakeMaterializer()
        self._designs: dict[str, MaterializedDesign] = {}
        self._spent = Cost()
        self.calls = 0

    @property
    def spent(self) -> Cost:
        """Monotone, like T2's meter: the loop charges the delta between checks."""
        return self._spent

    def check(self, assembly: Any) -> list[FailureCertificate]:
        spec = spec_of(assembly)
        self.calls += 1
        if self.mode == "crash":
            # What a dead external service looks like from the cascade's side: the
            # materialiser itself refuses, so there is nothing to look at.
            raise RuntimeError(
                "hunyuan3d SubmitHunyuanTo3DRapidJob 调用失败 [ResourceInsufficient]: "
                "资源包积分已用尽，请至控制台开通后付费"
            )
        cached = self._designs.get(spec.content_key())
        if cached is None:
            cached = self._materializer(spec, self.root)
            self._designs[spec.content_key()] = cached
            self._spent = plus(self._spent, geom=len(cached.meshes))
        self._spent = plus(self._spent, geom=1)  # one cheap verifier attempt
        clearance = spec.clearance_mm
        if self.mode == "always_fail":
            return [_certificate("insufficient_clearance", spec, clearance)]
        band = spec.constraints
        if clearance < band.min_clearance_mm - 1e-9:
            return [_certificate("insufficient_clearance", spec, clearance)]
        if clearance > band.max_clearance_mm + 1e-9:
            return [_certificate("excessive_clearance", spec, clearance)]
        return []


class _FakeVerifier:
    """The minimum a registry entry needs for `pah design sanity` to describe it."""

    spec = VerifierSpec(
        id="fake_fit", codes=DESIGN_CODES["fit_clearance"], cost_class="cheap", fixtures={}
    )


class FakeDesignRegistry:
    """T2's catalogue, gated: one always-passing verifier, nothing disabled."""

    def __init__(self) -> None:
        self._verifier = _FakeVerifier()

    def register(self, verifier: Any) -> None:  # pragma: no cover - composition only
        self._verifier = verifier

    def ids(self) -> tuple[str, ...]:
        return (self._verifier.spec.id,)

    def get(self, verifier_id: str) -> Any:
        if verifier_id != self._verifier.spec.id:
            raise KeyError(verifier_id)
        return self._verifier

    def disabled(self) -> frozenset[str]:
        return frozenset()

    def why_disabled(self, verifier_id: str) -> str:
        return ""

    def sanity(self) -> dict[str, bool]:
        return {self._verifier.spec.id: True}


class FakeOperators:
    """T3's action space: one pure operator that sets the designed clearance."""

    def __init__(self) -> None:
        self.applied: list[tuple[str, float]] = []

    def ids(self) -> tuple[str, ...]:
        return ("set_clearance",)

    def applicable(self, assembly: Any, certs: list[FailureCertificate]) -> list[str]:
        return ["set_clearance"] if certs else []

    def get(self, operator_id: str) -> Any:
        if operator_id != "set_clearance":
            raise KeyError(operator_id)
        return operator_id

    def apply(self, assembly: Any, plan: RepairPlan) -> Any:
        spec = spec_of(assembly)
        requested = float(plan.params["clearance_mm"])
        d_peg = float(spec.interface.params["d_peg_mm"])
        edited = spec.with_interface(params={"d_hole_mm": d_peg + 2.0 * requested})
        self.applied.append((plan.operator, requested))
        return assembly_from_spec(spec=edited, parent=assembly.id, version=assembly.version + 1)


@dataclass
class ClearancePolicy:
    """A deterministic policy: walk the designed clearance to `target`, one step at a time.

    Once at `target` it keeps nudging by 0.01 mm, so an always-failing verifier
    produces *distinct* designs for as long as the budget lasts. Without that, the
    loop's cycle guard would stop the episode early and a budget test would be
    measuring the guard instead of the budget.
    """

    target: float = 0.30
    step: float = 0.10
    name: str = "rule"

    def propose(
        self,
        assembly: Any,
        certs: list[FailureCertificate],
        history: list[Any],
        budget_left: Any,
    ) -> RepairPlan | None:
        spec = spec_of(assembly)
        current = spec.clearance_mm
        if abs(current - self.target) < 1e-12:
            requested = current + 0.01
        elif current < self.target:
            requested = min(self.target, current + self.step)
        else:
            requested = max(self.target, current - self.step)
        parts = (spec.body.id, spec.insert.id)
        return RepairPlan(
            id=f"plan-{len(history)}",
            operator="set_clearance",
            targets=parts,
            params={"clearance_mm": requested, "parts": parts},
            cost_est=Cost(geom_calls=1),
            proposed_by="rule",
            reason=f"fake: {current:.3f} -> {requested:.3f} mm",
        )


def install(
    monkeypatch: Any,
    *,
    mode: str = "band",
    policies: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Point the composition root's resolvers at the doubles; returns their handles.

    Patching the four `resolve_*` functions (rather than the classes they build) is
    what lets a test go through the *real* `build()` and the *real* `Runtime`: the
    composition order, the plugin pass, the sanity gate and the cascade construction
    are all still exercised.
    """
    from pah.app import runtime as runtime_module

    materializer = FakeMaterializer()
    state: dict[str, Any] = {"materializer": materializer, "cascades": []}
    chosen = dict(policies or {})

    def make_cascade(registry: Any, root: str | Path, materializer_: Any = None) -> BandCascade:
        # The runtime resolves the generator and hands it to the cascade; a test that
        # pins its own still sees the same object, so the wiring is exercised rather
        # than bypassed.
        cascade = BandCascade(
            registry, root, mode=mode, materializer=materializer_ or materializer
        )
        state["cascades"].append(cascade)
        return cascade

    def make_policy(name: str, **_kwargs: Any) -> Any:
        if name in chosen:
            return chosen[name]
        return ClearancePolicy(name=name)

    monkeypatch.setattr(runtime_module, "resolve_materializer", lambda: materializer)
    monkeypatch.setattr(runtime_module, "resolve_generator", lambda *_a, **_k: materializer)
    monkeypatch.setattr(runtime_module, "configured_generator", lambda *_a, **_k: "fake")
    monkeypatch.setattr(runtime_module, "resolve_design_registry", FakeDesignRegistry)
    monkeypatch.setattr(runtime_module, "resolve_cascade", make_cascade)
    monkeypatch.setattr(runtime_module, "resolve_operators", FakeOperators)
    monkeypatch.setattr(runtime_module, "resolve_policy", make_policy)
    return state
