"""Hunyuan3D adapter: the parts that can be tested without spending a generation.

A real call costs money and needs a Tencent Cloud account, so everything here is
the *deterministic* half of the adapter: prompt construction, fitting a generated
shape onto the designed interface, response parsing across schema drift, and the
refusal when keys are missing. If these are right, the only thing left that can
fail is the network call itself.
"""

from __future__ import annotations

import json

import pytest
import trimesh

from examples.hunyuan3d_materializer import (
    API_VERSION,
    DEFAULT_TIER,
    MODEL_ENV,
    SECRET_ID_ENV,
    SECRET_KEY_ENV,
    TIER_ENV,
    TIERS,
    _first_file_url,
    _fit_to_interface,
    _model,
    _prompt_for,
    _submit_payload,
    _suffix_of,
    _tier,
    materialize,
)

REPO_SPEC = "fixtures/design/demo-peg-hole-clear.json"


def _spec():
    from pathlib import Path

    from pah.design.spec import DesignSpec

    path = Path(__file__).resolve().parents[1] / REPO_SPEC
    return DesignSpec.model_validate(json.loads(path.read_text("utf-8")))


def _reference_box(diameter: float = 9.7, length: float = 14.0) -> trimesh.Trimesh:
    box = trimesh.creation.cylinder(radius=diameter / 2, height=length, sections=32)
    box.apply_translation((0.0, 0.0, -2.0 + length / 2))  # roughly the seated pose
    return box


# --------------------------------------------------------------------------- #
# what we ask for
# --------------------------------------------------------------------------- #
def test_the_prompt_asks_for_the_insert_and_tells_the_model_not_to_hit_a_tolerance():
    """A generated model cannot honour "10.0 mm"; the fit is applied afterwards, and
    the prompt says so instead of asking for something impossible."""
    spec = _spec()
    prompt = _prompt_for(spec)

    assert "cylindrical pin" in prompt
    assert "+Z" in prompt
    assert "Ignore any numeric tolerances" in prompt
    # The requirement text (which may contain mm values) is passed through, but the
    # adapter itself never claims a dimension: no parameter names leak into the prompt.
    assert "d_peg_mm" not in prompt
    assert "engage_mm" not in prompt


# --------------------------------------------------------------------------- #
# fitting the generated shape to the designed interface
# --------------------------------------------------------------------------- #
def test_a_generated_shape_is_scaled_to_the_designed_peg_diameter():
    spec = _spec()
    reference = _reference_box()
    oversized = trimesh.creation.cylinder(radius=10.0, height=30.0, sections=32)

    placed = _fit_to_interface(oversized, reference=reference, spec=spec)
    extents = placed.bounds[1] - placed.bounds[0]

    assert extents[0] == pytest.approx(9.7, abs=0.05)
    assert extents[1] == pytest.approx(9.7, abs=0.05)


def test_scaling_is_uniform_so_the_shape_is_preserved():
    """A non-uniform fit would turn a pin into an oval and still pass the fit check."""
    spec = _spec()
    reference = _reference_box()
    source = trimesh.creation.cylinder(radius=10.0, height=30.0, sections=32)
    ratio_before = source.extents[2] / source.extents[0]

    placed = _fit_to_interface(source, reference=reference, spec=spec)
    ratio_after = placed.extents[2] / placed.extents[0]

    assert ratio_after == pytest.approx(ratio_before, rel=1e-6)


def test_the_insert_lands_at_the_reference_seat():
    spec = _spec()
    reference = _reference_box()
    source = trimesh.creation.cylinder(radius=10.0, height=30.0, sections=32)

    placed = _fit_to_interface(source, reference=reference, spec=spec)

    assert placed.bounds[0][2] == pytest.approx(reference.bounds[0][2], abs=1e-6)
    assert (placed.bounds[0][0] + placed.bounds[1][0]) / 2 == pytest.approx(
        (reference.bounds[0][0] + reference.bounds[1][0]) / 2, abs=1e-6
    )


# --------------------------------------------------------------------------- #
# reading the response
# --------------------------------------------------------------------------- #
def test_the_file_url_is_found_wherever_the_schema_puts_it():
    """The response schema has moved between versions; a rename must not read as
    "the model produced nothing"."""
    shapes = [
        {"ResultFile3Ds": [{"FileUrl": "https://x/a.glb"}]},
        {"ResultFile3Ds": [{"Url": "https://x/b.glb"}]},
        {"Result": {"Files": [{"DownloadUrl": "https://x/c.glb?sig=1"}]}},
        {"Url": "https://x/d.obj"},
    ]
    for payload in shapes:
        assert _first_file_url(payload).startswith("https://x/")


def test_no_url_is_reported_as_empty_not_as_a_wrong_url():
    assert _first_file_url({"Status": "DONE", "ResultFile3Ds": []}) == ""


def test_a_signed_url_still_tells_us_the_format():
    assert _suffix_of("https://x/a.glb?sign=abc") == ".glb"
    assert _suffix_of("https://x/a.obj") == ".obj"
    assert _suffix_of("https://x/a") == ".glb"


# --------------------------------------------------------------------------- #
# refusals (no network, no cost)
# --------------------------------------------------------------------------- #
def test_missing_keys_name_the_variables_and_the_file(monkeypatch, tmp_path):
    monkeypatch.delenv(SECRET_ID_ENV, raising=False)
    monkeypatch.delenv(SECRET_KEY_ENV, raising=False)

    with pytest.raises(RuntimeError) as error:
        materialize(_spec(), tmp_path)

    message = str(error.value)
    assert SECRET_ID_ENV in message and SECRET_KEY_ENV in message
    assert ".env" in message


def test_the_adapter_pins_the_documented_endpoint_and_version():
    """The service is versioned; an adapter that forgets which version it speaks is
    the kind of drift that silently changes what a report means."""
    assert API_VERSION == "2025-05-13"
    assert TIERS["standard"] == ("SubmitHunyuanTo3DJob", "QueryHunyuanTo3DJob")
    assert TIERS["rapid"] == ("SubmitHunyuanTo3DRapidJob", "QueryHunyuanTo3DRapidJob")
    assert TIERS["pro"] == ("SubmitHunyuanTo3DProJob", "QueryHunyuanTo3DProJob")
    assert DEFAULT_TIER == "standard"


# --------------------------------------------------------------------------- #
# "which model am I integrating": tier picks the action pair, Model picks the version
# --------------------------------------------------------------------------- #
def test_the_tier_selects_the_action_pair(monkeypatch):
    monkeypatch.delenv(TIER_ENV, raising=False)
    assert _tier() == "standard"

    monkeypatch.setenv(TIER_ENV, "pro")
    assert _tier() == "pro"

    monkeypatch.setenv(TIER_ENV, "ultra")
    with pytest.raises(RuntimeError, match="not one of"):
        _tier()


def test_the_model_version_is_optional_and_only_sent_when_pinned(monkeypatch):
    """Sent empty, it would ask the service to look up a model named "" — a different
    request than "let the service choose"."""
    monkeypatch.delenv(MODEL_ENV, raising=False)
    assert _model() == ""
    assert "Model" not in _submit_payload("a pin")

    monkeypatch.setenv(MODEL_ENV, "HY-3D-2.5")
    payload = _submit_payload("a pin", model=_model())
    assert payload["Model"] == "HY-3D-2.5"


def test_the_payload_never_carries_both_a_prompt_and_an_image():
    """The service rejects that combination; failing here beats failing at the API."""
    text = _submit_payload("a pin")
    image = _submit_payload("ignored", image_url="https://x/a.png")

    assert "Prompt" in text and "ImageUrl" not in text
    assert "ImageUrl" in image and "Prompt" not in image
    assert text["ResultFormat"] == "GLB"
