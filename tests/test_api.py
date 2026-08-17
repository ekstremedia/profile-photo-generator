"""The HTTP contract.

Everything here runs against the fake backend, so what is being proved is the
API, the cache, the queue and the store - not the diffusion model. That is the
point: the whole request path is verifiable on a free CI runner in seconds.
"""

from __future__ import annotations

import base64
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from ppg.store.files import avatar_dir

MISSING_ID = "ff" * 16


def _generate(client: TestClient, **body) -> dict:
    response = client.post("/v1/avatars", json=body)
    assert response.status_code == 200, response.text
    return response.json()


# ---------------------------------------------------------------------------
# Meta
# ---------------------------------------------------------------------------


def test_healthz_is_always_ok(client: TestClient) -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["version"]


def test_readyz_reports_the_backend_state(client: TestClient) -> None:
    body = client.get("/readyz").json()
    assert body["backend"] == "fake"
    # "loading" is legitimate: the model warms up in a background task so
    # startup does not block on a 7GB checkpoint.
    assert body["status"] in {"ready", "loading", "error"}
    assert isinstance(body["model_loaded"], bool)
    assert body["queue_depth"] == 0
    # The template composer never probes Ollama, so there is nothing to report.
    assert body["ollama_reachable"] is None


def test_options_describes_this_instance(client: TestClient) -> None:
    body = client.get("/v1/options").json()

    assert body["sizes"] == [256, 128]
    assert body["formats"] == ["webp", "png"]
    assert body["age_bounds"] == {"min": 18, "max": 90}
    assert body["backend"] == "fake"
    assert body["composer"] == "template"

    # The gallery form is built from these, so every pinnable axis must appear.
    for axis in ("sex", "age_range", "ethnicity", "skin_tone", "profession"):
        assert body["axes"][axis], f"axis {axis} is empty"
    first = body["axes"]["sex"][0]
    assert set(first) == {"value", "weight", "label"}


def test_metrics_are_prometheus_shaped(client: TestClient) -> None:
    text = client.get("/metrics").text
    assert "ppg_avatars_total 0" in text
    assert "# TYPE ppg_queue_depth gauge" in text


# ---------------------------------------------------------------------------
# Generation and the image cache
# ---------------------------------------------------------------------------


def test_creating_an_avatar_returns_a_result(client: TestClient) -> None:
    response = client.post("/v1/avatars", json={"seed": 1234})
    assert response.status_code == 200

    body = response.json()
    assert body["id"] == body["hash"]
    assert body["seed"] == 1234
    assert body["backend"] == "fake"
    assert body["composer"] == "template"
    assert body["sizes"] == [256, 128]
    assert body["urls"]["default"] == f"/v1/avatars/{body['id']}/image"
    assert body["attributes"]["age"] >= 18
    assert body["persona"]["name"]
    assert response.headers["X-PPG-Composer"] == "template"


def test_an_identical_request_is_a_cache_hit(client: TestClient) -> None:
    first = client.post("/v1/avatars", json={"seed": 1234})
    assert first.headers["X-PPG-Cache"] == "miss"
    assert first.json()["cached"] is False

    second = client.post("/v1/avatars", json={"seed": 1234})
    # The content hash covers prompt, seed, model and sizes, so an identical
    # request must never render twice - that is the difference between
    # microseconds and seconds.
    assert second.headers["X-PPG-Cache"] == "hit"
    assert second.json()["cached"] is True
    assert second.json()["id"] == first.json()["id"]


def test_a_different_seed_is_a_different_avatar(client: TestClient) -> None:
    assert _generate(client, seed=1)["id"] != _generate(client, seed=2)["id"]


def test_pinned_attributes_come_back_in_the_response(client: TestClient) -> None:
    pins = {
        "sex": "male",
        "ethnicity": "east_asian",
        "glasses": "thick_acetate",
        "expression": "serious",
        "age": 41,
    }
    body = _generate(client, seed="pinned-key", **pins)

    attributes = body["attributes"]
    assert attributes["sex"] == "male"
    assert attributes["ethnicity"] == "east_asian"
    assert attributes["glasses"] == "thick_acetate"
    assert attributes["expression"] == "serious"
    assert attributes["age"] == 41
    assert attributes["age_range"] == "35-44"
    assert body["seed_key"] == "pinned-key"


def test_an_unknown_field_is_rejected(client: TestClient) -> None:
    # AvatarRequest is extra="forbid": a typo'd axis should fail loudly rather
    # than silently generating an unpinned face.
    assert client.post("/v1/avatars", json={"gender": "male"}).status_code == 422


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


def test_metadata_and_image_are_retrievable(client: TestClient) -> None:
    created = _generate(client, seed=7)
    avatar_id = created["id"]

    meta = client.get(f"/v1/avatars/{avatar_id}")
    assert meta.status_code == 200
    assert meta.json()["id"] == avatar_id
    assert meta.json()["prompt"] == created["prompt"]

    image = client.get(f"/v1/avatars/{avatar_id}/image")
    assert image.status_code == 200
    assert image.headers["content-type"] == "image/webp"
    assert image.headers["X-PPG-Size"] == "256"  # default is the largest size
    assert "immutable" in image.headers["cache-control"]


@pytest.mark.parametrize(
    ("requested", "served"),
    [
        (256, 256),  # exact
        (128, 128),  # exact, smaller rung
        (100, 128),  # nearest larger, rather than a 404
    ],
)
def test_the_size_query_picks_the_right_file(
    client: TestClient, requested: int, served: int
) -> None:
    avatar_id = _generate(client, seed=8)["id"]
    response = client.get(f"/v1/avatars/{avatar_id}/image", params={"size": requested})

    assert response.status_code == 200
    assert response.headers["X-PPG-Size"] == str(served)
    with Image.open(io.BytesIO(response.content)) as image:
        assert image.size == (served, served)


def test_png_can_be_requested_explicitly(client: TestClient) -> None:
    avatar_id = _generate(client, seed=9)["id"]
    response = client.get(f"/v1/avatars/{avatar_id}/image", params={"format": "png"})
    assert response.headers["content-type"] == "image/png"
    with Image.open(io.BytesIO(response.content)) as image:
        assert image.format == "PNG"


def test_a_bogus_id_is_a_404(client: TestClient) -> None:
    assert client.get(f"/v1/avatars/{MISSING_ID}").status_code == 404
    assert client.get(f"/v1/avatars/{MISSING_ID}/image").status_code == 404


def test_recent_avatars_are_listed_newest_first(client: TestClient) -> None:
    ids = [_generate(client, seed=seed)["id"] for seed in (21, 22, 23)]
    listed = [row["id"] for row in client.get("/v1/avatars", params={"limit": 3}).json()]
    assert sorted(listed) == sorted(ids)


# ---------------------------------------------------------------------------
# Deterministic by-seed lookup
# ---------------------------------------------------------------------------


def test_the_same_key_always_returns_byte_identical_content(client: TestClient) -> None:
    first = client.get("/v1/avatars/by-seed/ada@example.com")
    assert first.status_code == 200
    assert first.headers["content-type"] == "image/webp"

    second = client.get("/v1/avatars/by-seed/ada@example.com")
    # This is the whole promise of `by-seed`: an <img> tag keyed on a user's
    # email must not change face between page loads, restarts or machines.
    assert second.content == first.content

    other = client.get("/v1/avatars/by-seed/grace@example.com")
    assert other.status_code == 200
    assert other.content != first.content


def test_by_seed_honours_the_size_query(client: TestClient) -> None:
    response = client.get("/v1/avatars/by-seed/ada@example.com", params={"size": 128})
    with Image.open(io.BytesIO(response.content)) as image:
        assert image.size == (128, 128)


def test_an_empty_by_seed_key_is_a_400(client: TestClient) -> None:
    assert client.get("/v1/avatars/by-seed/%20").status_code == 400


# ---------------------------------------------------------------------------
# Batches and jobs
# ---------------------------------------------------------------------------


def test_a_batch_produces_n_distinct_avatars(client: TestClient) -> None:
    response = client.post("/v1/avatars/batch", params={"wait": 60}, json={"n": 4})
    assert response.status_code == 202  # a batch is always a job, even when waited on

    job = response.json()
    assert job["status"] == "done"
    assert job["total"] == 4
    assert job["completed"] == 4
    # `diversity="even"` exists so a contact sheet does not repeat itself.
    assert len(set(job["avatar_ids"])) == 4

    results = client.get(f"/v1/jobs/{job['id']}/results").json()
    assert len({r["attributes"]["ethnicity"] for r in results}) > 1


def test_a_batch_can_be_polled_instead_of_waited_on(client: TestClient) -> None:
    job = client.post("/v1/avatars/batch", json={"n": 2}).json()
    assert job["status"] in {"queued", "running", "done"}
    assert client.get(f"/v1/jobs/{job['id']}").status_code == 200


def test_an_unknown_job_is_a_404_that_explains_itself(client: TestClient) -> None:
    response = client.get("/v1/jobs/does-not-exist")
    assert response.status_code == 404
    # Jobs are in-memory, so "gone after a restart" is expected and the message
    # has to say so - otherwise it reads as data loss.
    assert "do not survive a restart" in response.json()["detail"]


# ---------------------------------------------------------------------------
# OpenAI-compatible shim
# ---------------------------------------------------------------------------


def test_images_generations_returns_a_decodable_png(client: TestClient) -> None:
    response = client.post(
        "/v1/images/generations",
        json={"prompt": "a friendly librarian in a knitted sweater", "size": "256x256"},
    )
    assert response.status_code == 200

    body = response.json()
    assert body["created"] > 0
    assert len(body["data"]) == 1

    raw = base64.b64decode(body["data"][0]["b64_json"])
    with Image.open(io.BytesIO(raw)) as image:
        # OpenAI clients expect a real PNG, not the WebP served over HTTP.
        assert image.format == "PNG"
        assert image.size == (256, 256)
    assert body["data"][0]["revised_prompt"]


def test_images_generations_can_return_urls_instead(client: TestClient) -> None:
    body = client.post(
        "/v1/images/generations",
        json={"prompt": "a baker", "size": "128x128", "response_format": "url"},
    ).json()
    assert body["data"][0]["url"].startswith("/v1/avatars/")
    assert body["data"][0]["b64_json"] is None


def test_images_generations_uses_user_as_the_seed_key(client: TestClient) -> None:
    def first_image(user: str) -> bytes:
        body = client.post(
            "/v1/images/generations", json={"prompt": "a baker", "user": user, "size": "128x128"}
        ).json()
        return base64.b64decode(body["data"][0]["b64_json"])

    assert first_image("ada@example.com") == first_image("ada@example.com")
    assert first_image("ada@example.com") != first_image("grace@example.com")


# ---------------------------------------------------------------------------
# Deletion
# ---------------------------------------------------------------------------


def test_deleting_an_avatar_removes_the_row_and_the_files(client: TestClient, settings) -> None:
    avatar_id = _generate(client, seed=99)["id"]
    directory = avatar_dir(settings.outputs_dir, avatar_id)
    assert directory.is_dir()

    assert client.delete(f"/v1/avatars/{avatar_id}").status_code == 204

    assert client.get(f"/v1/avatars/{avatar_id}").status_code == 404
    assert client.get(f"/v1/avatars/{avatar_id}/image").status_code == 404
    assert not directory.exists()  # the row and the bytes go together
    assert client.delete(f"/v1/avatars/{avatar_id}").status_code == 404


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------


def test_api_key_protects_v1_but_not_healthz(make_settings, make_client, monkeypatch) -> None:
    monkeypatch.setenv("PPG_API_KEY", "s3cr3t")
    secured = make_client(make_settings())

    # No token, and a wrong token, are both 401. Note that the guard lives on
    # the avatars and openai-compat routers: /v1/options and /metrics are served
    # by the meta router and stay open even with a key configured.
    assert secured.get("/v1/avatars").status_code == 401
    assert secured.post("/v1/avatars", json={"seed": 1}).status_code == 401
    assert secured.post("/v1/images/generations", json={"prompt": "a baker"}).status_code == 401
    assert secured.get("/v1/avatars", headers={"Authorization": "Bearer wrong"}).status_code == 401

    good = {"Authorization": "Bearer s3cr3t"}
    assert secured.get("/v1/avatars", headers=good).status_code == 200
    assert secured.post("/v1/avatars", json={"seed": 1}, headers=good).status_code == 200

    # Health checks stay open so an orchestrator does not need the key.
    assert secured.get("/healthz").status_code == 200
    assert secured.get("/readyz").status_code == 200


def test_no_api_key_means_no_authentication(client: TestClient) -> None:
    # Default configuration has to work with `docker compose up` and no setup.
    assert client.get("/v1/avatars").status_code == 200
