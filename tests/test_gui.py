"""GUI backend: upload, job execution, status polling, downloads."""

import json
import threading
import time
import urllib.error
import urllib.request

import pytest

from scantobim import __version__
from scantobim.gui.server import create_server
from scantobim.io.writers import write_point_cloud
from tests.synthetic import make_box_scan


@pytest.fixture()
def gui_server():
    server, state = create_server(port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_address[1]}"
    yield url, state
    server.shutdown()
    server.server_close()


def _get(url: str):
    with urllib.request.urlopen(url, timeout=30) as r:
        return r.status, r.read()


def _post(url: str, body: bytes, headers: dict | None = None):
    req = urllib.request.Request(url, data=body, headers=headers or {}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def test_page_and_meta(gui_server):
    url, _ = gui_server
    status, body = _get(url + "/")
    assert status == 200
    text = body.decode()
    assert "ScanToBIM" in text and __version__ in text
    assert "Modell erstellen" in text  # German UI

    status, body = _get(url + "/api/meta")
    meta = json.loads(body)
    assert meta["version"] == __version__


def test_upload_run_and_download(gui_server, tmp_path):
    url, _ = gui_server
    src = write_point_cloud(make_box_scan(density=700, noise=0.004), tmp_path / "scan.ply")

    # Upload.
    status, body = _post(
        url + "/api/upload", src.read_bytes(), {"X-Filename": "scan.ply"}
    )
    assert status == 200
    uploaded = json.loads(body)
    assert uploaded["name"] == "scan.ply" and uploaded["size"] > 0

    # Run a reconstruction (fast preset, extra GLB export).
    status, body = _post(
        url + "/api/run",
        json.dumps(
            {
                "mode": "reconstruct",
                "files": [uploaded["path"]],
                "options": {"preset": "fast", "texture": False, "formats": ["glb"]},
            }
        ).encode(),
    )
    assert status == 200
    job = json.loads(body)["job"]

    # Poll until finished.
    deadline = time.time() + 180
    while True:
        _, body = _get(url + f"/api/status?job={job}")
        s = json.loads(body)
        if s["state"] != "running":
            break
        assert time.time() < deadline, "job did not finish in time"
        time.sleep(0.3)

    assert s["state"] == "done", s.get("error")
    assert s["has_viewer"]
    names = {o["name"] for o in s["outputs"]}
    assert {"modell.html", "modell.glb", "bericht.json"} <= names
    assert s["summary"]["Flächen"] == 6
    assert any("Volumen" in k for k in s["summary"])

    # Viewer served inline, report downloadable.
    status, body = _get(url + f"/api/view?job={job}")
    assert status == 200 and b"<html" in body.lower()
    status, body = _get(url + f"/api/output?job={job}&name=bericht.json")
    assert json.loads(body)["planes"] == 6


def test_error_handling(gui_server, tmp_path):
    url, _ = gui_server

    # Unknown job.
    try:
        urllib.request.urlopen(url + "/api/status?job=nope", timeout=10)
        raise AssertionError("expected 404")
    except urllib.error.HTTPError as e:
        assert e.code == 404

    # addpath: missing file, wrong extension.
    status, body = _post(
        url + "/api/addpath", json.dumps({"path": str(tmp_path / "x.ply")}).encode()
    )
    assert status == 404
    doc = tmp_path / "a.docx"
    doc.write_text("x")
    status, _ = _post(url + "/api/addpath", json.dumps({"path": str(doc)}).encode())
    assert status == 400

    # Upload with a non-cloud extension is rejected.
    status, _ = _post(url + "/api/upload", b"x", {"X-Filename": "evil.exe"})
    assert status == 400

    # Run without files.
    status, _ = _post(
        url + "/api/run",
        json.dumps({"mode": "reconstruct", "files": [], "options": {}}).encode(),
    )
    assert status == 400

    # A failing job surfaces the error instead of hanging.
    bad = tmp_path / "empty.xyz"
    bad.write_text("0 0 0\n1 1 1\n")
    status, body = _post(
        url + "/api/run",
        json.dumps({"mode": "reconstruct", "files": [str(bad)], "options": {}}).encode(),
    )
    job = json.loads(body)["job"]
    deadline = time.time() + 60
    while time.time() < deadline:
        _, body = _get(url + f"/api/status?job={job}")
        s = json.loads(body)
        if s["state"] != "running":
            break
        time.sleep(0.2)
    assert s["state"] == "error"
    assert s["error"]


def test_initial_files_preloaded(tmp_path):
    src = write_point_cloud(make_box_scan(density=200), tmp_path / "vorab.ply")
    server, state = create_server(port=0, initial_files=[src])
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}"
        _, body = _get(url + "/api/meta")
        files = json.loads(body)["initial_files"]
        assert len(files) == 1 and files[0]["name"] == "vorab.ply"
    finally:
        server.shutdown()
        server.server_close()


def test_addpath_project_folder(gui_server, tmp_path):
    """A SLAM project folder registers as a preloaded project entry."""
    from tests.test_project import _build_project

    url, _ = gui_server
    root = _build_project(tmp_path, with_photos=False)
    status, body = _post(
        url + "/api/addpath", json.dumps({"path": str(root)}).encode()
    )
    assert status == 200
    entry = json.loads(body)
    assert entry["kind"] == "project"
    assert "cloud.ply" in entry["name"] and "s20_export" in entry["name"]


def test_profiles_save_load_delete(gui_server, tmp_path, monkeypatch):
    """Custom profiles persist, list and delete via the API."""
    monkeypatch.setattr(
        "scantobim.gui.server._PROFILE_FILE", tmp_path / "profiles.json"
    )
    url, _ = gui_server
    settings = {
        "preset": "detail", "source": "slam", "watertight": True,
        "advanced": {"distance_factor": 3.5}, "formats": ["step"],
    }
    status, body = _post(
        url + "/api/profiles",
        json.dumps({"name": "S20 außen", "settings": settings}).encode(),
    )
    assert status == 200
    _post(url + "/api/profiles",
          json.dumps({"name": "S20 innen fein", "settings": settings}).encode())

    _, body = _get(url + "/api/profiles")
    profiles = json.loads(body)["profiles"]
    assert set(profiles) == {"S20 außen", "S20 innen fein"}
    assert profiles["S20 außen"]["advanced"]["distance_factor"] == 3.5

    _post(url + "/api/profiles/delete", json.dumps({"name": "S20 außen"}).encode())
    _, body = _get(url + "/api/profiles")
    assert set(json.loads(body)["profiles"]) == {"S20 innen fein"}


def test_run_with_source_and_advanced(gui_server, tmp_path):
    """Source profile + advanced overrides reach the reconstruction config."""
    url, _ = gui_server
    src = write_point_cloud(make_box_scan(density=700, noise=0.004), tmp_path / "scan.ply")
    status, body = _post(url + "/api/upload", src.read_bytes(), {"X-Filename": "scan.ply"})
    uploaded = json.loads(body)
    status, body = _post(
        url + "/api/run",
        json.dumps({
            "mode": "reconstruct",
            "files": [uploaded["path"]],
            "options": {
                "preset": "fast", "texture": False, "source": "slam",
                "advanced": {"max_planes": 8, "distance_factor": 2.0},
            },
        }).encode(),
    )
    job = json.loads(body)["job"]
    deadline = time.time() + 180
    while True:
        _, body = _get(url + f"/api/status?job={job}")
        s = json.loads(body)
        if s["state"] != "running":
            break
        assert time.time() < deadline
        time.sleep(0.3)
    assert s["state"] == "done", s.get("error")
    _, body = _get(url + f"/api/output?job={job}&name=bericht.json")
    config = json.loads(body)["config"]
    assert config["max_planes"] == 8  # advanced override wins
    assert config["distance_factor"] == 2.0  # …also over the slam profile
    assert config["ghost_offset_tol"] == 0.03  # slam profile applied


def test_page_has_settings_ui(gui_server):
    url, _ = gui_server
    _, body = _get(url + "/")
    text = body.decode()
    assert "Quelle / Scanner" in text
    assert "Erweiterte Einstellungen" in text
    assert "gespeichertes Profil laden" in text
    # Mouseover explanations present on the tunables.
    assert text.count('title="') > 15
    assert "Vielfaches des Punktabstands" in text


def test_auto_winner_offered_and_saveable(gui_server, tmp_path, monkeypatch):
    """Auto run → status carries the winner config → saveable as profile."""
    monkeypatch.setattr(
        "scantobim.gui.server._PROFILE_FILE", tmp_path / "profiles.json"
    )
    url, _ = gui_server
    src = write_point_cloud(make_box_scan(density=500, noise=0.004), tmp_path / "scan.ply")
    _, body = _post(url + "/api/upload", src.read_bytes(), {"X-Filename": "scan.ply"})
    uploaded = json.loads(body)
    _, body = _post(
        url + "/api/run",
        json.dumps({
            "mode": "reconstruct",
            "files": [uploaded["path"]],
            "options": {"preset": "auto", "texture": False},
        }).encode(),
    )
    job = json.loads(body)["job"]
    deadline = time.time() + 300
    while True:
        _, body = _get(url + f"/api/status?job={job}")
        s = json.loads(body)
        if s["state"] != "running":
            break
        assert time.time() < deadline, "auto run did not finish"
        time.sleep(0.5)
    assert s["state"] == "done", s.get("error")
    winner = s["auto_winner"]
    assert winner and winner["advanced"]["distance_factor"] > 0

    # The button's flow: save the winner settings as a named profile.
    name = f"Auto ({winner['candidate']})"
    settings = {"preset": winner["preset"], "advanced": winner["advanced"]}
    status, body = _post(
        url + "/api/profiles", json.dumps({"name": name, "settings": settings}).encode()
    )
    assert status == 200
    assert name in json.loads(body)["profiles"]


def test_page_has_winner_button(gui_server):
    url, _ = gui_server
    _, body = _get(url + "/")
    assert "Gewinner-Einstellungen als Profil speichern" in body.decode()
