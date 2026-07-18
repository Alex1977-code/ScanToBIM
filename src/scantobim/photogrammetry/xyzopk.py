"""Photogrammetric exterior orientation: ``xyzopk.txt`` (SHARE SLAM S20 & Co.).

Some SLAM/photogrammetry exports ship camera poses not as a COLMAP model but
as one text line per photo: name, position X/Y/Z and the rotation angles
Omega/Phi/Kappa. The file carries **no intrinsics and no stated rotation
convention** — both are recovered here by *self-calibration against the
colorized point cloud*: the cloud was colored from exactly these photos, so
the correct convention + focal length is the one whose test projections
reproduce the point colors best.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from scantobim.photogrammetry.colmap import CameraPose


def read_xyzopk(path: str | Path) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Parse an xyzopk file → ``[(image name, xyz, opk in degrees), …]``.

    Tolerant of headers/comments, comma or whitespace separation, and the
    image name in the first or last column. Angles already in radians
    (all magnitudes ≤ 2π) are converted to degrees.
    """
    entries: list[tuple[str, np.ndarray, np.ndarray]] = []
    max_angle = 0.0
    for raw in Path(path).read_text(errors="replace").splitlines():
        line = raw.strip().lstrip("#")
        if not line:
            continue
        parts = line.replace(",", " ").replace(";", " ").split()
        if len(parts) < 7:
            continue
        name = None
        nums: list[float] = []
        for tok in parts:
            try:
                nums.append(float(tok))
            except ValueError:
                if name is None:
                    name = tok
        if name is None or len(nums) < 6:
            continue  # header line
        xyz = np.array(nums[0:3])
        opk = np.array(nums[3:6])
        max_angle = max(max_angle, float(np.abs(opk).max()))
        entries.append((name, xyz, opk))
    if entries and max_angle <= 2 * np.pi + 0.01:
        entries = [(n, xyz, np.rad2deg(opk)) for n, xyz, opk in entries]
    return entries


def read_imgpose(path: str | Path):
    """Parse an ImgPose file → ``[(name|None, xyz, quat(4), t|None)] …``.

    SHARE S20 exports ``images/ImgPose.txt``: per photo a position, a unit
    QUATERNION and a timestamp — unambiguous rotations, unlike Omega/Phi/
    Kappa. The parser is layout-tolerant: the quaternion is found as the
    first 4-number window with norm ≈ 1, the position as the 3 numbers
    right before it (or after), the timestamp as the largest remaining
    magnitude. The component ORDER (xyzw vs wxyz) stays open — the caller
    resolves it by color-scoring both.
    """
    entries = []
    for raw in Path(path).read_text(errors="replace").splitlines():
        line = raw.strip().lstrip("#")
        if not line:
            continue
        parts = line.replace(",", " ").replace(";", " ").split()
        name = None
        nums: list[float] = []
        for tok in parts:
            try:
                nums.append(float(tok))
            except ValueError:
                if name is None:
                    name = tok
        if len(nums) < 7:
            continue
        arr = np.array(nums)
        qi = None
        for i in range(len(arr) - 3):
            if 0.85 < float(np.linalg.norm(arr[i:i + 4])) < 1.15:
                qi = i
                break
        if qi is None:
            continue
        quat = arr[qi:qi + 4].copy()
        if qi >= 3:
            xyz = arr[qi - 3:qi].copy()
            used = set(range(qi - 3, qi + 4))
        elif len(arr) >= qi + 7:
            xyz = arr[qi + 4:qi + 7].copy()
            used = set(range(qi, qi + 7))
        else:
            continue
        rest = [v for j, v in enumerate(arr) if j not in used]
        t = float(max(rest, key=abs)) if rest else None
        entries.append((name, xyz, quat, t))
    return entries


def _fit_similarity(src, dst, with_scale: bool):
    """Umeyama fit ``s·R·src + t ≈ dst``; returns (s, R, t, median residual)."""
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    src_c = src - mu_s
    dst_c = dst - mu_d
    h = src_c.T @ dst_c / len(src)
    u, sv, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    s_diag = np.diag([1.0, 1.0, d])
    r = vt.T @ s_diag @ u.T
    if with_scale:
        var_s = (src_c**2).sum() / len(src)
        scale = float((sv * np.diag(s_diag)).sum() / max(var_s, 1e-12))
        scale = scale if np.isfinite(scale) and scale > 1e-6 else 1.0
    else:
        scale = 1.0
    t = mu_d - scale * (r @ mu_s)
    resid = np.linalg.norm(src @ (scale * r).T + t - dst, axis=1)
    return scale, r, t, float(np.median(resid))


def _align_raw_to_trajectory(raw, trajectory, stats_out: dict | None = None):
    """Register ImgPose positions onto the scanner trajectory.

    ImgPose files can live in their OWN coordinate frame (observed on the
    S20: cameras ~150 m beside the cloud). Both paths describe the SAME
    walk — but WHICH correspondence links them is not knowable up front
    (clock offsets, partial overlaps, reversed ordering, unit scales). So
    EVERY pairing hypothesis is fitted and the best residual wins:

    * time-matched (clamp-free: only photos inside the trajectory window)
    * time-matched with constant clock offset (start-aligned)
    * arc-length (same walked fraction), forward and reversed

    each as rigid (Kabsch) AND as similarity (Umeyama, catches unit
    mismatches). All residuals go into the report — the next run shows
    exactly which hypothesis fits instead of leaving us guessing.

    Returns ``(aligned_raw, align_rot)`` — ``align_rot`` is ``None`` when
    no hypothesis passes the residual gate.
    """
    if trajectory is None:
        return raw, None
    traj = np.asarray(trajectory, dtype=np.float64)
    if traj.ndim != 2 or len(traj) < 10:
        return raw, None
    xyz = np.array([x for (_n, x, _q, _t) in raw], dtype=np.float64)
    if len(xyz) < 8:
        return raw, None
    txyz = traj[:, :3]

    hypotheses: list[tuple[str, np.ndarray, np.ndarray]] = []  # (label, src, dst)

    def _add_time_hypothesis(label, pt, ok_t, tt_s, txyz_s):
        # Only photos strictly INSIDE the trajectory window — np.interp
        # clamps at the ends, and clamped many-to-one pairs poison the fit
        # (that was the 130 m residual of the first attempt).
        inside = ok_t & (pt >= tt_s[0]) & (pt <= tt_s[-1])
        if inside.sum() < 8:
            return
        dst = np.column_stack([
            np.interp(pt[inside], tt_s, txyz_s[:, i]) for i in range(3)
        ])
        hypotheses.append((label, xyz[inside], dst))

    if traj.shape[1] >= 4:
        times = np.array(
            [t if t is not None else np.nan for (_n, _x, _q, t) in raw],
            dtype=np.float64,
        )
        ok_t = np.isfinite(times)
        tt = traj[:, 3]
        order = np.argsort(tt)
        tt_s, txyz_s = tt[order], txyz[order]
        if ok_t.sum() >= 8 and tt_s[-1] > tt_s[0]:
            _add_time_hypothesis("zeit", times, ok_t, tt_s, txyz_s)
            shift = tt_s[0] - np.nanmin(times)
            _add_time_hypothesis(
                "zeit+offset", times + shift, ok_t, tt_s, txyz_s
            )

    def _arc(p):
        seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
        a = np.concatenate([[0.0], np.cumsum(seg)])
        return a / max(a[-1], 1e-9)

    a_p = _arc(xyz)
    a_t = _arc(txyz)
    for label, ap in (("bogenlaenge", a_p), ("bogenlaenge-rueckwaerts", 1.0 - a_p)):
        dst = np.column_stack([
            np.interp(ap, a_t, txyz[:, i]) for i in range(3)
        ])
        hypotheses.append((label, xyz, dst))

    best = None  # (residual, label, scale, r, t)
    residuals_report: dict[str, float] = {}
    for label, src, dst in hypotheses:
        for with_scale, suffix in ((False, ""), (True, "+skalierung")):
            scale, r, t, residual = _fit_similarity(src, dst, with_scale)
            if with_scale and not (0.001 <= scale <= 1000.0):
                continue  # no plausible unit factor
            key = label + suffix
            residuals_report[key] = round(residual, 3)
            # Strict improvement only — hypotheses are ordered by prior
            # plausibility (time first), ties must not flip to reversed
            # pairings on symmetric paths.
            if best is None or residual < best[0] - 1e-6:
                best = (residual, key, scale, r, t)

    if best is None:
        return raw, None
    residual, label, scale, r_a, t_a = best
    offset = float(np.linalg.norm(txyz.mean(axis=0) - xyz.mean(axis=0)))
    # Gate RELATIVE to the path extent: 1 m on a 100 m walk is a match,
    # 0.6 m on a 2 m path is a coincidence.
    extent = float(np.linalg.norm(txyz.max(axis=0) - txyz.min(axis=0)))
    gate = min(1.0, max(0.15, 0.02 * extent))
    if residual > gate:
        if stats_out is not None:
            stats_out["posen_ausrichtung"] = {
                "angewendet": False,
                "residuum_m": round(residual, 2),
                "versatz_m": round(offset, 1),
                "zuordnung": label,
                "residuen_aller_hypothesen_m": residuals_report,
            }
        return raw, None
    aligned = [
        (n, scale * (r_a @ np.asarray(x, dtype=np.float64)) + t_a, q, t)
        for (n, x, q, t) in raw
    ]
    if stats_out is not None:
        ang = np.degrees(np.arccos(np.clip((np.trace(r_a) - 1) / 2, -1, 1)))
        stats_out["posen_ausrichtung"] = {
            "angewendet": True,
            "versatz_m": round(offset, 1),
            "residuum_cm": round(residual * 100.0, 1),
            "rotation_deg": round(float(ang), 2),
            "skalierung": round(scale, 4),
            "zuordnung": label,
            "residuen_aller_hypothesen_m": residuals_report,
        }
    return aligned, r_a


def _project_for_score(pc, fx, px0, py0, fisheye):
    """(px, py, valid) for the color scoring — pinhole or POLYFISHEYE."""
    if fisheye is not None:
        from scantobim.photogrammetry.colmap import polyfisheye_px

        px, py = polyfisheye_px(
            pc, fisheye["fx"], fisheye["fy"], fisheye.get("a12") or 0.0,
            fisheye["cx"], fisheye["cy"], tuple(fisheye["poly"]),
        )
        max_t = np.deg2rad(fisheye.get("max_theta_deg") or 120.0)
        rho = np.sqrt(pc[:, 0] ** 2 + pc[:, 1] ** 2)
        valid = np.arctan2(rho, pc[:, 2]) <= max_t
        return px, py, valid
    with np.errstate(divide="ignore", invalid="ignore"):
        px = fx * pc[:, 0] / pc[:, 2] + px0
        py = fx * pc[:, 1] / pc[:, 2] + py0
    return px, py, pc[:, 2] > 0.2


def _quat_to_rot(q: np.ndarray, order: str) -> np.ndarray:
    """Unit quaternion → rotation matrix; ``order`` is 'xyzw' or 'wxyz'."""
    if order == "wxyz":
        w, x, y, z = q
    else:
        x, y, z, w = q
    n = np.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-12:
        return np.eye(3)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _rx(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _ry(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rz(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


# World→camera (COLMAP: x right, y down, z forward) rotation builders for
# the conventions seen in the wild. D flips the photogrammetric camera
# frame (x right, y up, z backward) into the COLMAP frame.
_D = np.diag([1.0, -1.0, -1.0])


def _conventions():
    def m1(o, p, k):
        return _rx(o) @ _ry(p) @ _rz(k)

    def m2(o, p, k):
        return _rz(k) @ _ry(p) @ _rx(o)

    return {
        "opk/cam2world": lambda o, p, k: _D @ m1(o, p, k).T,
        "kpo/cam2world": lambda o, p, k: _D @ m2(o, p, k).T,
        "opk/world2cam": lambda o, p, k: _D @ m1(o, p, k),
        "kpo/world2cam": lambda o, p, k: _D @ m2(o, p, k),
        "opk/z-forward": lambda o, p, k: m1(o, p, k).T,
        "kpo/z-forward": lambda o, p, k: m2(o, p, k).T,
        "opk/z-forward-w2c": lambda o, p, k: m1(o, p, k),
        "kpo/z-forward-w2c": lambda o, p, k: m2(o, p, k),
    }


def _build_cameras(
    entries, convention, fx, width, height,
    cx: float | None = None, cy: float | None = None,
    calibs: list | None = None,
) -> list[CameraPose]:
    from scantobim.photogrammetry.calibration import calibration_for_name

    cams = []
    for name, xyz, opk in entries:
        o, p, k = np.deg2rad(opk)
        rot = convention(o, p, k)
        cams.append(_make_pose(
            name, rot, xyz, fx, width, height, cx, cy,
            calibration_for_name(calibs, name) if calibs else None,
        ))
    return cams


def _make_pose(name, rot, xyz, fx, width, height, cx, cy, calib) -> CameraPose:
    """One CameraPose; POLYFISHEYE calibs bring their own exact intrinsics
    (per stereo side: left/ photos → fisheye_left, right/ → fisheye_right)."""
    if calib is not None and calib.get("model") == "POLYFISHEYE" and calib.get("poly"):
        return CameraPose(
            name=name, width=width, height=height,
            fx=float(calib["fx"]), fy=float(calib["fy"]),
            cx=float(calib["cx"]), cy=float(calib["cy"]),
            k1=0.0, rotation=rot, translation=-rot @ xyz,
            model="POLYFISHEYE", poly=tuple(calib["poly"]),
            a12=float(calib.get("a12") or 0.0),
            max_theta_deg=float(calib.get("max_theta_deg") or 120.0),
        )
    return CameraPose(
        name=name, width=width, height=height,
        fx=fx, fy=fx,
        cx=width / 2.0 if cx is None else cx,
        cy=height / 2.0 if cy is None else cy,
        k1=0.0, rotation=rot, translation=-rot @ xyz,
    )


def cameras_from_xyzopk(
    path: str | Path,
    images_dir: str | Path,
    cloud,
    stats_out: dict | None = None,
    sample_points: int = 30_000,
    probe_cameras: int = 8,
    calibration: dict | None = None,
) -> list[CameraPose]:
    """xyzopk poses → calibrated ``CameraPose`` list.

    Rotation convention and focal length are recovered by projecting a
    sample of the (photo-)colorized point cloud into a handful of probe
    photos: the combination that best reproduces the point colors wins.
    Without cloud colors the in-image fraction decides.

    ``calibration`` (from the scanner's calibration.yaml) narrows the
    focal-length search to a small band around the factory value and
    supplies the true principal point — sharper projections, faster search.
    The color check still validates it, so a stale calibration cannot hurt.
    """
    from scantobim.core.texture import _index_images

    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "xyzopk-Kalibrierung benötigt Pillow: pip install scantobim[photos]"
        ) from exc

    entries = read_xyzopk(path)
    if not entries:
        raise ValueError(f"{path}: keine Kameraposen gefunden")
    image_index = _index_images(Path(images_dir))

    # Image size from the first resolvable photo.
    width = height = None
    probe_photos: list[tuple[int, np.ndarray]] = []
    step = max(1, len(entries) // probe_cameras)
    for ei in range(0, len(entries), step):
        name = entries[ei][0]
        p = image_index.get(name) or image_index.get(Path(name).name)
        if p is None:
            # Names in xyzopk sometimes lack the extension.
            for ext in (".jpg", ".jpeg", ".png"):
                p = image_index.get(Path(name).stem + ext)
                if p is not None:
                    break
        if p is None or not p.exists():
            continue
        img = np.asarray(Image.open(p).convert("RGB"))
        if width is None:
            height, width = img.shape[:2]
        if img.shape[0] == height and img.shape[1] == width:
            probe_photos.append((ei, img))
        if len(probe_photos) >= probe_cameras:
            break
    if width is None:
        raise ValueError(
            f"{images_dir}: keine der xyzopk-Kameras hat ein auffindbares Foto"
        )

    pts = np.asarray(cloud.points, dtype=np.float64)
    colors = cloud.colors
    if len(pts) > sample_points:
        rng = np.random.default_rng(0)
        sel = rng.choice(len(pts), sample_points, replace=False)
        pts = pts[sel]
        colors = None if colors is None else colors[sel]

    from scantobim.photogrammetry.calibration import (
        calibration_for_name,
        pick_calibration,
        pick_calibrations,
    )

    sized = pick_calibrations(calibration, width, height)
    calib = pick_calibration(calibration, width, height)
    fisheye = calib if (calib and calib.get("model") == "POLYFISHEYE"
                       and calib.get("poly")) else None
    cx = cy = None
    if fisheye is not None:
        # Exact factory model (POLYFISHEYE): no sweep — the polynomial IS
        # the projection; only the rotation convention remains open.
        fx_grid = [float(calib["fx"])]
        cx, cy = calib.get("cx"), calib.get("cy")
        cam_name = calib.get("name")
        intrinsics_source = (
            f"calibration.yaml/{cam_name} POLYFISHEYE" if cam_name
            else "calibration.yaml POLYFISHEYE"
        )
    elif calib and calib.get("fx"):
        # Factory focal length OF THE MATCHING CAMERA (size-verified — the
        # 640x480 navigation camera must never calibrate the photo rig).
        # Narrow sweep instead of blind trust: undistorted photos may carry
        # a slightly different new-camera matrix.
        fx_grid = np.geomspace(0.85, 1.2, 5) * float(calib["fx"])
        cx = calib.get("cx")
        cy = calib.get("cy")
        cam_name = calib.get("name")
        intrinsics_source = (
            f"calibration.yaml/{cam_name}" if cam_name else "calibration.yaml"
        )
    else:
        fx_grid = np.geomspace(0.35, 1.8, 14) * width
        intrinsics_source = "selbstkalibriert"
    best = None  # (score, conv_name, fx)
    for conv_name, conv in _conventions().items():
        for fx in fx_grid:
            score = _score(
                entries, conv, float(fx), width, height,
                probe_photos, pts, colors, cx=cx, cy=cy, fisheye=fisheye,
            )
            if best is None or score > best[0]:
                best = (score, conv_name, float(fx))

    score, conv_name, fx = best
    if stats_out is not None:
        stats_out["convention"] = conv_name
        stats_out["fx"] = round(fx, 1)
        stats_out["score"] = round(float(score), 4)
        stats_out["cameras"] = len(entries)
        stats_out["intrinsics_quelle"] = intrinsics_source
    calibs = sized if fisheye is not None else None
    return _build_cameras(
        entries, _conventions()[conv_name], fx, width, height, cx=cx, cy=cy,
        calibs=calibs,
    )


def cameras_from_imgpose(
    path: str | Path,
    images_dir: str | Path,
    cloud,
    stats_out: dict | None = None,
    sample_points: int = 30_000,
    probe_cameras: int = 8,
    calibration: dict | None = None,
    trajectory=None,
) -> list[CameraPose]:
    """ImgPose (quaternion) poses → calibrated ``CameraPose`` list.

    Quaternions carry the full rotation — only component order (xyzw/wxyz),
    world↔camera direction and the axis flip remain open (8 candidates
    instead of the 8×14 sweep of the angle-based xyzopk path). Validated
    by the same color scoring against the colorized cloud.
    """
    from scantobim.core.texture import _index_images

    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "ImgPose-Kalibrierung benötigt Pillow: pip install scantobim[photos]"
        ) from exc

    raw = read_imgpose(path)
    # ImgPose frames are not guaranteed to match the cloud frame — register
    # the photo path onto the scanner trajectory first (rigid, gated by
    # residual). Without this the observed S20 export put every camera
    # ~150 m beside the building.
    raw, align_rot = _align_raw_to_trajectory(raw, trajectory, stats_out)
    entries = [(n, xyz, quat) for n, xyz, quat, _t in raw if n is not None]
    if not entries:
        raise ValueError(
            f"{path}: keine benannten Kameraposen gefunden (ImgPose ohne "
            "Bildnamen — xyzopk wird verwendet)"
        )
    image_index = _index_images(Path(images_dir))

    width = height = None
    probe_photos: list[tuple[int, np.ndarray]] = []
    step = max(1, len(entries) // probe_cameras)
    for ei in range(0, len(entries), step):
        name = entries[ei][0]
        p = image_index.get(name) or image_index.get(Path(name).name)
        if p is None:
            for ext in (".jpg", ".jpeg", ".png"):
                p = image_index.get(Path(name).stem + ext)
                if p is not None:
                    break
        if p is None or not p.exists():
            continue
        img = np.asarray(Image.open(p).convert("RGB"))
        if width is None:
            height, width = img.shape[:2]
        if img.shape[0] == height and img.shape[1] == width:
            probe_photos.append((ei, img))
        if len(probe_photos) >= probe_cameras:
            break
    if width is None:
        raise ValueError(
            f"{images_dir}: keine der ImgPose-Kameras hat ein auffindbares Foto"
        )

    pts = np.asarray(cloud.points, dtype=np.float64)
    colors = cloud.colors
    if len(pts) > sample_points:
        rng = np.random.default_rng(0)
        sel = rng.choice(len(pts), sample_points, replace=False)
        pts = pts[sel]
        colors = None if colors is None else colors[sel]

    from scantobim.photogrammetry.calibration import (
        calibration_for_name,
        pick_calibration,
        pick_calibrations,
    )

    sized = pick_calibrations(calibration, width, height)
    calib = pick_calibration(calibration, width, height)
    fisheye = calib if (calib and calib.get("model") == "POLYFISHEYE"
                       and calib.get("poly")) else None
    cx = cy = None
    if fisheye is not None:
        fx_grid = [float(calib["fx"])]  # exact factory model — no sweep
        cx, cy = calib.get("cx"), calib.get("cy")
        cam_name = calib.get("name")
        intrinsics_source = (
            f"calibration.yaml/{cam_name} POLYFISHEYE" if cam_name
            else "calibration.yaml POLYFISHEYE"
        )
    elif calib and calib.get("fx"):
        fx_grid = np.geomspace(0.85, 1.2, 5) * float(calib["fx"])
        cx = calib.get("cx")
        cy = calib.get("cy")
        cam_name = calib.get("name")
        intrinsics_source = (
            f"calibration.yaml/{cam_name}" if cam_name else "calibration.yaml"
        )
    else:
        fx_grid = np.geomspace(0.35, 1.8, 10) * width
        intrinsics_source = "selbstkalibriert"

    def _candidates():
        for order in ("xyzw", "wxyz"):
            for transpose in (False, True):
                for flip in (False, True):
                    def build(q, _o=order, _t=transpose, _f=flip):
                        rot = _quat_to_rot(q, _o)
                        if _t:
                            rot = rot.T
                        if _f:
                            rot = _D @ rot
                        if align_rot is not None:
                            # positions were re-registered into the cloud
                            # frame — the world→camera rotation follows.
                            rot = rot @ align_rot.T
                        return rot
                    yield f"{order}{'ᵀ' if transpose else ''}{'/flip' if flip else ''}", build

    best = None  # (score, label, fx, rot_builder)
    for label, build in _candidates():
        rots = [build(q) for _n, _xyz, q in entries]
        for fx in fx_grid:
            score = _score_rotations(
                entries, rots, float(fx), width, height,
                probe_photos, pts, colors, cx=cx, cy=cy, fisheye=fisheye,
            )
            if best is None or score > best[0]:
                best = (score, label, float(fx), rots)

    score, label, fx, rots = best
    if stats_out is not None:
        stats_out["convention"] = f"quaternion {label}"
        stats_out["fx"] = round(fx, 1)
        stats_out["score"] = round(float(score), 4)
        stats_out["cameras"] = len(entries)
        stats_out["intrinsics_quelle"] = intrinsics_source
    cams = []
    for (name, xyz, _q), rot in zip(entries, rots):
        cams.append(_make_pose(
            name, rot, xyz, fx, width, height, cx, cy,
            calibration_for_name(sized, name) if fisheye is not None else None,
        ))
    return cams


def _score_rotations(
    entries, rots, fx, width, height, probe_photos, pts, colors,
    cx: float | None = None, cy: float | None = None,
    fisheye: dict | None = None,
) -> float:
    """Like :func:`_score`, but with precomputed per-entry rotations."""
    px0 = width / 2.0 if cx is None else cx
    py0 = height / 2.0 if cy is None else cy
    total = 0.0
    n = 0
    for ei, photo in probe_photos:
        xyz = entries[ei][1]
        rot = rots[ei]
        pc = (pts - xyz) @ rot.T
        px, py, valid = _project_for_score(pc, fx, px0, py0, fisheye)
        if valid.sum() < 50:
            continue
        ok = valid & (px >= 0) & (px <= width - 1) & (py >= 0) & (py <= height - 1)
        if ok.sum() < 50:
            continue
        frac = float(ok.mean())
        if colors is None:
            total += frac
            n += 1
            continue
        sy = py[ok].round().astype(int)
        sx = px[ok].round().astype(int)
        diff = np.abs(
            photo[sy, sx].astype(np.float64) - colors[ok].astype(np.float64)
        ).mean()
        total += (1.0 - diff / 255.0) + 0.1 * frac
        n += 1
    return total / n if n else -1.0


def _score(
    entries, conv, fx, width, height, probe_photos, pts, colors,
    cx: float | None = None, cy: float | None = None,
    fisheye: dict | None = None,
) -> float:
    """Mean agreement of projected cloud points with the probe photos."""
    px0 = width / 2.0 if cx is None else cx
    py0 = height / 2.0 if cy is None else cy
    total = 0.0
    n = 0
    for ei, photo in probe_photos:
        name, xyz, opk = entries[ei]
        o, p, k = np.deg2rad(opk)
        rot = conv(o, p, k)
        pc = (pts - xyz) @ rot.T
        px, py, valid = _project_for_score(pc, fx, px0, py0, fisheye)
        if valid.sum() < 50:
            continue
        ok = valid & (px >= 0) & (px <= width - 1) & (py >= 0) & (py <= height - 1)
        if ok.sum() < 50:
            continue
        frac = float(ok.mean())
        if colors is None:
            total += frac
            n += 1
            continue
        sy = py[ok].round().astype(int)
        sx = px[ok].round().astype(int)
        diff = np.abs(
            photo[sy, sx].astype(np.float64) - colors[ok].astype(np.float64)
        ).mean()
        # Color agreement dominates; in-image fraction breaks ties.
        total += (1.0 - diff / 255.0) + 0.1 * frac
        n += 1
    return total / n if n else -1.0
