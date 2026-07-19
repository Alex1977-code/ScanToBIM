"""Erzeugt ein kompaktes S20-Test-Fixture aus dem echten Steuerhaus-Scan.
Struktur identisch zum Original-Projektordner, damit die automatische
Projekterkennung von ScanToBIM dieselben Codepfade durchlaeuft."""
import os, shutil, struct, sys
from PIL import Image

SRC = r"C:\Users\alexanderm\Desktop\steuerhaus_nur_aussen"
DST = r"C:\Users\alexanderm\Desktop\steuerhaus_mini_testdaten"
SUB = os.path.join("Steuerhaus nur aussen", "steuerhaus_nur_aussen", "output")
LAS_STEP = 30          # jeder 30. Punkt -> ~667k Punkte, LAS < 25 MB
                       # (GitHub-Web-Upload-Limit: 25 MB pro Datei)
IMG_STEP = 20          # jedes 20. Bildpaar -> ~19 Paare
IMG_QUALITY = 82       # Originalgroesse beibehalten (Kamera-Auswahl nach
                       # Bildgroesse!), nur staerker komprimieren

def subsample_las(src, dst, step):
    with open(src, "rb") as f:
        hdr_pre = f.read(375)
        point_off = struct.unpack("<I", hdr_pre[96:100])[0]
        reclen = struct.unpack("<H", hdr_pre[105:107])[0]
        npts = struct.unpack("<Q", hdr_pre[247:255])[0]
        f.seek(0)
        pre = bytearray(f.read(point_off))   # Header + VLRs unveraendert
        keep = (npts + step - 1) // step
        # LAS-1.4-Zaehler patchen, EVLR/Waveform-Verweise nullen
        struct.pack_into("<Q", pre, 227, 0)          # start of waveform
        struct.pack_into("<Q", pre, 235, 0)          # start of first EVLR
        struct.pack_into("<I", pre, 243, 0)          # number of EVLRs
        struct.pack_into("<Q", pre, 247, keep)       # number of points
        struct.pack_into("<15Q", pre, 255, keep, *([0] * 14))  # by return
        with open(dst, "wb") as out:
            out.write(pre)
            f.seek(point_off)
            CHUNK = 200000
            written = 0
            remaining = npts   # nicht in evtl. EVLRs nach den Punkten laufen
            while remaining > 0:
                buf = f.read(min(CHUNK, remaining) * reclen)
                if not buf:
                    break
                n = len(buf) // reclen
                remaining -= n
                for i in range(0, n, step):
                    out.write(buf[i * reclen:(i + 1) * reclen])
                    written += 1
                # Schritt-Versatz ueber Chunkgrenzen beibehalten
                rest = n % step
                if rest:
                    skip = min(step - rest, remaining)
                    f.seek(skip * reclen, 1)
                    remaining -= skip
    print(f"  {os.path.basename(dst)}: {written:,} Punkte")
    return written

def filter_lines(src, dst, keep_names, header_lines=1):
    with open(src, encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    out = lines[:header_lines]
    for ln in lines[header_lines:]:
        if any(k in ln for k in keep_names):
            out.append(ln)
    with open(dst, "w", encoding="utf-8") as f:
        f.writelines(out)
    print(f"  {os.path.relpath(dst, DST)}: {len(out)-header_lines} Eintraege")

os.makedirs(DST, exist_ok=True)
out_dir = os.path.join(DST, SUB)
img_dst = os.path.join(out_dir, "images")
for d in [os.path.join(DST, "info"), img_dst,
          os.path.join(img_dst, "left"), os.path.join(img_dst, "right")]:
    os.makedirs(d, exist_ok=True)

print("LAS subsampeln ...")
for name in ["steuerhaus_nur_aussen_uncolorized.las",
             "steuerhaus_nur_aussen_colorized.las"]:
    subsample_las(os.path.join(SRC, SUB, name),
                  os.path.join(out_dir, name), LAS_STEP)

print("Kleine Dateien kopieren ...")
for rel in ["project_info.json", "frame_pose.txt", "preview_image.jpg",
            os.path.join("info", "calibration.yaml"),
            os.path.join(SUB, "trajectory.txt"),
            os.path.join(SUB, "images", "Left.opt"),
            os.path.join(SUB, "images", "Right.opt")]:
    shutil.copy2(os.path.join(SRC, rel), os.path.join(DST, rel))

print("Bildpaare auswaehlen ...")
left_dir = os.path.join(SRC, SUB, "images", "left")
lefts = sorted(x for x in os.listdir(left_dir) if x.endswith(".jpg"))
sel = lefts[::IMG_STEP]
keep_names = set()
for lname in sel:
    rname = lname.replace("left_", "right_")
    rsrc = os.path.join(SRC, SUB, "images", "right", rname)
    if not os.path.exists(rsrc):
        continue
    for side, fname in (("left", lname), ("right", rname)):
        src_img = os.path.join(SRC, SUB, "images", side, fname)
        im = Image.open(src_img)
        im.save(os.path.join(img_dst, side, fname), quality=IMG_QUALITY)
    keep_names.add(lname)
    keep_names.add(rname)
print(f"  {len(keep_names)//2} Paare ({len(keep_names)} Bilder), Originalgroesse")

print("Posen-/Zeitdateien filtern ...")
filter_lines(os.path.join(SRC, SUB, "images", "ImgPose.txt"),
             os.path.join(img_dst, "ImgPose.txt"), keep_names)
filter_lines(os.path.join(SRC, SUB, "images", "xyzopk.txt"),
             os.path.join(img_dst, "xyzopk.txt"), keep_names)
filter_lines(os.path.join(SRC, SUB, "images", "left", "leftImgTime.txt"),
             os.path.join(img_dst, "left", "leftImgTime.txt"), keep_names,
             header_lines=0)
filter_lines(os.path.join(SRC, SUB, "images", "right", "rightImgTime.txt"),
             os.path.join(img_dst, "right", "rightImgTime.txt"), keep_names,
             header_lines=0)

total = sum(os.path.getsize(os.path.join(r, f))
            for r, _, fs in os.walk(DST) for f in fs)
print(f"FERTIG: {DST}  ({total/1e6:.1f} MB)")
