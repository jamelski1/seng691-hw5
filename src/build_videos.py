"""Build slideshow videos for the strongest clusters.

For each chosen cluster: take its most central images (closest to the
cluster center in embedding space), render each as a 1280x720 frame with
a blurred-fill background and short fade in/out, concatenate them, and
lay the algorithmically chosen music track underneath (see
make_music.py). Requires ffmpeg on the PATH.

Usage: python build_videos.py [cluster_id ...]
With no arguments it builds the three largest clusters.
"""
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "output"
SECONDS_PER_IMAGE = 2.6
MAX_IMAGES = 16
W, H = 1280, 720


def run(cmd):
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def segment(image_path, out_path):
    vf = (
        f"[0:v]scale={W}:{H}:force_original_aspect_ratio=increase,"
        f"crop={W}:{H},boxblur=24:6[bg];"
        f"[0:v]scale={W}:{H}:force_original_aspect_ratio=decrease[fg];"
        f"[bg][fg]overlay=(W-w)/2:(H-h)/2,"
        f"fade=t=in:st=0:d=0.4,fade=t=out:st={SECONDS_PER_IMAGE - 0.4}:d=0.4,"
        f"format=yuv420p"
    )
    run(["ffmpeg", "-y", "-loop", "1", "-t", str(SECONDS_PER_IMAGE),
         "-i", str(image_path), "-filter_complex", vf,
         "-r", "30", "-c:v", "libx264", "-preset", "veryfast",
         "-crf", "22", str(out_path)])


def main():
    with open(OUT / "cluster_summary.json") as f:
        summary = json.load(f)
    with open(OUT / "music_choices.json") as f:
        choices = json.load(f)
    data = np.load(OUT / "features.npz")
    X, files = data["X"], list(data["files"])
    with open(OUT / "clusters.csv") as f:
        assign = {r["file"]: int(r["cluster"]) for r in csv.DictReader(f)}

    if len(sys.argv) > 1:
        wanted = [int(a) for a in sys.argv[1:]]
    else:
        by_size = sorted(summary["clusters"], key=lambda c: -c["size"])
        wanted = [c["cluster"] for c in by_size[:3]]

    for cid in wanted:
        members = [f for f in files if assign[f] == cid]
        center = X[[files.index(f) for f in members]].mean(axis=0)
        center = center / np.linalg.norm(center)
        members.sort(key=lambda f: -float(X[files.index(f)] @ center))
        members = members[:MAX_IMAGES]
        track = choices[str(cid)]["chosen_track"]
        print(f"cluster {cid}: {len(members)} images, music '{track}'",
              flush=True)
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            segs = []
            for i, rel in enumerate(members):
                seg = td / f"seg_{i:02d}.mp4"
                segment(BASE / "data" / "images" / rel, seg)
                segs.append(seg)
            listfile = td / "list.txt"
            listfile.write_text("".join(f"file '{s}'\n" for s in segs))
            joined = td / "joined.mp4"
            run(["ffmpeg", "-y", "-f", "concat", "-safe", "0",
                 "-i", str(listfile), "-c", "copy", str(joined)])
            out_file = OUT / f"slideshow_cluster_{cid}.mp4"
            duration = SECONDS_PER_IMAGE * len(members)
            run(["ffmpeg", "-y", "-i", str(joined),
                 "-i", str(OUT / "music" / f"{track}.wav"),
                 "-c:v", "copy", "-c:a", "aac", "-b:a", "160k",
                 "-af", f"afade=t=out:st={duration - 2}:d=2",
                 "-shortest", str(out_file)])
        print(f"  wrote {out_file.name}", flush=True)


if __name__ == "__main__":
    main()
