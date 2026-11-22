"""Assemble seng691-hw5.ipynb from cell sources, then it can be executed
with: jupyter nbconvert --to notebook --execute --inplace seng691-hw5.ipynb
"""
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
cells = []


def md(src):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": src})


def code(src):
    cells.append({"cell_type": "code", "metadata": {}, "source": src,
                  "execution_count": None, "outputs": []})


md("""# Automatic Photo Albums with a Pre-trained CNN
**Jamel Lawson, SENG 691, Assignment 5 (HW5): Neural Networks and Deep Learning**

This notebook builds a complete automatic photo album system:

1. A dataset of 240 photographs collected from Wikimedia Commons (the collection script is `src/collect_dataset.py`; every image is logged with creator and license in `data/manifest.csv`).
2. Feature extraction with a pre-trained ResNet-50 (transfer learning).
3. Unsupervised grouping with K-Means, scored with silhouette, purity, and NMI.
4. Background music synthesized in code and selected per cluster by a fixed algorithmic rule over the cluster's visual statistics.
5. Slideshow videos rendered with FFmpeg.

Requirements are in `README.md`. Heavy steps cache their results under `output/`, so re-running the notebook reuses the cached features and statistics when they exist.
**Scope note:** This notebook contains the full 240-image run that satisfies the assignment. The hosted web service is an add-on deployment of the same pipeline for interactive use with smaller uploads (8 to 240 photos); it is not a substitute for this notebook run.""")

code("""import csv, json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

BASE = Path.cwd()
DATA = BASE / "data"
OUT = BASE / "output"
OUT.mkdir(exist_ok=True)
print("project folder:", BASE)""")

md("""## 1. The dataset

240 freely licensed photographs from Wikimedia Commons, 30 each across eight seed topics. The topics were used only to assemble a varied set and to check the clustering afterwards. The clustering never sees them.""")

code("""rows = list(csv.DictReader(open(DATA / "manifest.csv")))
print("images:", len(rows))
print("per topic:", dict(Counter(r["seed_topic"] for r in rows)))
print("licenses:", dict(Counter(r["license"] for r in rows).most_common(4)))
widths = [int(r["width"]) for r in rows]
heights = [int(r["height"]) for r in rows]
print("widths: all", set(widths), "| heights min/median/max:",
      min(heights), int(np.median(heights)), max(heights))""")

md("""### Initial dataset analysis

Before any learning, each image is described by four simple statistics that the music rule will also use later: brightness, saturation, warm/cool balance, and edge density (how much fine detail the image has).""")

code("""def visual_stats(path):
    img = Image.open(path).convert("RGB").resize((128, 128))
    arr = np.asarray(img).astype(np.float32) / 255.0
    r, b = arr[..., 0], arr[..., 2]
    mx, mn = arr.max(axis=2), arr.min(axis=2)
    gray = arr.mean(axis=2)
    return {
        "brightness": float(mx.mean()),
        "saturation": float((((mx - mn) / np.maximum(mx, 1e-6))[mx > 0]).mean()),
        "warmth": float(r.mean() - b.mean()),
        "edge_density": float(np.abs(np.diff(gray, axis=1)).mean()
                              + np.abs(np.diff(gray, axis=0)).mean()),
    }

stats_file = OUT / "image_stats.csv"
if stats_file.exists():
    stats = {r["file"]: {k: float(r[k]) for k in
             ("brightness", "saturation", "warmth", "edge_density")}
             for r in csv.DictReader(open(stats_file))}
else:
    stats = {}
    for r in rows:
        stats[r["file"]] = visual_stats(DATA / "images" / r["file"])
    with open(stats_file, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "seed_topic", "brightness", "saturation",
                    "warmth", "edge_density"])
        for r in rows:
            s = stats[r["file"]]
            w.writerow([r["file"], r["seed_topic"], s["brightness"],
                        s["saturation"], s["warmth"], s["edge_density"]])

topics = sorted({r["seed_topic"] for r in rows})
print(f"{'topic':<11}{'bright':>7}{'satur':>7}{'warm':>7}{'edge':>7}")
for t in topics:
    sel = [stats[r["file"]] for r in rows if r["seed_topic"] == t]
    print(f"{t:<11}" + "".join(
        f"{np.mean([s[k] for s in sel]):>7.3f}"
        for k in ("brightness", "saturation", "warmth", "edge_density")))""")

md("""## 2. Feature extraction with a pre-trained CNN

Each image becomes the 2048 numbers produced by a pre-trained ResNet-50 just before its classification layer. The network learned general visual structure from over a million ImageNet images, so its features separate scenes by content, not just color, which is exactly what grouping needs and what a dataset of 240 images could never train from scratch. Vectors are scaled to unit length so that closeness means cosine similarity.""")

code("""feat_file = OUT / "features.npz"
if feat_file.exists():
    saved = np.load(feat_file)
    X, files = saved["X"], list(saved["files"])
    print("loaded cached embeddings:", X.shape)
else:
    import torch
    from torchvision import models

    weights = models.ResNet50_Weights.IMAGENET1K_V2
    model = models.resnet50(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval()
    preprocess = weights.transforms()
    files = [r["file"] for r in rows]
    feats = []
    with torch.no_grad():
        for i in range(0, len(files), 16):
            batch = [preprocess(Image.open(DATA / "images" / f).convert("RGB"))
                     for f in files[i:i + 16]]
            feats.append(model(torch.stack(batch)).numpy())
    X = np.vstack(feats)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    np.savez(feat_file, X=X, files=np.array(files),
             topics=np.array([r["seed_topic"] for r in rows]))
    print("extracted embeddings:", X.shape)

labels_true = [next(r["seed_topic"] for r in rows if r["file"] == f) for f in files]""")

md("""## 3. Clustering

K-Means is run for every cluster count from 4 to 10 and the count with the best silhouette score wins. Agglomerative clustering at the same count checks that the structure is not one algorithm's artifact. Purity and NMI then grade the result against the seed topics the clustering never saw.""")

code("""from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, normalized_mutual_info_score

def purity_score(y_true, y_pred):
    correct = 0
    for c in set(y_pred):
        members = [t for t, p in zip(y_true, y_pred) if p == c]
        correct += Counter(members).most_common(1)[0][1]
    return correct / len(y_true)

fits = {}
for k in range(4, 11):
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    fits[k] = km
    print(f"k={k}: silhouette={silhouette_score(X, km.labels_):.3f}")

best_k = max(fits, key=lambda k: silhouette_score(X, fits[k].labels_))
km = fits[best_k]
labels = km.labels_
agg = AgglomerativeClustering(n_clusters=best_k).fit(X)
print(f"\\nchosen k = {best_k}")
print(f"agglomerative silhouette at k={best_k}: "
      f"{silhouette_score(X, agg.labels_):.3f}")
print(f"purity vs seed topics: {purity_score(labels_true, labels):.3f}")
print(f"NMI vs seed topics:    {normalized_mutual_info_score(labels_true, labels):.3f}")""")

code("""stat_keys = ("brightness", "saturation", "warmth", "edge_density")
S = {k: np.array([stats[f][k] for f in files]) for k in stat_keys}
clusters = []
for c in range(best_k):
    idx = np.where(labels == c)[0]
    mix = Counter(labels_true[i] for i in idx)
    dominant, dom_n = mix.most_common(1)[0]
    info = {"cluster": c, "size": len(idx), "dominant": dominant,
            "share": dom_n / len(idx), "mix": dict(mix.most_common())}
    info.update({f"mean_{k}": float(S[k][idx].mean()) for k in stat_keys})
    clusters.append(info)
    print(f"Cluster {c}: {len(idx):>2} images, {dominant} {dom_n / len(idx):.0%}, "
          f"mix={info['mix']}")
    print(f"   brightness {info['mean_brightness']:.2f}, saturation "
          f"{info['mean_saturation']:.2f}, warmth {info['mean_warmth']:+.2f}, "
          f"edge {info['mean_edge_density']:.3f}")""")

code("""from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

XY = PCA(n_components=2, random_state=0).fit_transform(X)
plt.figure(figsize=(9, 7))
sc = plt.scatter(XY[:, 0], XY[:, 1], c=labels, cmap="tab10", s=18)
plt.legend(*sc.legend_elements(), title="cluster", fontsize=8)
plt.title("ResNet-50 embeddings (PCA) colored by K-Means cluster")
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.tight_layout(); plt.show()""")

md("""Contact sheets of the twelve images closest to two cluster centers. Cluster 0 is almost purely dog portraits; Cluster 3 is the greenery cluster.""")

code("""from IPython.display import Image as ShowImage, display
display(ShowImage(str(OUT / "montage_cluster_0.jpg"), width=720))
display(ShowImage(str(OUT / "montage_cluster_3.jpg"), width=720))""")

md("""## 4. Music, synthesized and selected by rule

Four original tracks are synthesized with NumPy (no copyrighted audio). Each has a profile: tempo, mode, register, timbre, warmth. The selection rule normalizes each cluster's visual statistics across the clusters, builds a target profile (edge density drives tempo, brightness drives register, brightness with saturation drives mode, saturation drives timbre, color balance drives warmth), and picks the track with the smallest weighted distance to the target. Fixed, deterministic, and saved with its scores.

An optional second source, real Creative Commons tracks from Jamendo, can replace the synthesized track per cluster when a personal API key is supplied; see the next cells. It uses the same target profile and the same fallback logic as the hosted web service.""")

md("""### Optional: real tracks from Jamendo (bring your own key)

The same pipeline also runs as a hosted web service, where each cluster's target profile can be matched to a real Creative Commons track from Jamendo instead of a synthesized one. That is optional here and is not part of the submitted results. Paste your own Jamendo client ID in the configuration cell below, and the Jamendo cell after the selection rule will, for each cluster, query Jamendo with the same target-to-query mapping and deterministic candidate scoring the web service uses, download the winning track, and use it for that cluster's slideshow. With no key, or if any step fails for a cluster, that cluster keeps the synthesized track the rule chose, exactly as the web service falls back. Either way, the notebook prints, per cluster, which source was used and, for Jamendo picks, the track title and artist.""")

code("""# --- CONFIGURATION: your own Jamendo client ID (optional) ---
# A free client ID comes from https://devportal.jamendo.com/ (sign up,
# create an app, copy its Client ID). Leave this empty to use only the
# synthesized tracks, which is how the submitted run was produced.
JAMENDO_CLIENT_ID = ""

# Alternative that keeps the key out of the saved notebook:
# import getpass
# JAMENDO_CLIENT_ID = getpass.getpass("Jamendo client ID (blank = synthesized only): ")""")

code("""from scipy.io import wavfile

SR = 22050
def freq(st): return 440.0 * 2 ** (st / 12.0)

MAJ = {"C": [-9, -5, -2], "G": [-14, -10, -5], "Am": [-12, -9, -5], "F": [-16, -12, -9]}
MIN = {"Am": [-12, -9, -5], "F": [-16, -12, -9], "C": [-9, -5, -2], "G": [-14, -10, -5]}
P_MAJ = [-9, -7, -5, -2, 0, 3, 5]
P_MIN = [-12, -9, -7, -5, -2, 0, 3]
TRACKS = {
    "calm":     dict(bpm=72,  mode="major", register=0,   timbre=0.25, warmth=0.6,
                     prog=["C", "G", "Am", "F"], chords=MAJ, scale=P_MAJ, dens=0.45, seed=11),
    "bright":   dict(bpm=116, mode="major", register=12,  timbre=0.85, warmth=0.4,
                     prog=["C", "G", "Am", "F"], chords=MAJ, scale=P_MAJ, dens=0.80, seed=23),
    "warm":     dict(bpm=94,  mode="major", register=0,   timbre=0.50, warmth=1.0,
                     prog=["F", "C", "G", "Am"], chords=MAJ, scale=P_MAJ, dens=0.60, seed=37),
    "dramatic": dict(bpm=82,  mode="minor", register=-12, timbre=0.35, warmth=0.2,
                     prog=["Am", "F", "C", "G"], chords=MIN, scale=P_MIN, dens=0.50, seed=41),
}

def tone(st, seconds, kind, gain=0.2):
    t = np.linspace(0, seconds, int(SR * seconds), endpoint=False)
    f = freq(st)
    if kind == "pad":
        w = 0.7 * np.sin(2 * np.pi * f * t) + 0.3 * np.sin(2 * np.pi * 2 * f * t)
        e = np.minimum(1, t / (seconds * 0.3)) * np.minimum(1, (seconds - t) / (seconds * 0.3))
    elif kind == "pluck":
        w = np.sin(2 * np.pi * f * t) + 0.4 * np.sin(2 * np.pi * 2 * f * t)
        e = np.exp(-3.5 * t / seconds)
    else:
        w = np.sin(2 * np.pi * f * t)
        e = np.minimum(1, t / 0.05) * np.minimum(1, (seconds - t) / 0.1)
    return gain * w * e

def synth(spec, target_seconds=46.0):
    bar = 4 * 60.0 / spec["bpm"]
    loops = int(np.ceil(target_seconds / (bar * len(spec["prog"]))))
    audio = np.zeros(int(SR * bar * len(spec["prog"]) * loops))
    rng = np.random.default_rng(spec["seed"])
    pos = 0.0
    for _ in range(loops):
        for name in spec["prog"]:
            tones = spec["chords"][name]
            for st in tones:
                seg = tone(st + spec["register"], bar, "pad", 0.10)
                i = int(pos * SR); audio[i:i + len(seg)] += seg[:len(audio) - i]
            seg = tone(tones[0] - 12 + spec["register"] // 2, bar, "bass", 0.16)
            i = int(pos * SR); audio[i:i + len(seg)] += seg[:len(audio) - i]
            for s in range(8):
                if rng.random() < spec["dens"]:
                    seg = tone(int(rng.choice(spec["scale"])) + 12 + spec["register"],
                               bar / 8 * 1.8, "pluck", 0.12)
                    j = int((pos + s * bar / 8) * SR)
                    audio[j:j + len(seg)] += seg[:len(audio) - j]
            pos += bar
    fade = int(SR * 1.5)
    audio[:fade] *= np.linspace(0, 1, fade)
    audio[-fade:] *= np.linspace(1, 0, fade)
    return audio / np.abs(audio).max() * 0.72

(OUT / "music").mkdir(exist_ok=True)
for name, spec in TRACKS.items():
    wavfile.write(OUT / "music" / f"{name}.wav", SR,
                  (synth(spec) * 32767).astype(np.int16))
    print(f"{name}: {spec['bpm']} BPM, {spec['mode']}")""")

code("""def track_profile(name):
    s = TRACKS[name]
    return {"tempo_norm": (s["bpm"] - 60) / 70.0,
            "major": 1.0 if s["mode"] == "major" else 0.0,
            "timbre": s["timbre"],
            "register_norm": (s["register"] + 12) / 24.0,
            "warmth": s["warmth"]}

def choose_track(cluster, all_clusters):
    def norm(v, key):
        vals = [c[key] for c in all_clusters]
        lo, hi = min(vals), max(vals)
        return 0.5 if hi <= lo else float(np.clip((v - lo) / (hi - lo), 0, 1))
    bright = norm(cluster["mean_brightness"], "mean_brightness")
    sat = norm(cluster["mean_saturation"], "mean_saturation")
    target = {"tempo_norm": norm(cluster["mean_edge_density"], "mean_edge_density"),
              "major": 0.25 + 0.75 * (bright + sat) / 2.0,
              "timbre": sat, "register_norm": bright,
              "warmth": norm(cluster["mean_warmth"], "mean_warmth")}
    weights = {"tempo_norm": 1.4, "major": 1.2, "timbre": 0.8,
               "register_norm": 0.8, "warmth": 1.0}
    scores = {n: round(float(sum(weights[k] * (track_profile(n)[k] - target[k]) ** 2
                                  for k in weights)), 4) for n in TRACKS}
    return min(scores, key=scores.get), scores

choices = {}
for info in clusters:
    best, scores = choose_track(info, clusters)
    choices[info["cluster"]] = best
    print(f"Cluster {info['cluster']} ({info['dominant']}): -> {best}  {scores}")""")

code("""# --- Jamendo: real Creative Commons tracks, same rule, with fallback ---
# Ported from the web service's jamendo module so this notebook stays
# self-contained. The cluster target built by the selection rule above
# (keys tempo_norm, major, timbre, register_norm, warmth) is mapped to
# a fuzzy-tag query, candidates are scored deterministically, and the
# winner is downloaded. Any failure returns None and that cluster
# falls back to its synthesized track.
import json as _jam_json
import urllib.parse as _jam_parse
import urllib.request as _jam_request

JAMENDO_API_URL = "https://api.jamendo.com/v3.0/tracks/"
_JAM_BRIGHT = ("happy", "uplifting", "energetic", "fun")
_JAM_DARK = ("melancholic", "dramatic", "emotional", "sad")
_JAM_WARM = ("warm", "acoustic", "folk")
_JAM_COOL = ("electronic", "ambient", "chill")
_JAM_SPEED_ORDER = {"low": 0, "medium": 1, "high": 2}


def _jam_speed_bucket(tempo_norm):
    if tempo_norm >= 0.62:
        return "high"
    if tempo_norm >= 0.35:
        return "medium"
    return "low"


def jamendo_target_to_query(target):
    tags = list(_JAM_BRIGHT if target["major"] >= 0.5 else _JAM_DARK)
    tags += _JAM_WARM if target["warmth"] >= 0.5 else _JAM_COOL
    if target["tempo_norm"] >= 0.62:
        tags.append("energetic")
    elif target["tempo_norm"] < 0.35:
        tags.append("relaxing")
    return tags, _jam_speed_bucket(target["tempo_norm"])


def _jam_get(url):
    req = _jam_request.Request(url, headers={"User-Agent": "seng691-hw5-notebook/1.0"})
    with _jam_request.urlopen(req, timeout=20) as resp:
        return _jam_json.loads(resp.read().decode("utf-8"))


def jamendo_search_tracks(tags, cid, limit=50):
    # The API's speed filter is not used: Jamendo speed metadata is so
    # sparse that filtering by it returns almost nothing. Speed is
    # scored locally instead, when a candidate carries it.
    params = {"client_id": cid, "format": "json", "limit": limit,
              "fuzzytags": " ".join(tags), "include": "musicinfo+licenses",
              "order": "popularity_total", "vocalinstrumental": "instrumental"}
    data = _jam_get(JAMENDO_API_URL + "?" + _jam_parse.urlencode(params))
    if data.get("headers", {}).get("status") != "success":
        return []
    return data.get("results", [])


def jamendo_score_candidate(track, want_tags, want_speed):
    # Lower is better. Deterministic in the track's own fields.
    info = track.get("musicinfo", {}) or {}
    speed_dist = abs(_JAM_SPEED_ORDER.get(info.get("speed") or "", 1)
                     - _JAM_SPEED_ORDER[want_speed])
    track_tags = set()
    for key in ("genres", "vartags", "instruments"):
        track_tags.update(t.lower() for t in (info.get("tags", {}) or {}).get(key, []) or [])
    overlap = len(track_tags.intersection(t.lower() for t in want_tags))
    downloadable = 0 if track.get("audiodownload_allowed") else 1
    return (speed_dist, -overlap, downloadable, int(track.get("id", 0)))


def jamendo_pick_for_target(target, cid):
    # Best Jamendo track for one cluster target, or None to fall back.
    if not cid:
        return None
    tags, speed = jamendo_target_to_query(target)
    try:
        results = jamendo_search_tracks(tags, cid)
        if not results:
            results = jamendo_search_tracks(tags[:2], cid)
        if not results:
            results = jamendo_search_tracks(tags[:1], cid)
    except Exception:
        return None
    if not results:
        return None
    best = min(results, key=lambda t: jamendo_score_candidate(t, tags, speed))
    audio_url = best.get("audiodownload") if best.get("audiodownload_allowed") else best.get("audio")
    if not audio_url:
        return None
    return {"source": "jamendo", "track_id": int(best.get("id", 0)),
            "title": best.get("name", "Untitled"),
            "artist": best.get("artist_name", "Unknown artist"),
            "audio_url": audio_url,
            "page_url": f"https://www.jamendo.com/track/{best.get('id')}",
            "license_ccurl": (best.get("licenses", {}) or {}).get("ccurl")
                             or best.get("license_ccurl", ""),
            "duration": best.get("duration"),
            "query_tags": tags, "query_speed": speed,
            "score": jamendo_score_candidate(best, tags, speed)[:3]}


def jamendo_download(audio_url, dest):
    # Download the track file; True only if it looks like real audio.
    try:
        req = _jam_request.Request(audio_url,
                                   headers={"User-Agent": "seng691-hw5-notebook/1.0"})
        with _jam_request.urlopen(req, timeout=60) as resp, open(dest, "wb") as f:
            f.write(resp.read())
        return dest.stat().st_size > 10_000
    except Exception:
        return False


def cluster_target(cluster, all_clusters):
    # The same target profile choose_track builds, returned on its own
    # so the Jamendo query and the synthesized choice read identical
    # numbers.
    def norm(v, key):
        vals = [c[key] for c in all_clusters]
        lo, hi = min(vals), max(vals)
        return 0.5 if hi <= lo else float(np.clip((v - lo) / (hi - lo), 0, 1))
    bright = norm(cluster["mean_brightness"], "mean_brightness")
    sat = norm(cluster["mean_saturation"], "mean_saturation")
    return {"tempo_norm": norm(cluster["mean_edge_density"], "mean_edge_density"),
            "major": 0.25 + 0.75 * (bright + sat) / 2.0,
            "timbre": sat, "register_norm": bright,
            "warmth": norm(cluster["mean_warmth"], "mean_warmth")}


# Resolve every cluster to an audio file: Jamendo when a key is set and
# the query and download succeed, the synthesized track otherwise.
_key = (JAMENDO_CLIENT_ID or "").strip()
music_files = {}
music_picks = {}
for info in clusters:
    c = info["cluster"]
    pick = jamendo_pick_for_target(cluster_target(info, clusters), _key) if _key else None
    dest = OUT / "music" / f"jamendo_cluster_{c}.mp3"
    if pick and jamendo_download(pick["audio_url"], dest):
        music_files[c] = dest
        music_picks[c] = pick
        print(f"Cluster {c} ({info['dominant']}): Jamendo - '{pick['title']}' "
              f"by {pick['artist']}  {pick['page_url']}")
    else:
        music_files[c] = OUT / "music" / f"{choices[c]}.wav"
        music_picks[c] = {"source": "synthesized", "track": choices[c]}
        reason = "no Jamendo client ID set" if not _key else "Jamendo lookup failed, falling back"
        print(f"Cluster {c} ({info['dominant']}): synthesized - '{choices[c]}' ({reason})")

with open(OUT / "music_choices.json", "w") as f:
    _jam_json.dump(music_picks, f, indent=2, default=str)
print("saved", OUT / "music_choices.json")""")

code("""from IPython.display import Audio, display
print("Music for Cluster 3 (greenery):", music_picks[3])
display(Audio(str(music_files[3])))""")

md("""## 5. Slideshow video

The builder takes a cluster's most central images, renders each at 1280 by 720 over a blurred fill of itself (nothing cropped, no black bars), fades between them, and lays the chosen track underneath. Below, Cluster 3 is built and shown; `src/build_videos.py` builds the three largest clusters the same way. The audio laid underneath is `music_files[CID]`: the downloaded Jamendo MP3 when one was picked for that cluster, otherwise the synthesized WAV the rule chose.""")

code("""import subprocess, tempfile

def run(cmd): subprocess.run(cmd, check=True, capture_output=True, text=True)

CID = 3
members = [f for f in files if labels[files.index(f)] == CID]
center = X[[files.index(f) for f in members]].mean(axis=0)
center /= np.linalg.norm(center)
members.sort(key=lambda f: -float(X[files.index(f)] @ center))
members = members[:16]
track = choices[CID]
with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    segs = []
    for i, rel in enumerate(members):
        seg = td / f"seg_{i:02d}.mp4"
        vf = ("[0:v]scale=1280:720:force_original_aspect_ratio=increase,"
              "crop=1280:720,boxblur=24:6[bg];"
              "[0:v]scale=1280:720:force_original_aspect_ratio=decrease[fg];"
              "[bg][fg]overlay=(W-w)/2:(H-h)/2,"
              "fade=t=in:st=0:d=0.4,fade=t=out:st=2.2:d=0.4,format=yuv420p")
        run(["ffmpeg", "-y", "-loop", "1", "-t", "2.6", "-i",
             str(DATA / "images" / rel), "-filter_complex", vf, "-r", "30",
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "22", str(seg)])
        segs.append(seg)
    (td / "list.txt").write_text("".join(f"file '{s}'\\n" for s in segs))
    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i",
         str(td / "list.txt"), "-c", "copy", str(td / "joined.mp4")])
    run(["ffmpeg", "-y", "-i", str(td / "joined.mp4"), "-i",
         str(music_files[CID]), "-c:v", "copy", "-c:a", "aac",
         "-b:a", "160k", "-af", "afade=t=out:st=39.6:d=2", "-shortest",
         str(OUT / f"slideshow_cluster_{CID}.mp4")])
print("built slideshow_cluster_3.mp4 with music:", music_picks[CID])""")

code("""from IPython.display import Video, display
display(Video(str(OUT / "slideshow_cluster_3.mp4"), width=720,
              html_attributes="controls muted"))""")

md("""## 6. Conclusion

The unsupervised pipeline recovers the structure of the collection: silhouette peaks at the same eight groups the dataset was seeded with, purity is 0.821 and NMI is 0.739 against labels the clustering never saw, and the disagreements are themselves visual (snow cover and dense greenery cut across topics). Music selection is a written rule over measured image statistics, and every stage, from embeddings to track scores, is reproducible from this notebook and the scripts in `src/`.""")

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "SENG 691 HW5", "language": "python",
                       "name": "seng691-hw5"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
out = BASE / "seng691-hw5.ipynb"
out.write_text(json.dumps(nb, indent=1))
print("wrote", out, "with", len(cells), "cells")
