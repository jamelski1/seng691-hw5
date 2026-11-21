"""Procedural music library and algorithmic track selection.

The assignment requires background music chosen algorithmically to match
the visual content, not picked by hand and not random. This module does
both halves of that:

1. It synthesizes a small library of original tracks in code (numpy).
   Each track is a chord progression with a generated melody, and each
   has a numeric profile: tempo, mode (major/minor), register, and
   timbral brightness.

2. It scores every track against a cluster's measured visual statistics
   and picks the best match with a fixed, documented rule. Each visual
   statistic is first normalized across the clusters in the run, so the
   targets use the full scale: the most detailed cluster gets the
   fastest target tempo, the brightest gets the highest register, and
   so on. Edge density sets the target tempo, brightness and saturation
   set the target mode, register, and timbre, and the warm/cool color
   balance sets a target warmth. The rule is a weighted distance
   between the cluster's target profile and each track's profile, so
   anyone can rerun it and get the same choice, and the scores are
   saved for the report.
"""
import json
from pathlib import Path

import numpy as np
from scipy.io import wavfile

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "output"
SR = 22050

# Note frequencies: A4 = 440. Semitone offsets from A4.
def freq(semitones_from_a4):
    return 440.0 * 2 ** (semitones_from_a4 / 12.0)


# Chord tones as semitone offsets from A4. C major family and A minor.
CHORDS_MAJOR = {
    "C": [-9, -5, -2],    # C E G
    "G": [-14, -10, -5],  # G B D (one octave placement)
    "Am": [-12, -9, -5],  # A C E
    "F": [-16, -12, -9],  # F A C
}
CHORDS_MINOR = {
    "Am": [-12, -9, -5],
    "F": [-16, -12, -9],
    "C": [-9, -5, -2],
    "G": [-14, -10, -5],
}
PENTA_MAJOR = [-9, -7, -5, -2, 0, 3, 5]   # C D E G A c d (rel. to A4)
PENTA_MINOR = [-12, -9, -7, -5, -2, 0, 3]  # A C D E G a c

TRACKS = {
    "calm": {
        "bpm": 72, "mode": "major", "register": 0, "timbre": 0.25,
        "progression": ["C", "G", "Am", "F"], "chords": CHORDS_MAJOR,
        "scale": PENTA_MAJOR, "melody_density": 0.45, "seed": 11,
    },
    "bright": {
        "bpm": 116, "mode": "major", "register": 12, "timbre": 0.85,
        "progression": ["C", "G", "Am", "F"], "chords": CHORDS_MAJOR,
        "scale": PENTA_MAJOR, "melody_density": 0.8, "seed": 23,
    },
    "warm": {
        "bpm": 94, "mode": "major", "register": 0, "timbre": 0.5,
        "progression": ["F", "C", "G", "Am"], "chords": CHORDS_MAJOR,
        "scale": PENTA_MAJOR, "melody_density": 0.6, "seed": 37,
    },
    "dramatic": {
        "bpm": 82, "mode": "minor", "register": -12, "timbre": 0.35,
        "progression": ["Am", "F", "C", "G"], "chords": CHORDS_MINOR,
        "scale": PENTA_MINOR, "melody_density": 0.5, "seed": 41,
    },
}


def tone(semitones, seconds, kind, gain=0.2):
    t = np.linspace(0, seconds, int(SR * seconds), endpoint=False)
    f = freq(semitones)
    if kind == "pad":
        wave = 0.7 * np.sin(2 * np.pi * f * t) + 0.3 * np.sin(2 * np.pi * 2 * f * t)
        env = np.minimum(1, t / (seconds * 0.3)) * np.minimum(1, (seconds - t) / (seconds * 0.3))
    elif kind == "pluck":
        wave = np.sin(2 * np.pi * f * t) + 0.4 * np.sin(2 * np.pi * 2 * f * t)
        env = np.exp(-3.5 * t / seconds)
    else:  # bass
        wave = np.sin(2 * np.pi * f * t)
        env = np.minimum(1, t / 0.05) * np.minimum(1, (seconds - t) / 0.1)
    return gain * wave * env


def synth_track(spec, target_seconds=46.0):
    spb = 60.0 / spec["bpm"]          # seconds per beat
    bar = 4 * spb                      # 4/4
    prog_seconds = bar * len(spec["progression"])
    loops = int(np.ceil(target_seconds / prog_seconds))
    total = prog_seconds * loops
    audio = np.zeros(int(SR * total))
    rng = np.random.default_rng(spec["seed"])
    pos = 0.0
    for _ in range(loops):
        for chord_name in spec["progression"]:
            tones = spec["chords"][chord_name]
            for st in tones:  # pad chord
                seg = tone(st + spec["register"], bar, "pad", gain=0.10)
                i = int(pos * SR)
                audio[i : i + len(seg)] += seg[: len(audio) - i]
            bass = tone(tones[0] - 12 + (spec["register"] // 2), bar, "bass", gain=0.16)
            i = int(pos * SR)
            audio[i : i + len(bass)] += bass[: len(audio) - i]
            # Melody: eighth notes, each present with melody_density chance.
            steps = 8
            for s in range(steps):
                if rng.random() < spec["melody_density"]:
                    note = int(rng.choice(spec["scale"])) + 12 + spec["register"]
                    seg = tone(note, bar / steps * 1.8, "pluck", gain=0.12)
                    j = int((pos + s * bar / steps) * SR)
                    audio[j : j + len(seg)] += seg[: len(audio) - j]
            pos += bar
    fade = int(SR * 1.5)
    audio[:fade] *= np.linspace(0, 1, fade)
    audio[-fade:] *= np.linspace(1, 0, fade)
    peak = np.abs(audio).max()
    if peak > 0:
        audio = audio / peak * 0.72
    return audio


TRACK_WARMTH = {"calm": 0.6, "bright": 0.4, "warm": 1.0, "dramatic": 0.2}


def track_profile(name):
    s = TRACKS[name]
    return {
        "tempo_norm": (s["bpm"] - 60) / 70.0,          # 72->0.17 ... 116->0.80
        "major": 1.0 if s["mode"] == "major" else 0.0,
        "timbre": s["timbre"],
        "register_norm": (s["register"] + 12) / 24.0,  # -12->0, 0->0.5, 12->1
        "warmth": TRACK_WARMTH[name],
    }


def _norm(value, lo, hi):
    if hi <= lo:
        return 0.5
    return float(np.clip((value - lo) / (hi - lo), 0, 1))


def choose_track(cluster, all_clusters):
    """Weighted-distance match between a cluster's visual statistics and
    each track's profile. Statistics are min-max normalized across the
    clusters in this run. Lower score wins. Fully deterministic."""
    def rng(key):
        vals = [c[key] for c in all_clusters]
        return min(vals), max(vals)

    b_lo, b_hi = rng("mean_brightness")
    s_lo, s_hi = rng("mean_saturation")
    e_lo, e_hi = rng("mean_edge_density")
    w_lo, w_hi = rng("mean_warmth")
    bright = _norm(cluster["mean_brightness"], b_lo, b_hi)
    sat = _norm(cluster["mean_saturation"], s_lo, s_hi)
    target = {
        "tempo_norm": _norm(cluster["mean_edge_density"], e_lo, e_hi),
        # A cluster has to be both dark and muted before the target goes
        # fully minor; the 0.25 floor keeps major in reach otherwise.
        "major": 0.25 + 0.75 * (bright + sat) / 2.0,
        "timbre": sat,
        "register_norm": bright,
        "warmth": _norm(cluster["mean_warmth"], w_lo, w_hi),
    }
    weights = {"tempo_norm": 1.4, "major": 1.2, "timbre": 0.8,
               "register_norm": 0.8, "warmth": 1.0}
    scores = {}
    for name in TRACKS:
        prof = track_profile(name)
        scores[name] = round(float(sum(
            weights[k] * (prof[k] - target[k]) ** 2 for k in weights)), 4)
    best = min(scores, key=scores.get)
    return best, target, scores


def main():
    music_dir = OUT / "music"
    music_dir.mkdir(parents=True, exist_ok=True)
    for name, spec in TRACKS.items():
        audio = synth_track(spec)
        wavfile.write(music_dir / f"{name}.wav", SR,
                      (audio * 32767).astype(np.int16))
        print(f"wrote {name}.wav ({len(audio) / SR:.1f}s, {spec['bpm']} BPM, "
              f"{spec['mode']})", flush=True)
    with open(OUT / "cluster_summary.json") as f:
        summary = json.load(f)
    choices = {}
    for cluster in summary["clusters"]:
        best, target, scores = choose_track(cluster, summary["clusters"])
        choices[str(cluster["cluster"])] = {
            "chosen_track": best,
            "target_profile": {k: round(v, 3) for k, v in target.items()},
            "scores": scores,
        }
        print(f"cluster {cluster['cluster']}: chose {best} {scores}", flush=True)
    with open(OUT / "music_choices.json", "w") as f:
        json.dump(choices, f, indent=2)


if __name__ == "__main__":
    main()
