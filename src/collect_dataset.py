"""Collect a 240-image dataset from Wikimedia Commons.

Eight everyday photo topics, 30 images each, found through the Commons
API (no account needed). Only JPEG photographs at least 1600 px wide are
kept. Every image is validated with Pillow and logged in a manifest with
its creator, license, and source page, so the dataset is fully
referenced for the report.

Usage: python collect_dataset.py
Output: data/images/<topic>/*.jpg and data/manifest.csv
"""
import csv
import html
import io
import json
import re
import time
import urllib.request
import urllib.parse
from pathlib import Path

from PIL import Image

BASE = Path(__file__).resolve().parent.parent
IMG_DIR = BASE / "data" / "images"
MANIFEST = BASE / "data" / "manifest.csv"
UA = {"User-Agent": "SENG691-HW5/1.0 (UMBC student project)"}
API = "https://commons.wikimedia.org/w/api.php"

TOPICS = {
    "mountains": "mountain landscape photograph",
    "beach": "beach ocean shore photograph",
    "city": "city skyline street photograph",
    "forest": "forest trees path photograph",
    "desert": "desert sand dunes photograph",
    "snow": "snow winter landscape photograph",
    "dogs": "dog portrait photograph",
    "flowers": "flowers garden photograph",
}
PER_TOPIC = 30
MIN_SHORT_SIDE = 500


def api_get(params):
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


def fetch_bytes(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read()


def clean(text):
    text = re.sub(r"<[^>]+>", " ", text or "")
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def search_topic(query, want):
    """Yield imageinfo dicts for JPEG photos matching the query."""
    offset = 0
    seen = 0
    while seen < want * 3:  # scan up to 3x candidates to fill the quota
        data = api_get({
            "action": "query", "format": "json",
            "generator": "search", "gsrsearch": query,
            "gsrnamespace": 6, "gsrlimit": 50, "gsroffset": offset,
            "prop": "imageinfo",
            "iiprop": "url|size|mime|extmetadata", "iiurlwidth": 1600,
        })
        pages = (data.get("query") or {}).get("pages") or {}
        batch = sorted(pages.values(), key=lambda p: p.get("index", 0))
        if not batch:
            return
        for page in batch:
            seen += 1
            info = (page.get("imageinfo") or [{}])[0]
            if info.get("mime") != "image/jpeg":
                continue
            if info.get("width", 0) < 1600 or info.get("height", 0) < 1000:
                continue
            if not info.get("thumburl"):
                continue
            yield page, info
        if "continue" not in data:
            return
        offset = data["continue"]["gsroffset"]
        time.sleep(0.5)


def main():
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for topic, query in TOPICS.items():
        out_dir = IMG_DIR / topic
        out_dir.mkdir(exist_ok=True)
        got = 0
        for page, info in search_topic(query, PER_TOPIC):
            if got >= PER_TOPIC:
                break
            try:
                raw = fetch_bytes(info["thumburl"])
                img = Image.open(io.BytesIO(raw)).convert("RGB")
            except Exception:
                continue
            if min(img.size) < MIN_SHORT_SIDE:
                continue
            name = f"{topic}_{got + 1:02d}.jpg"
            img.save(out_dir / name, "JPEG", quality=90)
            meta = info.get("extmetadata") or {}
            title = page.get("title", "").replace("File:", "")
            title = re.sub(r"\.(jpe?g|JPG)$", "", title)
            rows.append({
                "file": f"{topic}/{name}",
                "seed_topic": topic,
                "title": title[:120],
                "creator": clean((meta.get("Artist") or {}).get("value", ""))[:80],
                "license": clean((meta.get("LicenseShortName") or {}).get("value", "")),
                "source_page": info.get("descriptionurl") or "",
                "image_url": info["thumburl"],
                "width": img.size[0],
                "height": img.size[1],
            })
            got += 1
            time.sleep(0.2)
        print(f"{topic}: {got} images", flush=True)
    with open(MANIFEST, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"TOTAL: {len(rows)} images, manifest at {MANIFEST}")


if __name__ == "__main__":
    main()
