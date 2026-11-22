"""Feature extraction and clustering for the HW5 image album system.

For every collected image this script computes:
  * a 2048-d embedding from a pre-trained ResNet-50 (ImageNet weights,
    classification head removed, global average pooling output), and
  * simple visual statistics used later for music selection and cluster
    description: brightness, saturation, warm/cool balance, edge density,
    and the three dominant colors.

It then clusters the embeddings with K-Means (k chosen by silhouette
score over a small range, with agglomerative clustering run once for
comparison) and evaluates the result against the seed topics the images
were collected under (purity and normalized mutual information), which
gives an honest check of whether the groups are meaningful.

Outputs (in output/):
  features.npz, image_stats.csv, clusters.csv, cluster_summary.md,
  cluster_summary.json, pca_scatter.png, montage_cluster_<i>.jpg
"""
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

BASE = Path(__file__).resolve().parent.parent
IMG_DIR = BASE / "data" / "images"
OUT = BASE / "output"
BATCH = 16


def load_manifest():
    with open(BASE / "data" / "manifest.csv") as f:
        return list(csv.DictReader(f))


def visual_stats(path):
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    small = img.resize((128, 128))
    arr = np.asarray(small).astype(np.float32) / 255.0
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    mx, mn = arr.max(axis=2), arr.min(axis=2)
    brightness = float(mx.mean())
    saturation = float((((mx - mn) / np.maximum(mx, 1e-6))[mx > 0]).mean()) if (mx > 0).any() else 0.0
    warmth = float(r.mean() - b.mean())
    gray = arr.mean(axis=2)
    gx = np.abs(np.diff(gray, axis=1)).mean()
    gy = np.abs(np.diff(gray, axis=0)).mean()
    edge_density = float(gx + gy)
    # Dominant colors: quick k-means (k=3) on a pixel subsample.
    from sklearn.cluster import KMeans

    pixels = arr.reshape(-1, 3)[::37]
    km = KMeans(n_clusters=3, n_init=4, random_state=0).fit(pixels)
    dom = []
    for center in km.cluster_centers_:
        rgb = tuple(int(round(c * 255)) for c in center)
        dom.append("#%02x%02x%02x" % rgb)
    return {
        "brightness": round(brightness, 4),
        "saturation": round(saturation, 4),
        "warmth": round(warmth, 4),
        "edge_density": round(edge_density, 4),
        "dominant_colors": " ".join(dom),
    }


def extract_embeddings(paths):
    import torch
    from torchvision import models, transforms

    weights = models.ResNet50_Weights.IMAGENET1K_V2
    model = models.resnet50(weights=weights)
    model.fc = torch.nn.Identity()  # keep the 2048-d pooled features
    model.eval()
    preprocess = weights.transforms()
    feats = []
    with torch.no_grad():
        for i in range(0, len(paths), BATCH):
            batch = []
            for p in paths[i : i + BATCH]:
                img = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
                batch.append(preprocess(img))
            x = torch.stack(batch)
            emb = model(x).numpy()
            feats.append(emb)
            print(f"embedded {min(i + BATCH, len(paths))}/{len(paths)}", flush=True)
    X = np.vstack(feats)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)  # cosine geometry
    return X


def purity_score(labels_true, labels_pred):
    total = len(labels_true)
    correct = 0
    for c in set(labels_pred):
        members = [t for t, p in zip(labels_true, labels_pred) if p == c]
        correct += Counter(members).most_common(1)[0][1]
    return correct / total


def main():
    OUT.mkdir(exist_ok=True)
    rows = load_manifest()
    paths = [BASE / "data" / "images" / r["file"] for r in rows]
    topics = [r["seed_topic"] for r in rows]

    print("computing visual statistics ...", flush=True)
    stats = [visual_stats(p) for p in paths]
    with open(OUT / "image_stats.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "seed_topic", "brightness", "saturation",
                    "warmth", "edge_density", "dominant_colors"])
        for r, s in zip(rows, stats):
            w.writerow([r["file"], r["seed_topic"], s["brightness"],
                        s["saturation"], s["warmth"], s["edge_density"],
                        s["dominant_colors"]])

    print("extracting ResNet-50 embeddings ...", flush=True)
    X = extract_embeddings(paths)
    np.savez(OUT / "features.npz", X=X,
             files=np.array([r["file"] for r in rows]),
             topics=np.array(topics))

    from sklearn.cluster import KMeans, AgglomerativeClustering
    from sklearn.metrics import silhouette_score, normalized_mutual_info_score

    print("choosing k by silhouette score ...", flush=True)
    results = {}
    for k in range(4, 11):
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
        sil = silhouette_score(X, km.labels_)
        results[k] = (sil, km)
        print(f"  k={k}: silhouette={sil:.3f}", flush=True)
    best_k = max(results, key=lambda k: results[k][0])
    best_sil, km = results[best_k]
    labels = km.labels_
    agg = AgglomerativeClustering(n_clusters=best_k).fit(X)
    agg_sil = silhouette_score(X, agg.labels_)
    print(f"chosen k={best_k} (silhouette {best_sil:.3f}); "
          f"agglomerative at same k: {agg_sil:.3f}", flush=True)

    purity = purity_score(topics, labels)
    nmi = normalized_mutual_info_score(topics, labels)
    print(f"purity vs seed topics: {purity:.3f}; NMI: {nmi:.3f}", flush=True)

    with open(OUT / "clusters.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "seed_topic", "cluster"])
        for r, lab in zip(rows, labels):
            w.writerow([r["file"], r["seed_topic"], int(lab)])

    stats_arr = {
        key: np.array([s[key] for s in stats])
        for key in ("brightness", "saturation", "warmth", "edge_density")
    }
    centers = km.cluster_centers_
    centers = centers / np.linalg.norm(centers, axis=1, keepdims=True)
    dists = 1 - X @ centers.T  # cosine distance to each center

    summary = {
        "n_images": len(rows),
        "chosen_k": int(best_k),
        "silhouette_by_k": {int(k): round(float(v[0]), 4) for k, v in results.items()},
        "silhouette_kmeans": round(float(best_sil), 4),
        "silhouette_agglomerative_same_k": round(float(agg_sil), 4),
        "purity_vs_seed_topics": round(float(purity), 4),
        "nmi_vs_seed_topics": round(float(nmi), 4),
        "clusters": [],
    }
    lines = [f"# Cluster summary (k = {best_k})\n"]
    for c in range(best_k):
        idx = np.where(labels == c)[0]
        member_topics = Counter(topics[i] for i in idx)
        dominant, dom_count = member_topics.most_common(1)[0]
        order = idx[np.argsort(dists[idx, c])]
        exemplars = [rows[i]["file"] for i in order[:5]]
        info = {
            "cluster": int(c),
            "size": int(len(idx)),
            "dominant_seed_topic": dominant,
            "dominant_share": round(dom_count / len(idx), 3),
            "topic_mix": dict(member_topics.most_common()),
            "mean_brightness": round(float(stats_arr["brightness"][idx].mean()), 4),
            "mean_saturation": round(float(stats_arr["saturation"][idx].mean()), 4),
            "mean_warmth": round(float(stats_arr["warmth"][idx].mean()), 4),
            "mean_edge_density": round(float(stats_arr["edge_density"][idx].mean()), 4),
            "exemplars": exemplars,
        }
        summary["clusters"].append(info)
        lines.append(
            f"## Cluster {c}: {len(idx)} images, mostly {dominant} "
            f"({info['dominant_share']:.0%})\n"
            f"Topic mix: {dict(member_topics.most_common())}\n"
            f"Brightness {info['mean_brightness']:.2f}, saturation "
            f"{info['mean_saturation']:.2f}, warmth {info['mean_warmth']:+.2f}, "
            f"edge density {info['mean_edge_density']:.3f}\n"
            f"Closest to center: {', '.join(exemplars)}\n"
        )
        # Contact-sheet montage of the 12 most central images.
        thumbs = []
        for i in order[:12]:
            t = ImageOps.exif_transpose(Image.open(paths[i])).convert("RGB")
            t.thumbnail((320, 320))
            thumbs.append(t)
        W = 320 * 4
        H = 320 * 3
        sheet = Image.new("RGB", (W, H), "white")
        for j, t in enumerate(thumbs):
            sheet.paste(t, ((j % 4) * 320, (j // 4) * 320))
        sheet.save(OUT / f"montage_cluster_{c}.jpg", quality=88)

    with open(OUT / "cluster_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(OUT / "cluster_summary.md", "w") as f:
        f.write("\n".join(lines))

    from sklearn.decomposition import PCA
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    XY = PCA(n_components=2, random_state=0).fit_transform(X)
    plt.figure(figsize=(9, 7))
    scatter = plt.scatter(XY[:, 0], XY[:, 1], c=labels, cmap="tab10", s=18)
    plt.legend(*scatter.legend_elements(), title="cluster", fontsize=8)
    plt.title("ResNet-50 embeddings (PCA) colored by K-Means cluster")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(OUT / "pca_scatter.png", dpi=150)
    print("wrote summaries, montages, and PCA scatter to output/", flush=True)


if __name__ == "__main__":
    main()
