# SENG 691 — Assignment 5 (HW5): Automatic Photo Albums

A system that takes a set of photographs, understands them with a
pre-trained convolutional neural network, groups similar images into
collections, and generates slideshow-style video albums whose background
music is selected algorithmically from each collection's visual content.

## Pipeline

1. **Collect** (`src/collect_dataset.py`) — downloads 240 freely
   licensed photographs from Wikimedia Commons across eight seed topics
   (mountains, beach, city, forest, desert, snow, dogs, flowers) and
   writes `data/manifest.csv` with the creator, license, and source page
   of every image.
2. **Analyze** (`src/analyze.py`) — extracts a 2048-dimensional
   embedding for every image from a pre-trained ResNet-50 (ImageNet
   weights, classification head removed), computes visual statistics
   (brightness, saturation, warm/cool balance, edge density, dominant
   colors), clusters the embeddings with K-Means (cluster count chosen
   by silhouette score; agglomerative clustering run for comparison),
   and evaluates the clusters against the seed topics with purity and
   normalized mutual information. Writes cluster assignments, a summary
   with per-cluster descriptions, contact-sheet montages, and a PCA
   scatter plot to `output/`.
3. **Music** (`src/make_music.py`) — synthesizes a small library of
   original music tracks in code and selects one per cluster with a
   fixed weighted-distance rule that matches the cluster's visual
   statistics (edge density drives tempo, brightness and saturation
   drive mode, register, and timbre). The choice is deterministic and
   the scores are saved to `output/music_choices.json`.
4. **Video** (`src/build_videos.py`) — builds a 1280x720 slideshow
   video for each of the three largest clusters from their most central
   images (blurred-fill background, fades) with the selected track
   underneath.

## Requirements

- Python 3.10 or newer
- ffmpeg on the PATH (for the video step)
- An internet connection for the dataset download and for the one-time
  download of the ResNet-50 weights (about 100 MB)

## Install

```bash
python3 -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install scikit-learn pillow numpy matplotlib scipy
```

The CPU build of PyTorch is enough; no GPU is required. On a machine
with a CUDA GPU, the regular `pip install torch torchvision` build also
works with no code changes.

## Run

The whole pipeline also runs as a Jupyter notebook, `seng691-hw5.ipynb`,
which is the recommended way to read the project: it narrates each stage
and carries the executed outputs (scores, plots, contact sheets, the
music choices, and one built video). Open it with `jupyter notebook
seng691-hw5.ipynb` and run all cells; cached features and statistics in
`output/` are reused when present.

Or run the scripts directly:

```bash
python src/collect_dataset.py   # builds data/images/ and data/manifest.csv
python src/analyze.py           # features, clustering, summaries in output/
python src/make_music.py        # tracks in output/music/, choices saved
python src/build_videos.py      # slideshow_cluster_<id>.mp4 in output/
```

To build a video for a specific cluster instead of the three largest:
`python src/build_videos.py 2` (any cluster ids may be given).

## Dataset

All images are freely licensed photographs (Creative Commons or public
domain) retrieved from Wikimedia Commons
(https://commons.wikimedia.org). `data/manifest.csv` lists every image
with its title, creator, license, and source page.
The seed topics were used only to assemble a varied dataset and to
evaluate the clustering; the clustering itself is unsupervised and never
sees the topic labels.

## Outputs

- `output/clusters.csv` — cluster assignment for every image
- `output/cluster_summary.md` / `.json` — sizes, topic mixes, visual
  statistics, and exemplar images per cluster
- `output/montage_cluster_<id>.jpg` — contact sheet of each cluster
- `output/pca_scatter.png` — embeddings projected to two dimensions
- `output/music_choices.json` — the chosen track and scores per cluster
- `output/slideshow_cluster_<id>.mp4` — generated video albums
