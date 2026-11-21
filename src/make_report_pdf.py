"""Render report.md to a styled PDF with the result figures embedded."""
from pathlib import Path

import markdown
from weasyprint import HTML

BASE = Path(__file__).resolve().parent.parent

CSS = """
@page { size: Letter; margin: 2.2cm 2.2cm 2.4cm 2.2cm;
        @bottom-center { content: counter(page); font-size: 9pt; color: #555; } }
body { font-family: Georgia, 'Times New Roman', serif; font-size: 11pt;
       line-height: 1.5; color: #111; }
h1 { font-size: 16pt; line-height: 1.3; text-align: center; margin-bottom: 4pt; }
h2 { font-size: 13pt; margin-top: 20pt; }
h3 { font-size: 11.5pt; margin-top: 14pt; }
p { text-align: justify; margin: 6pt 0; }
figure { text-align: center; margin: 12pt 0; page-break-inside: avoid; }
figure img { max-width: 100%; }
figcaption { font-size: 9.5pt; color: #333; margin-top: 4pt; }
"""

FIG_PCA = """
<figure>
  <img src="output/pca_scatter.png" style="width: 78%">
  <figcaption>Figure 1. The 240 ResNet-50 embeddings projected to two
  dimensions with PCA, colored by K-Means cluster. Clusters occupy
  visibly distinct regions even in two dimensions.</figcaption>
</figure>
"""

FIG_MONTAGE = """
<figure>
  <img src="output/montage_cluster_0.jpg" style="width: 88%">
  <figcaption>Figure 2. Contact sheet of the 12 images closest to the
  center of Cluster 0 (dog portraits, 97 percent dogs).</figcaption>
</figure>
<figure>
  <img src="output/montage_cluster_3.jpg" style="width: 88%">
  <figcaption>Figure 3. Contact sheet of the 12 images closest to the
  center of Cluster 3 (the greenery cluster, 67 percent forest with
  other vegetation heavy images).</figcaption>
</figure>
"""


def main():
    text = (BASE / "report.md").read_text()
    html = markdown.markdown(text)
    html = html.replace("<h2>4.2", FIG_PCA + "<h2>4.2", 1)
    html = html.replace("<h3>4.3", FIG_MONTAGE + "<h3>4.3", 1)
    doc = f"<html><head><style>{CSS}</style></head><body>{html}</body></html>"
    HTML(string=doc, base_url=str(BASE)).write_pdf(BASE / "report.pdf")
    print("wrote report.pdf")


if __name__ == "__main__":
    main()
