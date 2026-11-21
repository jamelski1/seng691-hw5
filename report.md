# Automatic Photo Albums: Grouping Images with a Pre-trained CNN and Generating Slideshow Videos with Algorithmically Selected Music

Jamel Lawson
SENG 691, Special Topics in Software Engineering, Fall 2026
Assignment 5 (HW5): Neural Networks and Deep Learning

## 1. Introduction

Photo collections grow faster than anyone organizes them. This project builds a system that takes a set of photographs it has never seen, works out which images belong together, groups them into meaningful collections, and then produces slideshow style video albums for those collections. The background music for each album is not picked by a person and is not random. It is selected by a documented algorithm that reads measurable properties of the images in the collection, such as how bright, saturated, warm, and detailed they are, and matches those properties to a music track with a corresponding profile.

The system has four stages. First, a dataset of 240 photographs is collected from a public source. Second, every image is passed through a pre-trained convolutional neural network, ResNet-50, and represented by the 2048 numbers the network produces just before its classification layer. Images the network sees as similar end up close together in that number space. Third, those representations are grouped with K-Means clustering, and the quality of the grouping is measured both on its own terms and against the topics the photographs were originally collected under. Fourth, for the largest collections, the system renders a slideshow video and lays underneath it the music track its selection rule chose.

## 2. Dataset

The dataset contains 240 photographs downloaded from Wikimedia Commons, a public repository of freely licensed media, through its programming interface. Eight everyday topics were used to assemble a varied collection: mountains, beach, city, forest, desert, snow, dogs, and flowers, with 30 images per topic, for 240 images in total. Only JPEG photographs at least 1600 pixels wide were accepted, and every image was opened and validated before it was kept. All delivered images are 1920 pixels wide, with heights from 504 to 3412 pixels and a median height of 1280 pixels, so the set mixes landscape and portrait orientations the way a real photo library does.

Each image is recorded in a manifest file with its title, creator, license, and the address of its source page. Most images are Creative Commons licensed (101 under CC BY-SA 4.0, with the rest spread across other CC licenses), 37 are public domain, and 5 are CC0. The full manifest ships with the source code, and the dataset can be rebuilt at any time by running the collection script, so the dataset reference is the Wikimedia Commons collection described by that manifest rather than a single fixed archive.

The topic labels deserve one clarification. The topics were used only to assemble a varied dataset and, later, to check the clustering from the outside. The clustering itself is unsupervised and never sees the labels. If the groups the system finds line up with the topics, that alignment is evidence the grouping is meaningful, because nothing in the pipeline was told the answer in advance.

An initial analysis of the dataset shows the topics differ in measurable ways before any learning is involved. Average brightness runs from 0.43 for forest images and 0.47 for dog portraits up to 0.65 for beach and 0.69 for desert images, on a scale where 1 is fully bright. Saturation averages 0.31 across the set but reaches much higher in flower and desert photographs. Edge density, a simple measure of how much fine detail an image contains, averages 0.074 and is highest in forest scenes full of leaves and branches. These differences matter later, because the same statistics that describe the dataset are the ones the music selection rule reads.

## 3. Method

### 3.1 Feature extraction with a pre-trained CNN

Each image is represented by the output of a pre-trained ResNet-50 network (He et al., 2016) with its ImageNet weights, using the version 2 weight set in PyTorch. The final classification layer is removed, so the representation is the 2048 dimensional vector produced by the network's global average pooling layer. This is transfer learning in its simplest form. The network was trained to recognize a thousand kinds of objects in over a million images, and the layers below the classifier learn general visual structure, edges, textures, parts, and scene layout, that transfers to images the network was never trained on. Using a pre-trained network is the justified choice here for two reasons. The dataset is far too small to train a comparable network from scratch, and a hand built feature set of colors and edges alone cannot capture the difference between, for example, a beach and a desert, which can share a palette while differing completely in content.

Before clustering, each 2048 dimensional vector is scaled to unit length. With unit length vectors, closeness is measured by the angle between vectors, called cosine similarity, which compares the pattern of features rather than their overall magnitude. Two photographs of the same kind of scene produce vectors pointing in nearly the same direction even when one is brighter than the other.

### 3.2 Clustering

The vectors are grouped with K-Means clustering (MacQueen, 1967), which partitions the images into a chosen number of clusters so that each image sits in the cluster whose center it is closest to. The number of clusters was not assumed. K-Means was run for every cluster count from 4 to 10 and each result was scored with the silhouette coefficient (Rousseeuw, 1987), which rates, for every image, how much closer it is to its own cluster than to the next nearest cluster, on a scale from negative 1 to 1. The count with the best silhouette score was kept. As a check that the result does not depend on one algorithm, agglomerative hierarchical clustering was also run at the same cluster count and scored the same way.

Because the seed topics are known, the clusters can also be graded from the outside with two standard measures. Purity asks, for each cluster, what fraction of its members come from its single largest topic, averaged over the dataset, so a purity of 1 would mean every cluster is single topic. Normalized mutual information, or NMI, measures how much knowing an image's cluster tells you about its topic, corrected for chance, with 1 meaning the clustering and the topics carry the same information. Neither measure is available in a real deployment, where no labels exist. They are used here only to verify that the unsupervised result is meaningful.

### 3.3 Algorithmic music selection

The music library is four original tracks synthesized in code with NumPy, so the project depends on no copyrighted audio. Each track is a chord progression with a generated melody and has a numeric profile. The calm track runs at 72 beats per minute in a major key with soft pad tones. The warm track runs at 94 beats per minute in a major key with a mellower timbre. The dramatic track runs at 82 beats per minute in a minor key in a low register. The bright track runs at 116 beats per minute in a major key in a high register with plucked, brighter tones.

Selection works as follows. For each cluster, the system averages four visual statistics over its images: brightness, saturation, warm versus cool color balance, and edge density. Each statistic is then normalized across the clusters in the run, so the brightest cluster scores 1 on brightness and the darkest scores 0, and likewise for the others. The normalized statistics are combined into a target profile. Edge density sets the target tempo, because detailed, busy images carry more visual energy. Brightness sets the target register, and brightness together with saturation sets how major or minor the target is, with a floor so that only a cluster that is both dark and muted leans fully minor. Saturation sets the target timbre, and color balance sets the target warmth. The system then computes a weighted distance between the cluster's target profile and each track's profile and selects the track with the smallest distance. The rule is fixed in code, deterministic, and saved with its scores for every cluster, so the choice for any album can be explained and reproduced exactly.

### 3.4 Slideshow video generation

For each selected cluster, the system takes the images closest to the cluster center, the most representative members, and renders them as a 1280 by 720 video with FFmpeg. Each image is shown for 2.6 seconds with a short fade in and fade out. Because the photographs have mixed orientations, each frame places the full image, scaled to fit, over a blurred and darkened enlargement of itself, so no part of any photograph is cropped away and no black bars appear. The selected music track is laid underneath with a fade out at the end.

## 4. Results

### 4.1 Choosing the number of clusters

The silhouette scores for K-Means rose steadily from 0.106 at 4 clusters to a peak of 0.150 at 8 clusters, then fell to 0.143 at 9 and 0.141 at 10. The system therefore kept 8 clusters. Agglomerative clustering at 8 clusters scored 0.147, nearly identical, which indicates the grouping structure is in the data rather than an artifact of one algorithm. A silhouette of 0.15 is modest in absolute terms, and that is expected. These are real photographs of overlapping natural categories, not synthetic points, and categories such as snowy mountains genuinely sit between topics. The external scores say the groups are nonetheless meaningful: purity is 0.821, meaning 82 percent of all images sit in a cluster dominated by their own topic, and NMI is 0.739, a strong agreement between the found clusters and the seed topics for an unsupervised method that never saw the labels.

### 4.2 The eight clusters

Cluster 0 holds 31 images and is 97 percent dogs, almost entirely close portraits, with a single snowy image of a dog. Cluster 1 holds 19 images and is entirely beach, bright shoreline scenes. Cluster 2 holds 27 images and is 96 percent city, streets and skylines, with one beach image. Cluster 3 is the largest at 42 images and the loosest: 67 percent forest, with flower, beach, and mountain images that share dense green vegetation. It reads as a greenery cluster rather than strictly a forest cluster. Cluster 4 holds 38 images, 71 percent mountains with 9 snow images, and its most central members are snowy peaks, so it is best described as high country where mountains and snow meet. Cluster 5 holds 30 images and is 90 percent desert; it is also the warmest and one of the brightest and most saturated clusters, matching its sand and sky content. Cluster 6 holds 22 images and is 96 percent flowers, with the highest average saturation of any cluster at 0.47. Cluster 7 holds 31 images and is 61 percent snow, but it also absorbed city streets, forest, and flower images photographed under snow cover, so it is a winter cluster defined by snow itself rather than by one topic.

Two patterns in these results are worth noting. First, where the system disagrees with the seed topics, it usually disagrees for a visible reason. Snow is a stronger visual signal than the scene underneath it, so snowy scenes gather together across topics, and dense vegetation pulls flower and shoreline images toward the forest group. Second, the clusters the system is most confident about, dogs, beach, city, desert, and flowers, are the ones a person would name the same way, which is the behavior an automatic album organizer needs.

### 4.3 Music selections

The selection rule gave the warm track to the desert and flowers clusters, the two warmest and most saturated groups. It gave the calm track to the beach, high country, and winter clusters, the bright and open groups. It gave the dramatic, minor key track to the dogs, city, and forest clusters, which are the three darkest and most muted groups in the set. The bright track was not selected for any cluster in this run. Reading the target profiles shows why: no cluster was simultaneously bright, saturated, and highly detailed, which is the corner of the profile space the bright track occupies. The rule and every score are saved in the output files, so each of these choices can be traced to the numbers that produced it.

### 4.4 Generated videos

Slideshow videos were generated for the three largest clusters: the greenery cluster (Cluster 3, music: dramatic), the high country cluster (Cluster 4, music: calm), and the dog portraits cluster (Cluster 0, music: dramatic). Each video presents the 16 images closest to its cluster center at 1280 by 720 resolution and runs 41.6 seconds. The delivered files are slideshow_cluster_3.mp4 (5.3 MB), slideshow_cluster_4.mp4 (4.2 MB), and slideshow_cluster_0.mp4 (2.5 MB), each verified to contain an H.264 video stream and an AAC audio stream carrying the selected track. The video builder accepts any cluster id, so an album can be generated for every cluster with the same command.

## 5. Discussion and limitations

The system meets the assignment's requirements end to end, and its main strength is that every stage can be inspected. The clustering is not a black box claim: it is scored intrinsically by silhouette and extrinsically by purity and NMI, and the contact sheet montages shipped with the output let a reader see each cluster directly. The music selection is likewise a written rule over measured quantities, not a lookup table written by hand for these particular clusters, and it would make its choices for new clusters the same way.

Several limitations are clear. First, the silhouette scores are modest, and two of the eight clusters are defined by a visual condition, snow cover and dense greenery, rather than by a single topic. A larger dataset, or features from a vision model trained with image and text together, would likely separate those cases better. Second, the seed topics come from search queries, so the labels themselves contain some noise, which caps the purity any method could score. Third, the music library is small and synthesized, and the mapping from visual statistics to musical character is a reasonable design choice rather than a learned or validated one. A natural extension is to test the mapping with listeners, or to learn it from data pairing images with music. Fourth, the assignment allows capture time or GPS metadata to refine grouping. The delivered photographs carry no consistent metadata of that kind, so that refinement was not used; the visual features alone carried the grouping. Finally, the videos use a fixed time per image. Letting the music's beat set the cut points would tie the two halves of the system together more tightly.

## 6. Conclusion

A pre-trained ResNet-50, used purely as a feature extractor, turns 240 unorganized photographs into a space where K-Means finds eight clusters that agree with human topic labels at 0.821 purity and 0.739 NMI without ever seeing those labels. On top of that grouping, a fixed and documented rule over brightness, saturation, warmth, and edge density selects background music per cluster, and the pipeline renders finished slideshow albums. The result is a complete automatic photo album system whose decisions, from cluster counts to music tracks, can each be traced back to a number and a rule.

## References

He, K., Zhang, X., Ren, S., and Sun, J. (2016). Deep Residual Learning for Image Recognition. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 770 to 778.

MacQueen, J. (1967). Some Methods for Classification and Analysis of Multivariate Observations. Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability, pages 281 to 297.

Rousseeuw, P. J. (1987). Silhouettes: A Graphical Aid to the Interpretation and Validation of Cluster Analysis. Journal of Computational and Applied Mathematics, 20, pages 53 to 65.

Wikimedia Commons. https://commons.wikimedia.org. Image dataset collected through the Wikimedia Commons API; per image creators and licenses are listed in data/manifest.csv.

## Deliverables and how to run

The written report is this document. The source code is in the src folder (collect_dataset.py, analyze.py, make_music.py, build_videos.py) with install and run instructions in README.md. The dataset reference is the manifest at data/manifest.csv, which lists every image with its creator, license, and source page on Wikimedia Commons. The generated videos are slideshow_cluster_3.mp4, slideshow_cluster_4.mp4, and slideshow_cluster_0.mp4 in the output folder, with cluster assignments, summaries, montages, the PCA scatter plot, and the music selection scores alongside them.
