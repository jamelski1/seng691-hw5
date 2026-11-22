# Assignment 5 demo and walkthrough video: recording plan

Dr. Samarah confirmed on Nov 21, 2026 that the submission needs a
separate video with a demonstration and a walkthrough, in addition to
the generated slideshow videos. Target length: 5 to 7 minutes,
narrated screen recording, recorded by Jamel on his own machine.

## Before recording, stage these tabs and files

1. The Colab notebook, already executed, scrolled to the top:
   https://colab.research.google.com/github/jamelski1/seng691-hw5/blob/main/seng691-hw5.ipynb
2. The live site home page: https://photo-albums-y7ir.onrender.com
3. A finished results page as backup, in case a live run is slow on
   camera. Run a job shortly before recording and keep its results
   page open: job results are stored on the server's disk and are
   wiped whenever the service redeploys, so an older results link
   will not work.
4. One generated slideshow file ready to play from the notebook run
   (output/slideshows in the repo outputs, or the Drive copy).
5. Notifications off, microphone checked, browser zoom at 100%.

## Part 1: the notebook run, about 3.5 minutes

This is the assignment itself, so it gets the most time.

- (0:00) Say what the system does in one or two sentences: it groups
  200 or more photos into collections by content and builds each
  collection a slideshow video whose music is chosen by a rule from
  the collection's look, nothing picked by hand.
- (0:30) Dataset: the manifest and the 240 photos, 30 each across
  eight topics (mountains, beach, city, forest, desert, snow,
  dogs, and flowers), all from Wikimedia Commons with licenses
  recorded.
- (1:00) Features: the measured statistics per photo, then the
  ResNet-50 embeddings, 2048 numbers per photo, and why content
  features group better than color alone.
- (1:45) Clustering: KMeans with k chosen by silhouette, k = 8.
  Show the PCA scatter and the montages. Give the scores against
  the known topics: purity 0.821, NMI 0.739.
- (2:30) Music: show the selection rule, how brightness, saturation,
  warmth, and detail become a target and the lowest weighted
  distance wins. Show the Jamendo option cell: a reader can paste
  their own free Jamendo client ID and get recorded Creative
  Commons tracks instead; the submitted run used the synthesized
  library. Show music_choices.json.
- (3:15) Play 15 to 20 seconds of one generated slideshow.

## Part 2: the web service add-on, about 2.5 minutes

- (3:45) Frame it in one sentence: the same pipeline is also
  deployed as a web service, an add-on; the notebook above is the
  full assignment run.
- (4:00) On the home page, click "Run the 240-photo example" (or
  upload the 24 test photos if a shorter run is better on camera).
  Show the status page and name the stages as they complete.
- (4:45) On the results page: the run summary, one collection's
  cover sheet and statistics, its matched track with the Jamendo
  attribution and the search the rule ran, and the scoring table.
- (5:30) Play the collection's slideshow and point out the beat
  timing: each photo stays for a whole number of beats and the
  fades land on the beat, measured from the track itself.
- (6:00) Close: where everything lives. Report, code with install
  instructions, dataset link, the generated slideshow videos, and
  this demonstration video.

If the live run is still working when Part 2 starts, begin the run,
switch back to the notebook tab to finish Part 1, then return to the
finished results page (or use the backup results tab above).

## Submission checklist with the demo included

- Report (report.pdf)
- Source code with install instructions (the GitHub repo)
- Dataset link (the manifest in the repo)
- Generated slideshow videos (at least one; three are in the
  package outputs)
- This demonstration and walkthrough video, saved as
  SENG691-Assignment5-Demo-Walkthrough.mp4 and included with the
  Blackboard submission
