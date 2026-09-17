# Multimodal Vision: From Pixels to Models

Fourteen scripts answering one question from three directions:
**how does the information in a picture become something a program can use?**

Scripts 1-7 build the answer from the pixels up, with no pretrained weights anywhere:
colour, edges, shapes, gradients, corners, motion, and then the networks that learn
those steps instead of being told them. Scripts 8-11 hand the picture to a model
somebody else trained and audit what comes back. Scripts 12-14 return to training,
with the arithmetic of a convolution, what input resolution costs, and where a
detection metric comes from.

Every script generates its own inputs, and the parts that also read a public dataset
take its path on the command line. Nothing here is scored against an answer that was
not known in advance: either the data was drawn by the script, or the labels came
with it.

Every script here generates its own inputs. Nothing is downloaded, no photograph or
document is shipped, and every ground truth is recorded at the moment it is drawn.
That is not a convenience: **an audit against data whose answer is only approximately
known is not an audit.**

| | Calling a model | Training one |
| :--- | :--- | :--- |
| Shape | A prompt and an image go out, JSON comes back | Labelled data goes in, weights come out |
| Precondition | The subject appeared in the general training data | The subject is specific to one setting |
| Deliverable | Free text or JSON fields | Boxes, classes and confidences |
| Failure | **A confident answer with a wrong field** | **A metric that computes and means nothing** |

The two halves also touch: a document parser's layout stage is itself a detector,
usually a YOLO variant. The half that calls a model has the half that trains one
inside it.

---

## Choosing between the two

| Task | Route | Why |
| :--- | :--- | :--- |
| Unstructured understanding — a scene, a form, a document in an unfamiliar script | Vision language model | Semantic reading is the point; paraphrase is acceptable |
| Exact character recovery — serial numbers, amounts, standard forms | Dedicated OCR | The literal string matters; paraphrase is a defect |
| Common objects, counted or located | A pretrained detector | The object is in the public datasets already |
| One setting's own parts and defects | A detector trained on a few hundred images | The object is in no public dataset |
| Anything that must return coordinates | A detector, or a model that emits boxes | A chat endpoint often describes instead of pointing |

Two of these are load-bearing for the scripts below and are demonstrated rather than
asserted: the third row of the failure taxonomy in script 08 is a character-level
misread, which is the OCR boundary; and script 09 measures what a model returns when
asked to point.

---

## 1. Colour tracking and optical flow

`01_color_tracking_and_optical_flow.py`

A clip is rendered frame by frame — a coloured ellipse on a known path that grows,
turns, and loses more than half its light halfway through — so every stage of the
classical tracking pipeline can be scored against the mask that drew it. The run saves
the clip as `synthetic_tracking.mp4` and four key frames — the first, the last bright
one, the first dimmed one and the last — so the motion and the light change can be
watched rather than taken on trust.

### Colour: which space survives the light

| Rule | IoU, bright frames | IoU, dimmed frames |
| :--- | ---: | ---: |
| Box in the BGR cube | 0.972 | **0.000** |
| Hue and saturation in HSV | 0.972 | **0.984** |

Dimming multiplies B, G and R by one factor. Hue and saturation are functions of the
ratios between the channels, which that factor leaves alone; the box tests absolute
levels, so the object walks out of it. The object's mean colour goes from
`BGR [165 91 26] HSV [106 216 165]` to `BGR [74 41 11] HSV [106 217 73]`: **only V
moved.** (OpenCV holds the channels as B, G, R in that order, which is what the box
is written against.)

**The scope is exactly that one kind of change**: a light that dims scales all three
channels together. A light that changes colour does not, and hue moves with it.

`colour_thresholds.png` shows it directly: frame 0 and the first dimmed frame, each as
the frame, the BGR box mask, the HSV mask and the true mask. On the dimmed row the BGR
mask is empty and the HSV mask still holds the ellipse.

### Morphology and connected components

The raw mask of the first frame holds **41 components** — the ellipse plus scattered
specks. Erosion drops everything thinner than the 3x3 element, dilation restores the
rim, and a closing fills the pinholes: **1 component, IoU 1.000 against the truth.**
`morphology_stages.png` lays the four stages beside the true mask, one row for the
first frame and one for the last, so the specks disappearing and the holes closing
can be seen stage by stage.

Two labelling algorithms are written out and reconciled with OpenCV:

| Connectivity | Two-pass | Flood fill | OpenCV |
| :--- | ---: | ---: | ---: |
| 4 | 42 (from 55 provisional labels) | 42 | 42 |
| 8 | 41 (from 47 provisional labels) | 41 | 41 |

All three group exactly the same pixels, and the largest component's bounding box
agrees to the pixel. The gap between 42 and 41 is two squares that touch only at a
corner. The provisional-label counts are what the equivalence table is for: a U
shape is labelled as two arms until the scan reaches the bar joining them.
`component_boxes.png` puts the two results side by side, 4-connectivity on the left
and 8 on the right, with a box around every component of at least 20 pixels: the
corner-touching squares get two boxes on the left and one on the right, and the specks
are counted but left unboxed.

### Back-projection, mean shift and CAMSHIFT

A 16-bin hue histogram of the first frame, back-projected onto every frame, matches
`cv2.calcHist` and `cv2.calcBackProject` **exactly (largest gap 0.0000 and 0)**. The
saturation gate is what makes it usable: of the probability mass in the last frame,
**0.489 lands on the object without the gate and 0.989 with it** — a grey pixel's hue
is set by noise and lands in every bin.

| | First third | Last third |
| :--- | ---: | ---: |
| Mean shift, fixed window: centre error | 1.66 px | 5.60 px |
| Mean shift: share of the object inside the window | 85.5% | 42.6% |
| CAMSHIFT: centre error | 0.03 px | 0.02 px |
| CAMSHIFT: long semi-axis error | 0.58 px | 0.55 px |
| CAMSHIFT: angle error | 0.27° | 0.27° |
| CAMSHIFT: share of the object inside the window | 100.0% | 100.0% |

The object grows from 1297 to 4037 pixels. A window fixed at the first frame's size
still finds the densest part, but sees less of it every frame; CAMSHIFT reads the
size and orientation out of the second moments and follows. The hand-written mean
shift picks **the same window as `cv2.meanShift` in 90 of 90 frames.**

`trajectory.png` draws all four centre paths over the last frame, with a legend, widest
first so none hides another: the true centre as a wide green band, the CAMSHIFT centre
as a narrower magenta line on it, the mean shift window centre in yellow, and the
largest component's centroid as a thin red line on top. Red and magenta run almost
together along the green band; yellow is the one that leaves it.

`backprojection.png` puts the last frame beside its back-projection without and with
the saturation gate, so the grey background lighting up without the gate can be seen.
`tracker_windows.png` shows frames 0, 30, 60 and 89 with the fixed mean shift window in
yellow and the CAMSHIFT ellipse in magenta, which lies on the object's own outline: the yellow box
stays the first frame's size while the object outgrows it, and the ellipse grows and
turns with it.

Two things bound what this table says. **Both trackers start from a box taken off the
first frame's true mask**, so these are the errors of keeping hold of an object, not
of finding one — the step above does the finding, from scratch, every frame. And the
comparison is between a window that cannot resize and one that can, on a clip where
the object doubles in size: with an object of fixed size and orientation, the fixed
window is the simpler rule and there is nothing to gain.

### Corners and motion

The structure tensor, the same matrix Harris scores and Lucas-Kanade inverts:

| Point | λ₁ | λ₂ | R |
| :--- | ---: | ---: | ---: |
| Flat, inside a square | 0.0000 | 0.0000 | 0.00000 |
| Edge, middle of a side | 1.4283 | 0.0000 | **-0.10200** |
| Corner | 0.9739 | 0.3369 | **+0.24218** |

All **16 true corners** are found, with no detection on the disk that has none.
`harris_corners.png` shows the response R as a colour map — red at corners, blue along
edges where R is negative, flat regions in between — with the three sample points of
the table marked, beside the detections.

Both motion methods run on one blurred-noise texture and two copies of it shifted by a
known amount. `motion_pair.png` shows the three frames side by side — before, after
(3.6, -2.4) and after (11.3, 6.8) — with a red cross at the same pixel in each, so the
texture can be seen sliding past a fixed point. The result images draw, on the first
frame, the true shift as a green arrow and each recovered vector as a yellow one, both
three times longer, so a success shows the two arrows lying together and a failure
shows them apart. `block_flow.png` has block matching on the small shift and on the
large one; `lucas_kanade_flow.png` has one level on the small shift, one level on the
large shift, and the 3-level pyramid on the large shift.

| Method | True shift | Median error |
| :--- | :--- | ---: |
| Block matching, 16x16, search radius 8 | (3.6, -2.4) | 0.58 px (100% within 1 px) |
| Block matching | (11.3, 6.8) | **8.43 px** — no correct candidate is in the search square |
| Lucas-Kanade, one level | (3.6, -2.4) | **0.010 px** |
| Lucas-Kanade, one level | (11.3, 6.8) | **12.816 px** |
| Lucas-Kanade, 3-level pyramid | (11.3, 6.8) | **0.008 px** |
| `cv2.calcOpticalFlowPyrLK` | (11.3, 6.8) | 0.008 px |

The last row of the aperture problem, on the rectangle scene shifted by (2.6, 1.7):

| Points | Smallest eigenvalue of G | Median error |
| :--- | ---: | ---: |
| Rectangle corners | 45892.4 | 0.014 px |
| Middles of the sides | **0.0** | 2.150 px |

On the three horizontal edges the recovered vectors are `(0.00, 1.70)` three times:
**the component across the edge is exact and the component along it is unobservable.**
`aperture_problem.png` draws both cases with the true shift in green and the recovered
vector in yellow, ten times longer: at the corners the two arrows coincide, and at the
middles of the sides the yellow arrow keeps only the part across the edge.

---

## 2. Edges and Hough voting

`02_edges_and_hough_voting.py`

Four lines in Hesse normal form and three filled disks, drawn at known parameters
under Gaussian noise of σ 25, then recovered.

### The Gaussian, and what each window keeps

| Size | σ | Mass inside the window | Noise left (from 25.0) | Edge strength left |
| ---: | ---: | ---: | ---: | ---: |
| 3 | 0.5 | 0.999 | 16.02 | 0.921 |
| 3 | 1.5 | **0.479** | 8.43 | 0.776 |
| 7 | 1.0 | 0.999 | 7.06 | 0.714 |
| 7 | 1.5 | 0.965 | 4.88 | 0.565 |

A window too small for its σ keeps less than half of the kernel's weight. It is
renormalised and still smooths, but it is no longer the Gaussian σ describes.

### Canny, stage by stage

A single threshold on the Sobel magnitude scores **precision 0.600** on the noisy
image and **1.000** after smoothing. Quantising the direction to the four lines
through a pixel's neighbours and comparing along the gradient removes **62.2% of the
total magnitude**, thinning 18830 pixels above the low threshold to 6691.

| Pair | Low | High | Edge pixels | Precision | Recall |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Both low | 20 | 50 | 11843 | **0.284** | 1.000 |
| Both high | 150 | 250 | 3045 | 1.000 | 0.913 |
| Wide spacing | 40 | 250 | 3161 | 1.000 | **0.947** |
| High threshold alone, no tracing | – | 250 | 2616 | 1.000 | 0.786 |

The wide pair lets the high threshold decide what an edge is and the low one decide
how far it is followed. Against `cv2.Canny` on the same smoothed image: **3270
pixels against 3269, 100% of each within 1 px of the other.** The hand-written
version snaps the gradient to four directions rather than interpolating between
neighbours, so it is a readable Canny rather than a line-for-line copy of OpenCV's.

Precision and recall here allow **2 px of tolerance**: a precision of 1.000 means every
detected edge pixel lies within 2 px of a true boundary, not that the two maps are
identical pixel for pixel. Smoothing and discretisation move an edge by a pixel or
two, which is why a tolerance is used at all.

### Votes

| Search | Accumulator | Votes | Time | Found |
| :--- | :--- | ---: | ---: | :--- |
| Lines, every angle | 801 x 180 | 568 980 | 36 ms | **the 4 true lines are the top 4 peaks** |
| Lines, lane angles only | 801 x 62 | 195 982 | 19 ms | both lane lines; the other two are outside the range |
| Circles, every 6° | 240 x 320 x 34 | 5 865 905 | 409 ms | 2 of 3 disks in the top 10 |
| Circles, along the gradient | same | **191 801** | **22 ms** | **all 3 disks, ranks 1, 2 and 3** |

The gradient at a rim points along the radius, so it names the direction the centre
lies in; sampling every angle spreads the same evidence around a whole ring. Disks
are paired with detections one to one, so no detection counts for two disks.

Both savings rest on a prior being right. The lane restriction cannot find a line
outside its range, however strong. The gradient-directed circle vote trusts each
pixel's gradient; the Canny edges of these disks give reliable directions, but on a
blurred or textured rim a wrong direction sends both votes to the wrong centre.

### The generalised Hough transform

The template is a yellow polygon on blue, binarised by a hue lookup, giving **279
edge points filed into 43 of the 72 gradient-angle bins**. Against a scene holding
the shape among three distractors, the peak lands **0.07 px** from the true reference
point. Extended over 6 scales and 19 rotations — 114 hypotheses in 0.1 s — the
strongest peak is **scale 0.7, rotation 25°, position error 0.33 px**, which is the
transform that was applied. The runner-up hypotheses are the neighbouring rotations
at half the peak height. The rotation step equals the 5° R-table bin, so each step is
a whole-bin shift, and the true 25° lies on that grid; an angle between steps would
be recovered only to the nearest one. This is a synthetic template among simple
distractors, and it shows the mechanism rather than robustness on cluttered images.

### What the images show

Every panel is labelled with its setting and, where there is one, its score.

| Image | Panels |
| :--- | :--- |
| `scene.png` | clean scene, noisy scene, true boundary |
| `gaussian_grid.png` | the nine size × σ combinations, each with noise and edge strength left |
| `sobel_threshold.png` | one threshold on the noisy and on the smoothed image |
| `canny_stages.png` | smoothed, magnitude, continuous and quantised direction, after suppression, what suppression removed |
| `hysteresis_pairs.png` | the three threshold pairs and the high threshold alone |
| `hough_lines.png` | accumulator and top lines over all angles, and over the lane angles only; lines matching a true one in green |
| `hough_circles.png` | centre votes and top ten circles for sampling every 6° and for voting along the gradient; true disks in green |
| `ght_match.png` | template, translation-only votes and match; peak per scale and rotation, winning votes and match |

---

## 3. HOG and Haar detectors

`03_hog_and_haar_detectors.py`

The two hand-built descriptors that carried detection before learned features: one
counts gradient directions, the other compares rectangle sums.

### Gradient orientation histograms under a small turn

A star is turned by 5°, a quarter of a 20° bin, and the cell histograms are compared:

| Voting | Distance, star against turned star | Relative to the star's own norm |
| :--- | ---: | ---: |
| Nearest bin | 201.982 | 0.913 |
| Split between the two nearest bins | 151.284 | 0.791 |
| Split between bins **and** between the four surrounding cells | **101.999** | **0.627** |

With the whole vote in one bin, a direction crossing a boundary moves all of its
weight; split between neighbours, it moves a quarter of it. The same holds for
pixels the turn carries across a cell border. Interpolation makes the change smaller
and smoother; it does not make the descriptor rotation invariant, and the turned star
still differs by 0.627 of its own norm.

### The descriptor

A 64x128 window gives 16x8 cells, 15x7 blocks of 2x2 cells, and a descriptor of
**3780 = 105 x 36** values after L2-Hys block normalisation and a final normalisation
over the whole window.

| Window | Person 1 | Person 2 | Person 3 | Mean |
| :--- | ---: | ---: | ---: | ---: |
| Unknown person | 0.468 | 0.453 | 0.504 | **0.475** |
| Unknown car | 0.861 | 0.841 | 0.863 | **0.855** |

Over 40 fresh windows of each kind, measured against the mean of the three reference
people: **people 0.456, clutter 0.675, cars 0.797** — the ordering the descriptor is
built to produce. This half validates the descriptor on rendered windows by distance
alone; no classifier or decision threshold is trained on it, which in a HOG detector
is the job of a linear SVM. The trained classifier in this script is the Haar one.

### Rectangle features and the integral image

| Window | Enumerated | Closed form | Horizontal | Vertical |
| :--- | ---: | ---: | ---: | ---: |
| 24x24 | 86 400 | **86 400** | 43 200 | 43 200 |
| 16x16 | 17 408 | **17 408** | 8 704 | 8 704 |

The hand-written integral image matches `cv2.integral` exactly (largest difference
0.0). On 2000 random rectangles over a 640x480 image, summing by slicing takes
**10.1 ms** and four lookups each take **0.17 ms**, for the same answers to 1.76e-10.
Those are single passes rather than a benchmark; what they show is that one cost
grows with the rectangle's area and the other does not.

### AdaBoost over every feature

Twenty rounds, each choosing one feature, one threshold and one polarity from all
17 408 of them (each threshold sits halfway between two distinct values, so a run of
equal values is never split and no training value lies on the threshold under either
polarity; on 300 random tie-heavy problems the error the search reports equals the
error the stump then makes, and the brute-force optimum, to 3.33e-16); the first feature found is a **top/bottom pair at (3, 5), each 10x3**,
which is the eye band against the cheeks below it.

| Threshold | Faces found | False alarms | Accuracy |
| ---: | ---: | ---: | ---: |
| 0.3 | 100.0% | 7.7% | 96.2% |
| 0.5 | 99.9% | **0.4%** | **99.8%** |
| 0.7 | 93.8% | 0.0% | 96.9% |

On real crops — 3000 TinyFace faces against 3000 CIFAR-10 images, 16x16 and
variance-normalised — the same twenty rounds reach **87.1% of faces at 13.3% false
alarms, 86.9% accuracy** at threshold 0.5, against 50.0% for predicting the larger
class. The first round's weighted error is 0.222 on real faces against 0.037 on the
rendered ones: **the same procedure, a harder problem.** The TinyFace images come
already cropped to the face, and this is classification of 16x16 windows, not a
detector scanning whole photographs at every position and scale; 86.9% is not a
face-detection benchmark score.

### What the images show

| Image | Panels |
| :--- | :--- |
| `windows.png` | the three reference people, the test person, the test car and a clutter window |
| `face_windows.png` | eight rendered faces above eight non-faces |
| `gradient_direction.png` | each reference person with gradient direction as colour and magnitude as opacity |
| `cell_histograms.png` | the star and the turned star with their cell histograms drawn in, then how much each cell changes under nearest, orientation and orientation-plus-spatial voting |
| `hog_descriptor.png` | cell histograms of the three people, the test person and the test car with their mean distances, and a chart of the distance of 40 new people, cars and clutter windows |
| `haar_feature_types.png` | one left/right and one top/bottom pair on a 24x24 grid, with how many of each there are |
| `integral_image.png` | a face window with the eye-band box, and its integral image with corners A, B, C, D giving the same sum |
| `adaboost_scores.png`, `adaboost_scores_tinyface.png` | how test faces and non-faces spread over the strong-classifier score, with the thresholds of the table |
| `haar_features.png`, `haar_features_tinyface.png` | the mean face with the first three chosen features |

---

## 4. Training mechanics

`04_training_mechanics_xor_softmax_batchnorm.py`

### XOR, and why a hidden layer

A single sigmoid unit, trained from 100 different starting points, reaches **75% at
best**. A half-plane contains the midpoint of any two points it contains, and both
XOR pairs share the midpoint (0.5, 0.5), which would have to lie on both sides.

Two-layer networks, full batch, learning rate 2.0, solved when every output is within
0.1 of its target:

| Hidden units | Initial sd | Solved | Median epochs |
| ---: | ---: | ---: | ---: |
| 2 | 0.1 | 54% | 5245 |
| 2 | 1.0 | 79% | 661 |
| 4 | 1.0 | **100%** | 505 |
| 8 | 1.0 | **100%** | **392** |

Two hidden units are the fewest that can represent XOR; the runs that do not solve it
settle at a loss of about 0.125, where both units compute nearly the same thing.
Weights that start near zero start the two units nearly identical, and the gradient
needs thousands of epochs to separate them.

### Softmax and cross-entropy

The analytic gradient (softmax minus one-hot, divided by the batch) matches a central
difference to **8.11e-10**. The loss is computed as logsumexp minus the true logit, with
no epsilon inside a log, so the function differenced is exactly the one differentiated. On logits `[1000, 1001, 1002]` the definition as written
returns `[nan, nan, nan]`; subtracting the largest logit, which cancels in the ratio,
returns `[0.09, 0.2447, 0.6652]`.

### Two networks on digits

| | Rendered digits | MNIST |
| :--- | ---: | ---: |
| 784-256-10 in numpy, 5 epochs, 203 530 parameters | 89.35% | **97.84%** (2.16% error) |
| CNN, 3 epochs, 156 010 parameters | 94.85% | **99.03%** |

The CNN has fewer parameters and a higher score. 200 960 of the MLP's parameters are
in the first layer alone, which learns one weight per pixel per unit and shares
nothing between positions, and weight sharing is the natural explanation. The two
networks also differ in optimiser, depth, epochs, BatchNorm and Dropout, so the
comparison shows that this CNN beats this MLP, not that sharing alone accounts for
the gap. Test accuracy is printed every epoch only to show the curve; nothing is
chosen from it.

### Training mode against evaluation mode

Both modes, same weights, on MNIST:

| Measurement | Result |
| :--- | :--- |
| Training images scored in evaluation mode / training mode | 99.34% / 99.24% |
| Test predictions that change with the mode | **56 of 10 000** |
| Dropout p=0.3 in training mode on a vector of ones | 0.299 zeroed, survivors scaled to 1.4286 = 1/(1-p), mean 1.0011 |
| BatchNorm in evaluation mode against the running-statistics formula | gap 9.54e-07 |
| BatchNorm in training mode against this batch's own mean and variance | gap 1.43e-06 |
| Running variance after one training-mode batch, against 0.9 old + 0.1 unbiased batch | gap 1.49e-08 |

The first 1000 test images fed **one at a time**, with each layer type switched
separately:

| BatchNorm | Dropout | Accuracy |
| :--- | :--- | ---: |
| training | training | **90.10%** |
| training | evaluation | 91.60% |
| evaluation | training | 98.40% |
| evaluation | evaluation | 98.80% |

Dropout alone costs 0.4 points; BatchNorm alone costs 7.2. With a single image,
BatchNorm normalises each channel by that image's own spatial mean and variance,
which erases how strongly a feature map responded overall. That is part of the
evidence; the running statistics carry it instead.

`torch.no_grad()` only stops gradients being recorded. Inside it a training-mode
network still drops activations and still updates BatchNorm's running statistics,
which is why the probes run on copies of the trained model.

### What the images show

Every image below is written twice, once per run: `outputs/training_mechanics/synthetic/`
holds the run on rendered digits and `outputs/training_mechanics/mnist/` the run on MNIST.
The file names are the same in both, which is why they are kept apart.

| Image | Panels |
| :--- | :--- |
| `xor_boundaries.png` | output over the input plane with the 0.5 line: the best single unit, the two-unit network at random weights, a trained seed that solves XOR, a trained seed that gets stuck |
| `xor_loss.png` | loss curves of three solved and three stuck two-unit seeds |
| `softmax_gradient.png` | the analytic and numerical gradient side by side, and the stable softmax of logits 1000, 1001, 1002 |
| `digits.png` | the first 20 training images with their labels |
| `training_curves.png` | loss and test accuracy per epoch for the numpy MLP and the CNN |
| `misclassified.png` | the first wrong test images of each network, true class against prediction |
| `cnn_feature_maps.png` | one test image and eight channels after each of the three convolution blocks |
| `mode_effects.png` | a dropout mask on a vector of ones, batch against running means of the first BatchNorm, and the four single-image mode combinations |

---

## 5. Grid detection and pose assembly

`05_grid_detection_and_pose_assembly.py`

Two dense output tensors, two ways of reading objects out of them.

### Anchors and the grid

Clustering over the training shapes with 1 - IoU as the distance and the median shape
as each centre — the YOLO anchor recipe, k-means in its assign-and-update loop but not
in its Euclidean distance or its mean:

| k | Anchors (w, h) | Mean best IoU |
| ---: | :--- | ---: |
| 1 | (35, 35) | 0.485 |
| 3 | (17, 17), (31, 31), (51, 55) | **0.713** |
| 5 | (15, 13), (21, 21), (29, 29), (41, 41), (55, 65) | 0.769 |

A 160x160 image at stride 16 with 3 anchors holds **300 slots**. Encoding the centre
as an offset inside its cell and the size as log(w / anchor_w) and decoding it back
costs **3.81e-06 px**, and only **2 of 12 066 boxes** find their slot already taken.

### Training and scoring

| Epoch | xy | wh | obj | noobj | class | total |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.612 | 1.817 | 3.175 | 24.123 | 2.714 | 33.441 |
| 5 | 0.207 | 0.211 | 0.461 | 0.516 | 0.111 | 1.505 |
| 30 | 0.033 | 0.035 | 0.007 | 0.011 | 0.001 | 0.086 |

The xy, wh and noobj columns are already weighted (by 5.0, 5.0 and 0.5), so the total
is their plain sum.

The hand-written per-class suppression keeps **the same boxes as
`torchvision.ops.batched_nms` in 500 of 500 test images** — agreement on these
candidates, not a proof of equivalence on every input — and the detector reaches
**mAP@0.5 = 0.974** (box 0.978, disk 0.979, triangle 0.967) over 1493 unseen boxes.
The images are rendered and their objects overlap by at most 0.1 IoU, so that score
shows the grid, anchor, loss, decode, suppression and mAP chain working end to end,
not what a small detector reaches on photographs; no detector is trained on COCO
below.

### Why every term rises near epoch 25

`05b_loss_spikes_and_step_size.py`

In the table above the loss falls smoothly, yet `loss_terms.png` shows every term rising
together in epochs 25 and 26 before falling back. The supplement reproduces that run with
every optimiser step recorded, then tests the explanations that come to mind. It changes
nothing about the detector, and its trajectory matches this one epoch for epoch.

**It is not one unlucky batch.** Across the rise the heaviest batch of an epoch costs
**3.20x and 2.24x** the epoch's median, against **2.16x to 2.44x** in the quiet epochs 21
to 23 — no outlier step exists. The rise is not a jolt either: the parameter update norm
and the loss climb together over hundreds of steps and decay the same way, which
`spike_anatomy.png` shows as a wide smooth hill rather than a spike.

**It is not Adam's denominator shrinking.** Between the quiet epochs and the rise:

| | Quiet, epochs 21-24 | Rising, epochs 25-26 | Ratio |
| :--- | ---: | ---: | ---: |
| Total loss | 0.1227 | 0.4092 | 3.33 |
| Gradient norm | 7.758 | 12.818 | 1.65 |
| Parameter update norm | 0.0357 | 0.1090 | 3.05 |
| Mean sqrt(v), second moment | 0.01052 | 0.00954 | **0.91** |
| Mean absolute m, first moment | 0.00058 | 0.00159 | 2.76 |
| Gradient cosine with the previous step | 0.386 | 0.358 | 0.93 |

The second moment does not dip during the rise; it declines slowly throughout and is
slightly higher at the peak than just before it. Successive gradients do not line up
either — that cosine moves inside its usual band rather than rising. What does change is
how far Adam travels per unit of gradient, **x1.82**, so the larger steps are not only
larger gradients. Which coordinates produce that is not measured here.

**The whole network moves, not only the head.** Relative movement per step grows by
**x2.0 to x5.1** across the seventeen parameter tensors; the 1x1 prediction head grows
least, by **x2.3 to x2.8**. And the terms rise by very different amounts — class **x18.4**
and objectness **x10.3**, against **x2.9** for the centre offsets — because the unbounded
cross-entropies answer to the scale of the logits while the coordinate terms sit behind a
sigmoid.

**Three controls place the cause.** Each trains the same detector from the same seed:

| Run | Break-up starts at | Loss it left |
| :--- | ---: | ---: |
| Batch order 3407, rate 1e-3 (the script's own) | epoch 25 | 0.1247 |
| Batch order 12345, rate 1e-3 | epoch 30 | 0.1175 |
| Batch order 3407, rate 5e-4, run to 60 epochs | epoch 56 | 0.0599 |

A different batch order moves the break-up instead of removing it, so it does not belong
to particular batches. Halving the step does not remove it either: it postpones it and
buys roughly half the loss first. Stopping that run at 30 epochs would have been
misleading, since it stands at **0.1326** there, above the level either full-rate run
broke from, and so looks perfectly clean — which is why it runs to 60.

**It repeats.** Four times the original training, at the original settings:

| Starts at | Leaving | Peaking at | Ratio |
| ---: | ---: | ---: | ---: |
| epoch 25 | 0.1247 | 0.5600 | 4.49 |
| epoch 54 | 0.0574 | 0.7342 | 12.79 |
| epoch 86 | 0.0359 | 0.6236 | 17.36 |
| epoch 117 | 0.0218 | 0.0341 | 1.56 |

Spaced **29, 32 and 31 epochs** apart, each leaving from a lower loss than the last, while
the trend underneath keeps falling: **0.0188 at epoch 101**, against 0.0857 after the 30
epochs this module trains. Script 05 stops at 30, so `loss_terms.png` catches only the
first.

That pattern — a break-up that survives a change of data order, that a smaller step
postpones to a lower loss, and that returns at a regular spacing — points at the step size
rather than at the data. What it does not say is what the step size is colliding with,
because none of these runs measures the loss surface. The next section does, and it finds
the obvious reading only half right.

### Direct curvature measurement

`05c_curvature_and_stability.py`

The section above ends on an inference it cannot check. This one measures the curvature of
the loss surface directly, epoch by epoch, on the same training run.

**How.** The detector has 248 488 parameters, so its Hessian would hold 0.1 trillion
entries and is never built. Curvature comes from Hessian-vector products — differentiate
the gradient's projection onto a vector — driven by 12 power iterations, which leaves at
most **9.8e-03** relative drift in an estimate. Everything is measured on **one probe set
fixed before training**: 128 training images in 4 batches of 32, so a change in curvature
cannot be a change of batch. The measurement takes gradients but never a step, restores the
batch-norm statistics its forward passes would otherwise move, and draws its vectors from
its own generator; the 30 epoch losses come out identical to script 05's.

Three curvatures, not one:

| Quantity | What it is |
| :--- | :--- |
| λ_max(H) | the sharpest direction anywhere in parameter space |
| uᵀHu | the curvature along the direction the parameters actually moved |
| λ_max after Adam's scaling | the same surface seen through diag(1/√v̂), which is the one Adam's step size answers to |

**The obvious version of the story is wrong.** Sharpening over training is real — λ_max
climbs from **1450** at epoch 1 to a plateau near **3100** — but that plateau is reached
around epoch 14, ten epochs before anything happens, and λ_max then *falls* into the
break-up: **3073** over the quiet epochs, **2837** at epoch 24, **2609** at epoch 25.
Sampling every 25 steps through epochs 22–27 says the same from inside the event: λ_max
falls from **2899** at epoch 24.2 to **2285** at epoch 25.2 while the loss climbs. What does
grow is how far the parameters actually move each step, from **0.019 to 0.075** — the
learning rate never changes, but Adam's own scaling lets a step cover four times the ground
it did while the loss was flat. That growth and the loss both pass a quarter above their
settled level at the same sample, epoch 24.00, so at this resolution neither leads the
other. The
curvature along the step Adam actually takes stays between **0.4 and 1.0** throughout,
three orders of magnitude below λ_max: the sharpest direction is not the one Adam walks in.

**The Adam-scaled curvature behaves completely differently.** It rises through training,
reaches its highest value of the whole run — **7402** — in the last epoch before the
break-up, and collapses to **5479** two epochs later.

**Two step sizes fix the quantity.** Each run is trained to its own first break-up:

| Run | Break-up | Loss it left | λ_max before | Adam-scaled before | rate × λ_max | rate × Adam-scaled |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| rate 1e-3 | epoch 25 | 0.1247 | 2836.7 | 7401.8 | 2.837 | **7.402** |
| rate 5e-4 | epoch 56 | 0.0599 | 3587.4 | 13246.1 | 1.794 | **6.623** |

Halving the step buys **2.08x lower loss** before the break-up, and the run gets there over
**1.26x** the raw curvature and **1.79x** the Adam-scaled curvature. The last column is the
point: a boundary of the form *step size × curvature* would hold that product fixed, and on
the Adam-scaled curvature the two runs agree to **11%** while breaking up 31 epochs and a
factor of two in loss apart. On the raw curvature they are 37% apart. The quantity the step
size meets is the preconditioned one, which is what Adam's own update is scaled by.

**Four break-ups at one step size say the same, without changing anything.** Trained four
times longer, the detector breaks up at epochs 25, 54, 86 and 117, each time from a lower
loss:

| Break-up | Loss it left | λ_max before | Adam-scaled before | rate × Adam-scaled |
| ---: | ---: | ---: | ---: | ---: |
| epoch 25 | 0.1247 | 2772.2 | 6688.7 | **6.689** |
| epoch 54 | 0.0574 | 2729.8 | 8194.7 | **8.195** |
| epoch 86 | 0.0359 | 1885.6 | 8004.9 | **8.005** |
| epoch 117 | 0.0218 | 1418.1 | 8598.1 | **8.598** |

The losses these leave from span **5.7x** and the raw curvature **1.95x** — and it falls
across the run, from 2772 to 1418, the opposite of sharpening — while the product spans
**1.29x**, from 6.69 to 8.60 around a median of 8.10. Of the three quantities, it is the one
the events hold fixed, over four of them and with no setting changed between. Traced over
the whole run it sawtooths: climbing to between 7 and 8.6, collapsing at each break-up,
climbing again.

**A run that acts on the warning does not break up.** The last experiment trains the same
detector again and halves its own learning rate the first time that product rises past
**1.4x its median over epochs 5–10** — a rule fixed beforehand that reads only early
training and knows nothing about where the break-up falls. It fires at **epoch 21**, four
epochs before the untouched run breaks up, and over 35 epochs **no break-up happens at
all**, ending at **0.0626** against the untouched run's 0.0857 after 30. The
Adam-scaled curvature keeps climbing afterwards, to 9870 by epoch 27, but at half the rate
the product stays near 4.9 and never reaches the level the events leave from.

So the reading that survives is not that curvature climbs until a fixed step no longer
fits. It is that **each step size has a curvature it can tolerate, training pushes the model
up to that level, and it breaks up there** — at half the rate, twice the tolerance and half
the loss; four times over within one run; and not at all if the rate comes down when the
product gets there.

Three things this does not establish. The Hessian here is of the smooth piece of the loss:
the ignore mask in the YOLO loss is a threshold, so it holds one setting while the Hessian
is taken and can jump between epochs. A *step size × curvature* boundary is the analysis of
gradient descent on a quadratic, and Adam on this loss is neither, so that product is a
diagnostic proxy and not Adam's stability condition — it is reported as a number to compare
across these runs, not as a threshold of 2. And the intervention lowers a learning rate,
which raises the curvature a run can take whenever it is applied; it shows that **acting on
the warning is enough**, not that the warning names the cause. Separating those would take a
rate cut matched in size and duration but applied away from the warning, which this script
does not run.

Cost: about 45 minutes and 245 curvature estimates — 30 epochs at the full rate plus 30
inside the break-up, 60 at half the rate, 120 for the repeats and 35 for the intervention.
`--long-epochs 0`, `--intervene 0` and `--half-rate-epochs 0` each drop one arm.

### The same encoding on COCO

36 334 non-crowd boxes from `instances_val2017`, letterboxed to 416x416:

| | Mean best IoU | Boxes whose slot is taken |
| :--- | ---: | ---: |
| 3 anchors, one 13x13 grid | 0.461 | **4617 (12.7%)** |
| 9 anchors, grids of 52, 26 and 13 | 0.614 | **860 (2.4%)** |

A stride-32 cell covers 32x32 input pixels, and everything inside it competes for the
same few slots. A stride-8 level gives small anchors sixteen times as many cells.

### Part confidence maps and part affinity fields

The maps are drawn from keypoints: a Gaussian per keypoint per channel, and, for each
of the 19 limbs, a two-channel band carrying the unit vector from one end to the
other. A candidate connection is scored by the mean of the field projected onto it —
the line integral — and pairs are matched greedily per limb.

Because the maps come from the labels rather than from a network, these runs isolate
the association step: they show what the field adds when the maps are right, not how
a full OpenPose model scores. Each limb type is matched on its own, and grouping the
limbs into one skeleton per person is not done here. Recall is taken over every limb
whose two keypoints are labelled visible, so a missed peak counts against it as well
as a missed pairing.

| Scenes | Rule | Precision | Recall |
| :--- | :--- | ---: | ---: |
| 200 rendered, two people overlapping | Affinity field | **98.8%** | 78.0% |
| | Shortest distance | 83.5% | 83.4% |
| COCO, 150 images, people apart | Affinity field | **100.0%** | 89.2% |
| | Shortest distance | 94.3% | 94.9% |
| COCO, 112 images, people overlapping | Affinity field | **97.4%** | 83.2% |
| | Shortest distance | **77.8%** | 75.7% |

Where people stand apart the nearest candidate is usually the right one, and distance
is nearly as good. **The field earns its cost exactly where the two rules disagree**:
it rejects a pairing whose band carries no direction, and distance never rejects
anything, which is why its recall is the higher of the two and its precision the
lower.

### What the images show

| Image | Panels |
| :--- | :--- |
| `anchors.png` | training box shapes coloured by their best anchor, for k = 1, 3 and 5, anchors as stars |
| `grid_encoding.png` | one training image on its grid: each box, its centre, the responsible cell, the chosen anchor and the slot |
| `loss_terms.png` | every loss term and the total over all 30 epochs |
| `nms_before_after.png` | three test images with every box above the drawing threshold, and what per-class suppression keeps |
| `detections.png` | four unseen images, true boxes in green and detections in red with class and score |
| `pr_curves.png` | the precision-recall curve behind each class's AP |
| `coco_boxes.png` | one COCO image with its non-crowd boxes |
| `coco_slots.png` | COCO box shapes with the 3 and 9 anchors, and the share of boxes that lose their slot at one scale and at three |
| `pose_maps_synthetic.png` | a two-person scene, its strongest confidence map and its affinity field magnitude |
| `paf_vectors.png` | the peaks found in the confidence maps, and the affinity fields as arrows along each limb |
| `pose_assembly.png` | the true skeletons, the limbs paired by the field and the limbs paired by distance, with how many are correct |
| `pose_maps_coco.png` | a COCO image with overlapping people, both maps, and the limbs paired by the field |

From `05b_loss_spikes_and_step_size.py`, into the same folder:

| Image | Panels |
| :--- | :--- |
| `spike_anatomy.png` | every one of the 3750 optimiser steps: total loss, gradient norm, parameter update norm, the cosine against the previous step with its running median, and Adam's two moments, with the rise shaded |
| `spike_batch_content.png` | batch loss against the boxes the batch holds, quiet epochs against rising ones, and each epoch's heaviest batch as a share of its median |
| `spike_terms_and_layers.png` | each loss term over training, and how much further every parameter tensor moves per step during the rise, the head apart from the body |
| `spike_controls.png` | the two batch orders and the halved rate together, and the 120-epoch run with every break-up marked |

From `05c_curvature_and_stability.py`, into the same folder:

| Image | Panels |
| :--- | :--- |
| `curvature_over_training.png` | loss, sharpest curvature, curvature along the step taken and curvature after Adam's scaling, epoch by epoch, with the quiet stretch and the break-up shaded |
| `curvature_inside_the_break_up.png` | the same three curvatures at 25-step resolution through epochs 22–27, where the loss leaves and returns |
| `curvature_lr_control.png` | loss, raw curvature, Adam-scaled curvature and the step-size × curvature proxy for both step sizes, with each run's break-up marked |
| `curvature_repeated_breakups.png` | the 120-epoch run's loss and proxy with all four break-ups marked, and the four lined up on the epoch each left from |

---

## 6. Segmentation and skip connections

`06_unet_segmentation_and_skip_connections.py`

Three architectures with the same job — a class for every pixel — separated by what
they do about resolution.

### The baseline every model has to beat

Background covers **89.8%** of the rendered dataset, so predicting it everywhere
scores:

| Metric | Constant prediction |
| :--- | ---: |
| Pixel accuracy | **89.78%** |
| Mean IoU | **0.180** |
| IoU, each of the four object classes | 0.000 |

Pixel accuracy starts high whatever the model does. Mean IoU gives every class the
same weight and scores a class that is never predicted as zero, which is why both
are reported below.

### Three networks

| Model | Parameters | Approx. receptive field (convolutions only) | Time | Pixel accuracy | Mean IoU |
| :--- | ---: | ---: | ---: | ---: | ---: |
| UNet with skips | 1 085 837 | 89 px | 22 s | **99.71%** | **0.964** |
| UNet without skips | 1 525 235 | 89 px | 27 s | 98.02% | 0.769 |
| Flat, no resampling | 314 261 | **13 px** | 90 s | 92.01% | 0.371 |
| Background everywhere | 0 | – | – | 89.78% | 0.180 |

The parameter counts are not matched. The UNet without skips is widened from 24 to 30
base channels and has 40% more parameters than the one with skips; the flat network is
kept narrow because every layer runs at full resolution. The receptive fields count the
3x3 convolutions and the halvings only — pooling windows and transposed convolutions
widen them a little — so they illustrate the gap rather than give exact figures.

The flat network has the same number of convolutions and no pooling, so its output
unit sees roughly 13 input pixels instead of 89 — and it costs four times the training
time for a quarter of the parameters, because parameters and computation are different
things and every one of its layers runs at full resolution.

### Where the skips show

| Model | Accuracy within 3 px of a boundary | Mean IoU there | Accuracy in the interior |
| :--- | ---: | ---: | ---: |
| UNet with skips | **98.62%** | **0.957** | 99.97% |
| UNet without skips | 91.02% | 0.743 | 99.70% |
| Flat, no resampling | 73.73% | 0.390 | 96.40% |

Interiors are decided by colour and every model gets them. **The gap is at the
boundaries**, which is the detail the decoder loses on the way down and the skip
connections hand back. The two UNets differ in width as well as in skips, so this is a
comparison of two configurations rather than a single-variable ablation; the width
difference favours the model without skips, and it is still the worse of the two.
Both train with the same flips, one coin per batch.

### Pascal VOC 2012

1464 training and 1449 validation images at 128x128, 21 classes, trained from
scratch. Label 255 marks the band drawn around every object and covers **5.47%** of
the pixels; it is excluded from the loss and from every count, because it is not a
class the model is asked to predict.

| Model | Pixel accuracy | Mean IoU | Classes ever predicted |
| :--- | ---: | ---: | ---: |
| UNet with skips | 71.54% | **0.059** | 8 |
| UNet without skips | 73.12% | 0.045 | 3 |
| Background everywhere | **73.33%** | 0.035 | 1 |

Background is 73.5% of the labelled pixels and the largest object class, person, is
4.8%. **Pixel accuracy falls below the baseline while mean IoU rises**: predicting an
object class costs background pixels, and only one of the two numbers pays for it.
From-scratch numbers on this little data are far below what a pretrained encoder
reaches; what is comparable here is the architectures against each other.

### What the images show

| Image | Panels |
| :--- | :--- |
| `shapes_dataset.png` | eight training images above their label maps |
| `receptive_field.png` | one output pixel on a test image with the approximate field of the UNet and of the flat network |
| `training_curves.png` | training loss and test mean IoU after every epoch for the three models |
| `metrics_compare.png` | pixel accuracy against mean IoU for the constant prediction and the three models, and IoU per class |
| `shapes_predictions.png` | six test images, their truth, and each model's prediction |
| `boundary_errors.png` | four test images with the boundary band and, for each model, the wrong pixels inside and outside it |
| `voc_predictions.png` | six VOC validation images, their truth with the ignore label in grey, and both UNets' predictions |
| `voc_training_curves.png` | training loss and validation mean IoU per epoch on VOC |

---

## 7. Attention and self-supervised representations

`07_attention_and_self_supervised_representations.py`

### Attention, written out

Scaled dot-product attention over 4 tokens of 96 values in 4 heads of 24: every row
of the attention weights sums to **1.000000**, and the result matches
`nn.MultiheadAttention` loaded with identity projections to **3.73e-07**.

Around it, an encoder block of 74 784 parameters: attention and a feed-forward
network, each added back onto its input and layer-normalised. Zeroing both
sub-layers leaves the block returning its normalised input exactly (largest gap
**0.00e+00**). The residual connections keep a direct path for the input through each
sub-layer; because the normalisation comes after each addition, that path is
LayerNorm(LayerNorm(x)) rather than x, so this post-norm block does not start as the
identity function.

### Two ways to make tokens

| Tokeniser | Tokens | Parameters | Rendered, end-to-end | Rendered, probe | CIFAR-10, end-to-end | CIFAR-10, probe |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Patches, one 4x4 square each | 64 | 236 073 | 84.30% | 84.75% | **57.30%** | 57.70% |
| Convolutional stem | 64 | 284 361 | **93.15%** | **95.90%** | 56.70% | **62.35%** |

Same token count, same three encoder blocks, same eight epochs. Cutting the image
into squares gives each token one patch and nothing of its neighbours; a
convolutional stem overlaps them, so a token already carries local structure before
attention starts relating tokens to one another. **That is worth nine points on the
rendered shapes; on CIFAR-10 this single run shows no advantage end to end** (0.6 points
the other way, from one seed and no repeats) — eight epochs of a three-block encoder on
12 000 photographs is short of what either tokeniser needs.

The last two columns of each pair are different measurements and must not be read as one
correcting the other. End-to-end accuracy is what the classifier reaches on its own task
with its own head. The probe throws that head away, freezes everything behind it and
trains one fresh Linear(96 → classes) on the pooled tokens, which asks how linearly
separable the representation already is. **On CIFAR-10 the two answers disagree in
direction**: end to end the patch tokeniser is 0.6 points ahead, while under the probe the
convolutional stem is 4.65 points ahead. One run each, so this is a reason to keep the two
numbers apart, not a finding about tokenisers.

### Two evaluations, six encoders

Six encoder configurations are trained on the same 12 000 images, three of them without
ever seeing a label, and each is then put through the same two evaluations. They answer
different questions and are never added together:

**Frozen linear probe.** Freeze the encoder, train only a fresh `Linear(width → classes)`
on its features, standardised by the training set's own mean and standard deviation. It
asks: *how linearly separable is the representation this encoder has already learned?* It
scores the representation.

**Fine-tuning.** Unfreeze the encoder, put a fresh `Linear(width → classes)` on it — never
the head it may already carry — and train the two together. Every encoder gets the same
labelled data, the same 8 epochs, batch 128, Adam at 1e-3 and cross-entropy; only the width
of its representation differs. It asks: *what does this model reach once it is allowed to
adapt to the labelled task?* It scores the adapted model.

Neither is the end-to-end accuracy of the section above, which belongs to a classifier
trained with its own head from the start. Three numbers, three questions.

| Representation | Labels in pretraining | Width | Pretraining | Probe | Fine-tuned | Gain |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| Patch-token transformer | all | 96 | 7 s | 84.75% | 88.80% | +4.05 |
| Convolutional-token transformer | all | 96 | 9 s | 95.90% | 98.40% | +2.50 |
| Supervised ConvNet | all | 256 | 17 s | **99.85%** | **100.00%** | +0.15 |
| Autoencoder | none | 128 | 6 s | 31.45% | 68.95% | **+37.50** |
| Masked autoencoder, half the patches hidden | none | 96 | 35 s | 44.95% | 89.70% | **+44.75** |
| Contrastive, with projection head | none | 256 | 34 s | 87.65% | **100.00%** | +12.35 |
| Guessing | – | – | – | 12.50% | 12.50% | – |

The gain column is in percentage points, and it runs opposite to the probe: the encoders
whose features are least linearly separable gain the most once they are allowed to keep
learning. A low probe score says the representation is not linearly separable, not that
the encoder is a poor starting point. **On these rendered shapes the fine-tuning column
saturates** — two configurations reach 100.00% and a third 98.40% — so it separates the six
far less than the probe does, and the ordering it appears to give should not be read as
one. The CIFAR-10 run below is where that column has room to move.

What the two evaluations unify is the data, the split, the objective and the classifier
they end in, not the encoders: the autoencoder is convolutional with 128 features over 12
epochs, the masked autoencoder a transformer with 96 features over 30, and the contrastive
encoder a ConvNet body with 256 features over 20, while three of the six had already seen
the class labels before either evaluation began. The ordering is therefore of these six
configurations under a common downstream protocol, not a causal ranking of training
methods. It is consistent with
reconstruction rewarding whatever fills the most pixels, so that an autoencoder spends
its capacity on the background it has to redraw, while the masked autoencoder is asked
for something it cannot copy — the patches its encoder never saw — and the contrastive
encoder for something a background cannot answer: which two of 512 views came from the
same image. The run does not test that explanation directly.

**The projection head is discarded after training, and in this setup it substantially
improves the features underneath it.** This is the best-controlled comparison here: the
two contrastive runs share the body architecture and every training setting, and the
head is their one design difference. Each still starts from its own random weights,
augmentations and batch order, so it is not a strict single-variable ablation. The loss pulls the
head's output onto a sphere and throws away whatever it does not need there; the body
is one layer removed from that pressure, and probing it is worth 21 points — from one
seed, one head design, one temperature and one training budget.

### The same two evaluations on CIFAR-10

12 000 training and 2000 held-out images, everything else unchanged. Both are drawn from
the first two official training batches and the official test batch is not read, so
these are comparisons between the methods, not CIFAR-10 test-set accuracies:

| Representation | Probe | Fine-tuned | Gain |
| :--- | ---: | ---: | ---: |
| Patch-token transformer | 57.70% | 56.00% | −1.70 |
| Convolutional-token transformer | 62.35% | 61.00% | −1.35 |
| Supervised ConvNet | **72.45%** | **63.00%** | **−9.45** |
| Autoencoder | 33.25% | 54.35% | +21.10 |
| Masked autoencoder | 38.35% | 60.95% | +22.60 |
| Contrastive, with projection head | 55.05% | 62.25% | +7.20 |
| Contrastive, no projection head | 49.60% | – | – |
| Guessing | 10.00% | 10.00% | – |

Photographs are harder than rendered shapes and the gap to the supervised reference
widens (a reference for this run, not an upper bound any label-free method must stay
under), but **the ordering of the three label-free training families does not change**
under the probe, and neither does the cost of dropping the projection head.

**The fine-tuning column does something the rendered shapes hid.** There it saturated;
here it compresses the six into 54.35% to 63.00%, a spread of under nine points against
the probe's 39. And for the three encoders that had already been trained with labels the
gain is *negative* — the supervised ConvNet loses **9.45 points**, ending below its own
frozen representation. Eight more epochs on 12 000 photographs with a freshly initialised
head is enough to move an already-fitted encoder somewhere worse on held-out images, while
the three label-free encoders, starting from features that no label had yet shaped, all
gain. This is one run per cell with no repeats, and nothing here was tuned against the
held-out split, so read it as a demonstration that the two evaluations can disagree in
sign — which is the reason to report both — rather than as a measurement of how much
fine-tuning costs.

Added cost of the two evaluations: the six fine-tuning runs take 59 s on the rendered
shapes (7, 9, 17, 3, 6 and 17 s) and 67 s on CIFAR-10, against the 146 s of encoder
training the script already did; the two probes added to the tokeniser section cost a few
seconds more.

### What the images show

Every image below is written twice, once per run:
`outputs/attention_and_representations/synthetic/` holds the run on the rendered objects and
`outputs/attention_and_representations/cifar10/` the run on CIFAR-10. The file names are the
same in both, which is why they are kept apart.

| Image | Panels |
| :--- | :--- |
| `attention_weights.png` | the 4x4 attention weights of every head for the first sequence, each row summing to 1 |
| `encoder_block.png` | input and block output value distributions, and the zeroed block against the identity line |
| `dataset.png` | the first 16 training images with their classes |
| `tokenisers.png` | the input window of two neighbouring tokens: 4x4 and disjoint for patches, 9x9 and overlapping for the convolutional stem |
| `autoencoder.png` | eight test images above their reconstructions |
| `masked_autoencoder.png` | eight test images, the half of the patches the encoder is shown, and the hidden half as the decoder predicts it |
| `augmentations.png` | eight test images and two random views of each |
| `training_losses.png` | every training loss per epoch, grouped into supervised, reconstruction and contrastive |
| `probe_accuracy.png` | linear probe accuracy of all seven configurations against guessing; grey for the encoders that saw labels in pretraining, hatched for the projection-head ablation |
| `representation_evaluation.png` | the six compared encoders under both evaluations side by side, each labelled with its feature width |
| `fine_tuning_losses.png` | the downstream cross-entropy of all six encoders over the shared fine-tuning epochs |

---

## 8. Field-level extraction audit

`08_vlm_field_extraction_audit.py`

A model that returns valid JSON with every requested key has demonstrated nothing.
This script renders claim forms whose every value it chose itself, asks for those
values back, and scores field by field.

### The form

Six fields, five of which carry a deliberate trap:

```python
TRAPS = {
    "policy_number": "characters that share a shape: the letter I against the digit 1",
    "vehicle_model": "a small badge with the real code, next to a bigger trim/engine label with its own digits",
    "severity": "three boxes, one of them ticked",
    "driver_name": "a field blacked out on the page",
    "road_surface": "a field left blank, with a filled neighbour to borrow from",
}
```

Each trap is something a person reads correctly and an extractor gets wrong in a way
that still looks like an answer:

- **`policy_number`** is `IF-4821-77` — a capital I where a digit 1 would sit.
- **`vehicle_model`** is `A6`, printed on a small chrome badge. The line beside it
  reads `Avant quattro 45 TFSI` in a larger, bolder face and carries its own digits.
  **An extractor that grabs the biggest number on the line gets the trim, not the
  model.**
- **`severity`** is not written anywhere. Three boxes are drawn and one is filled.
  The answer is carried by which box is dark.
- **`driver_name`** is covered by a black bar. **Redacted is not the same as empty**,
  and an extractor that treats them alike loses the distinction the redaction was
  made to preserve.
- **`road_surface`** is left blank with a filled `Weather: Rain` directly beneath it.

The prompt asks for `REDACTED` and `BLANK` as explicit values, so both states are
answerable rather than being forced into `null`:

```
for a value that is blacked out return the string REDACTED; for a field left
empty return the string BLANK; for the amount return digits and a decimal
point only.
```

### Two axes

The same form is rendered in English, French and German. **Only the labels change**;
every value stays identical, so a difference in the score is a difference in reading
the layout rather than in reading the data.

`severity` is the exception, and deliberately so: the expected answer is the option
printed beside the ticked box, which differs by language. Asking for a translated
word instead would score the model on its vocabulary rather than on which box it saw.

Each render is then degraded into what the same page looks like photographed on a
desk:

```python
PHOTO_ROTATION = 1.4
PHOTO_BLUR = 0.8
PHOTO_QUALITY = 55
```

plus a lighting gradient across the page. Nothing about the content changes — every
value is still there, and a person still reads all six.

### What the runs show

| Condition | Score |
| :--- | :--- |
| Clean render, three languages | **15/18 fields — 83%** |
| Photograph of the same three pages | **15/18 fields — 83%** |

The two conditions score the same and fail on **different fields**, three times each,
in all three languages:

```
english   render  vehicle_model    wanted 'A6', got 'AVANT QUATTRO 45 TFSI'
french    render  vehicle_model    wanted 'A6', got 'AVANT QUATTRO 45 TFSI'
german    render  vehicle_model    wanted 'A6', got 'AVANT QUATTRO 45 TFSI'

english   photo   road_surface     wanted 'BLANK', got 'RAIN'
french    photo   road_surface     wanted 'BLANK', got 'RAIN'
german    photo   road_surface     wanted 'BLANK', got 'RAIN'
```

**On the clean page the model reads the wrong text.** The model code sits on a small
badge; the trim line beside it is larger and carries its own digits, and the answer
that comes back is the trim. Nothing was degraded — the page is pristine, and the
extractor still took the more prominent string.

**On the photograph the model fills in an empty field from its neighbour.** Blur and
compression were enough to make `Weather: Rain` migrate one row up into the blank
`road_surface`. Under the clean render it answered `BLANK` correctly every time.

Everything else held in both conditions: the shape-ambiguous policy number, the ticked
box, the redaction and the amount.

> ⇒ **An overall score of 83% twice over would suggest the two conditions are equally
> hard. They are not — they are failing at different things**, which is only visible
> because the score is broken out by field.

### The taxonomy

`classify()` sorts each miss into a named kind rather than reporting a bare count:

| Kind | What it looks like |
| :--- | :--- |
| Field misalignment | A neighbouring value slides into an empty slot |
| Highlight state misread | A row of options is read as a set of characters, not as one that is lit |
| Character-level misread | A short code comes back one character different |
| Over- or under-inference | A value is invented, or a fallback is used where the page is legible |

**The point of naming the kind is that each one has a different fix.** A character
misread means the field belongs to OCR. A highlight misread means the field needs a
cropped region. A fallback used too freely means the prompt needs a constraint in
the other direction — a rule that says when *not* to use it.

---

## 9. Grounding and failure modes

`09_vlm_grounding_and_failure_modes.py`

Three known weaknesses, each put on a scale instead of described.

### Asking a model to point

The scene is a vehicle drawn side-on. The front wheel's box is **returned by the
drawing function**, not looked for afterwards, so the overlap computed later is
against a coordinate the script chose:

```python
front = (168, 268, 240, 340)
```

The model is asked for `{"box": [x1, y1, x2, y2]}` in pixels. What comes back:

```
{"box": [261, 261, 812, 376]}
```

Read as pixels, `x2 = 812` is off a 640-wide image. So the four numbers are scored
under every convention in use — pixels, a 0–1 fraction, a 0–1000 grid, and both axis
orders:

| Reading | Box | IoU |
| :--- | :--- | ---: |
| **0–1000 grid as y1 x1 y2 x2** | (167, 110, 241, 341) | **0.304** |
| pixels as x1 y1 x2 y2 | (261, 261, 812, 376) | 0.000 |
| pixels as y1 x1 y2 x2 | (261, 261, 376, 812) | 0.000 |
| 0–1000 grid as x1 y1 x2 y2 | (167, 110, 520, 158) | 0.000 |

**The convention was never stated, and reading it wrong turns a partial hit into a
flat zero.** Any pipeline that assumes one convention will report a working model as
broken, or the reverse.

Splitting the overlap by axis says more than the single number does. The best
reading holds 100% of the wheel's width and 100% of its height — the wheel is
entirely inside the box — but the box is **3.3 times the wheel's area**, 231 pixels
tall against the wheel's 72. It reaches up over the body. All of the missing overlap
is surplus, not misplacement.

### Dense small text

A departures board is drawn with 26 rows at 12 point, and the script keeps what every
row says. The question needs several rows at once: *list every distinct airline with
a flight in zone A.*

In the runs recorded here the answer was complete — 9 of 9 airlines, nothing missed
and nothing invented. **The check still ships**, because the failure it looks for is
not a short answer but a long one: a collapsed reply repeats an entry over and over
and stays fluent throughout.

```python
def repetition(items, text):
    """Return how many times the most repeated whole entry appears, and which one.

    Counting words would flag a correct answer here, because half these airlines
    have the word Air in the name. A collapsed reply repeats whole entries, so
    whole entries are what gets counted.
    """
```

That docstring records a real false positive from building this: a word-level counter
called a correct 9-item answer degenerate, because `Air` appeared four times across
four different airline names. **The unit the check counts has to be the unit the
failure repeats.**

### An image is an attachment, not a memory

Four follow-up questions, each asking for something the first reply never mentioned,
against two histories that differ in exactly one thing — whether the message the
image was attached to is still present, or has been replaced by its text:

| Probe | Drawn | Image kept | Image dropped |
| :--- | :--- | :--- | :--- |
| gate of row 1 | B5 | `'B5'` ok | `'A12'` no |
| time of row 1 | 16:20 | `'16:20'` ok | `'08:15'` no |
| airline of row 4 | Kestrel Airways | `'Kestrel Airw'` ok | `'Norwegian'` no |
| status of row 2 | Delayed | `'Delayed'` ok | `'DELAYED'` ok |

**4/4 with the image in the history, 1/4 without** — and the single hit is a status
field with three possible values.

The image does not have to ride on the newest message, but it does have to still be
on one of them. **A transcript stored as plain strings loses it silently**, and the
next answer comes back in the same confident shape as the ones that could see:
`A12`, `08:15`, `Norwegian` are all plausible and all invented.

---

## 10. Video by keyframe sampling

`10_video_keyframe_understanding.py`

An image model can stand in for a video model by sampling frames and stitching the
answers together. This script builds that stand-in and measures exactly what the
sampling costs.

### A clip whose events are scheduled, not observed

```python
WIDTH, HEIGHT = 480, 270
FPS = 30
DURATION_SECONDS = 4

IMPACT_FRAME = 78      # a scrape appears and stays
FLASH_FRAME = 45       # a brake lamp, on for three frames
FLASH_LENGTH = 3
```

The clip is rendered frame by frame, encoded to mp4, and then **read back out of the
file**. Nothing downstream touches the renderer, so the rest of the script works from
a video the way it would from any other.

Two events, deliberately different in duration: a scrape that appears at 2.60 s and
persists, and a brake lamp that is on for 100 ms.

### The sampling and its blind spot

At a stride of 10 frames, 12 of 120 frames are sampled — one every 0.33 s:

```
sampled frames: [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110]
frames landing inside the brake lamp window: none
```

**An event shorter than the stride is not hard to see, it is not sampled.** No
prompt and no model improves this; the frame carrying it was never sent.

### The timeline

Each frame is asked one question in isolation — `DAMAGED` or `CLEAN` — and the
answers are assembled afterwards:

| | Result |
| :--- | :--- |
| Per-frame agreement with the frame as drawn | **12/12** |
| First `DAMAGED` frame | 80, at 2.67 s |
| The scrape as scheduled | frame 78, at 2.60 s |
| Error | **late by 0.07 s** |

The estimate **can only ever be late**, because the change is invisible until the
next sample lands.

### What a coarser stride buys and costs

The same twelve answers, re-read at wider strides — no extra calls:

| Stride | Calls | Window | Estimate | Error |
| ---: | ---: | ---: | ---: | ---: |
| 10 | 12 | 0.33 s | 2.67 s | 0.07 s |
| 20 | 6 | 0.67 s | 2.67 s | 0.07 s |
| 30 | 4 | 1.00 s | 3.00 s | 0.40 s |
| 40 | 3 | 1.33 s | 2.67 s | 0.07 s |

**The window is what a stride guarantees; the error in any one run is wherever the
samples happened to fall inside it.** Stride 40 lands closer here than stride 30 does,
on fewer calls — which is luck, not a reason to sample less.

### What this is not

This reads each frame on its own and stitches the answers together afterwards. A
model built for video sees the frames together, which is how motion, order and
duration become answerable at all.

That route was evaluated and deliberately not taken here: the open checkpoints in
this class carry roughly 17 GB of weights, want 24 GB of memory to run, and pin a
video decoding library that does not build on this platform. The stand-in is honest
about being one — **it can locate an event to within a sampling window and can say
nothing about order or duration.**

---

## 11. Document layout audit

`11_document_layout_audit.py`

Parsing a PDF into blocks is easy. Knowing whether the *structure* survived is the
part that needs checking, and the check needs a document whose structure was known
before it was drawn.

### A document declared in code

Six sections, each rendered with a named flaw:

```python
#   "clean"   - full heading size and weight
#   "demoted" - drawn a shade above body size, under the classifier threshold
#   "fused"   - drawn on the same line as the paragraph that follows it
#   "art"     - drawn character by character on a staggered baseline
```

Laid out in two columns, body at 9.5 pt and headings at 14 pt, with the classifier
threshold at 12 pt. **Every failure mode below is a way of getting past that one
number.**

The staggered baseline deserves a note, because it is the one that surprises:

```python
BASELINE_STAGGER = 6
```

Six points is where the parser stops seeing one line and starts seeing one line per
character. A display title set with characters riding above and below the baseline is
perfectly legible on the page and comes back as individual glyph lines.

### Reading it back

58 text lines are recovered, each carrying its largest span size, position and font.
Sorting those lines top to bottom — the obvious thing to do — **jumps back to the
left column 3 times**: a two-column page has two reading orders and vertical position
picks the wrong one.

### The reconciliation

| Declared heading | Outcome |
| :--- | :--- |
| Scope of Cover | recovered |
| Excluded Events | recovered |
| Making a Claim | **demoted** — its own line, but under the size threshold |
| Settlement Basis | **fused** — glued to the paragraph that follows it |
| Renewal and Cancellation | recovered |
| Handling Disputes | **shattered** — 16 lines, one per character |

**Heading recall 3/6.**

And the number that a naive check would have produced instead:

> counting heading lines instead would have reported **19 headings for 6 sections**,
> and got both the number and the direction wrong

Nineteen is larger than six. A count alone does not just miss the problem, it
suggests the document is *richer* in structure than it is.

### What the misses cost downstream

Slicing by recovered heading gives **19 chunks for 6 sections**:

| Chunk | Words | |
| :--- | ---: | :--- |
| `'Scope of Cover'` | 56 | |
| `'Excluded Events'` | 147 | **also holds Making a Claim, Settlement Basis** |
| `'Renewal and Cancellation'` | 42 | |
| `'s'` | 50 | the last section's body, under a one-character heading |

One chunk silently absorbs two other sections. Another is headed by a single letter.
Fifteen more carry five words or fewer.

**The check that catches every heading failure above is one line: recovered heading
count against the count the document is known to have.** Reading the markdown and
finding it fluent catches none of it.

It does not catch everything, and the script says so: the reading order is read
correctly here and scrambled only by the sorted pass in step 3, which the heading
count cannot see. **That one needs its own check.**

---

## 12. Convolution from first principles

`12_conv_kernels_and_feature_maps.py`

No API, no training. What a convolution kernel computes, one window at a time,
verified against the framework.

### One kernel, checked by hand

A 5×5 binary image and a plus-shaped 3×3 kernel, convolved with explicit loops:

```
hand-computed output:
[[4. 3. 4.]
 [2. 4. 3.]
 [2. 3. 4.]]
max difference against nn.Conv2d: 0.00000000
```

The loop exists to be compared, not to be fast — every framework ships an optimised
routine for this, and the optimised one is also the one nobody can check.

The strongest cell is explained rather than stated: the kernel has 5 ones, and the
window under that cell put a 1 beneath each of them.

Two details make the comparison possible at all: `nn.Conv2d` takes four dimensions,
so a `(5, 5)` array gains two axes to become `(1, 1, 5, 5)`; and `bias=False`, because
a bias term would shift every output cell by an unknown constant.

### Output size is decided before any number is multiplied

| padding | stride | predicted | actual |
| ---: | ---: | ---: | ---: |
| 0 | 1 | 3 | 3 |
| 1 | 1 | 5 | 5 |
| 0 | 2 | 2 | 2 |
| 1 | 2 | 3 | 3 |

`(H + 2p − k) // s + 1`. Padding 1 holds the map at its input size; stride 2 halves it.

### Four kernels, four directions

The test image is rendered rather than photographed, so the answer to "did the
vertical kernel fire on a vertical edge" does not depend on what happens to be in a
photograph. Its edges are at coordinates the drawing function chose: a vertical
dark-to-light edge at column 40, a horizontal one at row 80.

```python
vertical = np.hstack([-np.ones((k, k // 2)), np.ones((k, k // 2))])
return np.stack([vertical, -vertical, vertical.T, -vertical.T])
```

```
   dark_to_light_x  sum=  0.0  first row=[-1.0, -1.0, 1.0, 1.0]
   light_to_dark_x  sum=  0.0  first row=[1.0, 1.0, -1.0, -1.0]
   dark_to_light_y  sum=  0.0  first row=[-1.0, -1.0, -1.0, -1.0]
   light_to_dark_y  sum=  0.0  first row=[1.0, 1.0, 1.0, 1.0]
```

**Every kernel sums to zero, so a flat region answers with zero whatever its
brightness.** Half negative and half positive: the response is large only where the
image crosses from the side the kernel subtracts to the side it adds. That is what
makes a kernel directional rather than merely edge-sensitive.

### Convolution, activation, pooling

| Stage | Shape | Range |
| :--- | :--- | :--- |
| conv | (1, 4, 157, 157) | [−6.75, 6.75] |
| relu | (1, 4, 157, 157) | [0.00, 6.75] |
| pool | (1, 4, 78, 78) | [0.00, 6.75] |

The activation zeroed **2.3%** of cells (2,226 of 98,596), every one of them an edge
running the wrong way for its kernel. Pooling kept **25%** of the cells.

**That count needs a tolerance, and the tolerance is the whole point.** A kernel whose
weights sum to zero answers a flat region with zero — but only to within floating
point. The transposed kernels accumulate in a different order and leave a residue
around 1e-7, and most of this image is flat, so counting every value below zero scores
that residue as signal:

```
negative cells, counted as < 0        25,789   (26.2%)
  of those, |value| < 1e-6            23,563   (91.4%)  <- rounding in flat regions
negative cells, counted as < -1e-6     2,226    (2.3%)  <- actual edges
```

**91% of the naive count is not an edge at all.** The script reports the 2.3% and
prints the residue separately, because the sentence after the number — *every one of
them an edge running the wrong way* — is only true of the smaller figure.

### Where the kernels actually fired

Reading one output row that crosses the bright block and nothing else:

```
   dark_to_light_x peaked at image column   40, drawn edge at   40, score 5.96
   light_to_dark_x peaked at image column   81, drawn edge at   81, score 5.96
```

Both peaks land on the coordinates the renderer used.

**But the largest response in the whole map is not on the block at all.** It sits at
(134, 88), on a diagonal stroke, at 6.75 against the block's 5.96. The stroke is
brighter than the block — 1.00 against 0.90 — and

> the kernel scores the size of the brightness step, not how well the edge lines up
> with its axis

An "edge detector" is a contrast-difference detector. Reporting the global maximum as
"where the vertical edge is" would have been wrong, and the script says so instead of
quietly picking the row that agrees.

---

## 13. Input resolution and network design

`13_cnn_input_resolution_mismatch.py`

A network designed for 224×224 inputs, handed a 32×32 one. **This script exists to
settle a common claim by measurement, and the measurement does not support it.**

### A task where a distance is the only signal

Four classes, distinguished only by how far apart two dots sit:

```python
SEPARATIONS = (3, 5, 7, 9)
DOT_PIXELS = 2
```

The pair jitters across the frame and is drawn horizontally or vertically, so neither
position nor axis carries the answer. Every class puts the same amount of light on the
frame:

```
mean brightness per class: 107.0, 107.0, 107.0, 107.0
  - identical, so brightness carries no answer
```

**No amount of blur leaves a shortcut behind.** Separation is all there is.

### Tracing the stem

| Stage | from 32 | from 224 |
| :--- | ---: | ---: |
| conv1 7×7 s2 | 16 | 112 |
| maxpool s2 | 8 | 56 |
| layer1 | 8 | 56 |
| layer2 | 4 | 28 |
| layer3 | 2 | 14 |
| layer4 | **1** | **7** |

The 32×32 input reaches **1×1** before the classifier, where the design expects 7×7.
The stem downsamples by 4× before any residual block runs, so one cell afterwards
covers 4 input pixels — and a 3-pixel gap is inside one cell.

That reads like a proof that the fine detail is gone. It is not.

### Three networks, same data

| | Parameters | Training time | Accuracy |
| :--- | ---: | ---: | ---: |
| ResNet-50 | 23,509,956 | 12 – 32 s | 96.7% – 99.8% |
| SizedForInput | 27,396 | 1 – 3 s | 100.0% |
| TooSmall | 152 | 1 – 2 s | **27.3%** |

(Ranges are quoted because this loop is not bit-for-bit deterministic on GPU and the
wall clock moves with the machine's power state. `SizedForInput` reached 100% in every
run and `TooSmall` 27.3% in every run.)

Chance is 25%. Every loop finishes and none of them raises anything.

**`TooSmall` is the arm that makes the comparison mean something.** It is the same kind
of network — one convolution, one pool, one linear head — with two channels instead of
sixteen and a single pool that throws away most of the resolution at once. Its loss
barely moves across twelve epochs (1.392 → 1.386) and it lands **2.3 points above
chance**. Without it, the run only shows that the oversized model is expensive; with it,
the run shows that the size was **chosen** rather than merely survived.

### The result, stated as the numbers support it

> the mismatch did NOT hide the fine detail. The stem samples at stride 2, but each
> of its kernels still spans 7 input pixels, so a 3-pixel gap survives in the channel
> values even once the map has gone coarse

What the mismatch actually cost is measurable elsewhere:

- **858× the parameters** and roughly **10× the training time**, for the same accuracy
- **a 1×1 final feature map**, so the pooling that follows averages a single cell —
  nothing downstream can ask *where*

"It cannot see the detail" was the intuition. The run says otherwise, and the run is
what gets reported. The usable conclusion is not that the architecture is blind, it is
that **the reason to size a network to its input is what you pay, and what shape comes
out the other end.**

---

## 14. Detection: auditing a split, and pricing a submission

`14_yolo_split_audit_and_submission.py`

Where a detection number comes from, and how little of it is the model.

### A dataset that records every instance as it draws it

160 images at 128×128, four defect classes on textured plates, 336 instances:

```python
CLASSES = ("scratch", "patch", "hole", "crack")
```

> The boxes are not detected here, they are recorded as they are painted. That is
> the whole point of synthesising the data: the ground truth cannot disagree with
> the image, so any disagreement later belongs to the model or to the split.

Labels are written as VOC XML and converted to the YOLO text layout, which is a real
conversion with a real trap:

```python
cx, cy = (x1 + x2) / 2 / width, (y1 + y2) / 2 / height
bw, bh = (x2 - x1) / width, (y2 - y1) / height
```

VOC stores absolute corners; YOLO stores a normalised centre and extent. Getting this
wrong produces boxes that are valid numbers in the wrong places, **which trains
without complaint**. A normalised value outside [0, 1] means the source box left the
image, and that row is dropped rather than learned.

### The split as a dataset usually arrives

```python
AS_ARRIVED = {"train": 128, "val": 2, "test": 30}
```

Almost everything in train, a validation set small enough to fit on one screen, and a
test set nobody looks at until the end.

The audit runs **before a single epoch**:

```
split     images   inst   scratch     patch      hole     crack
train        128    272        65        58        67        82
val            2      5         1         1         2         1
test          30     59        17        13        17        12
```

Five instances in validation. And the checkpoint the trainer keeps is the one that
scored best on exactly those two images.

### The metric, read twice

| Split | mAP@0.5 | Per class |
| :--- | ---: | :--- |
| val | **1.000** | scratch 1.000, patch 1.000, hole 1.000, crack 1.000 |
| test | **0.836** | scratch 0.784, patch 0.985, hole 0.762, crack 0.812 |

A perfect score, on five instances.

> the validation figure is an average over 5 instances and the test figure over 59,
> so 1.000 against 0.836 is a difference in sample size before it is anything else

**A perfect validation score on a split this size is not evidence of a perfect model;
it is evidence that the split is too small to disagree with anything.**

### The threshold the metric is defined at

Average precision is defined over the full ranked list of detections, so scoring has
to ask for boxes the detector is barely confident about:

```python
METRIC_CONFIDENCE = 0.001
VIEWING_CONFIDENCE = 0.25
```

Scoring the same weights at the default viewing threshold instead **keeps 58 of 751
detections and reports 0.797 against 0.836**.

The threshold that makes a picture readable is not the threshold the metric is defined
at, **and nothing warns when the two are swapped.**

### Two edits that move no box

The same 751 predictions, written three ways and scored by the same function:

| Submission | Rows | mAP@0.5 |
| :--- | ---: | ---: |
| as predicted | 751 | 0.8360 |
| grouped by image_id | 751 | 0.8360 |
| **confidence set to 1.0** | 751 | **0.1173** |

Sorting changed the file and not the score: average precision ranks the rows itself
before scoring them. **Flattening the confidence column did change it** — every row
now ties, and the ranking falls back to the order the rows were written in.

Neither edit touched a coordinate. **The metric is a property of the submitted list,
not only of the detector.** That cuts both ways: a submission can be improved by
fixing its ordering without retraining anything, and it can be destroyed by filling a
column with a constant that looked harmless.

---

## What the fourteen runs settle

1. **A colour rule is a rule about ratios or about levels, and only one of them
   survives the light.** The RGB box went from IoU 0.972 to 0.000 when the frame was
   dimmed; the HSV rule went to 0.984.
2. **Voting turns a set of edge pixels into parameters, and what you let it vote on
   is the cost.** The circles cost 5.87 million votes sampled blindly and 191 801
   along the gradient — 3% of the work, and all three disks instead of two.
3. **A descriptor is a choice about what to throw away.** Splitting each gradient
   vote between neighbouring bins and cells cut the distance a 5° turn produces by
   half; the integral image made a rectangle sum cost four reads at any size.
4. **Capacity and conditioning are different problems.** XOR needs a hidden layer at
   all, and then still fails from 46% of starting points when two hidden units start
   with small weights, and from none of them with eight units.
5. **The output tensor is not the answer; decoding it is.** Encode-decode round-trips
   to 3.81e-06 px, and the same grid drops 12.7% of COCO's boxes at one scale against
   2.4% at three.
6. **Two numbers can disagree about the same prediction.** On VOC, predicting object
   classes lowered pixel accuracy below the constant baseline while raising mean IoU.
7. **What an objective asks for is what the representation ends up holding.** Under
   the same linear probe: reconstruction 31%, masked patches 45%, contrastive 88%.
8. **A reply that parses is not a result.** Script 08 scored 83% on clean renders and
   83% on photographs of the same pages — the same JSON shape in both, the same
   headline number, and a different field failing in each.
9. **Coordinates come with an unstated convention.** Script 09 read the same four
   numbers six ways, scoring 0.304 under one and 0.000 under the rest; and an image
   is an attachment, not a memory — 4/4 with it in the history, 1/4 without.
10. **Sampling buys a bound, not an estimate.** An event shorter than the stride is
    not hard to see; it is not sampled.
11. **Count the structure you recovered against the structure you know is there.**
    Script 11 recovered 3 of 6 headings while reporting 19 heading lines.
12. **A kernel scores contrast, not alignment.** Script 12's strongest response was on
    a brighter diagonal, not on the axis-aligned edge it was built for.
13. **Measure the cost, not the intuition.** Script 13 set out to show a resolution
    mismatch hiding fine detail and found it did not — the cost is 858× the parameters
    and a 1×1 output, not accuracy.
14. **A metric is computed over a list.** Script 14 moved a submission from 0.836 to
    0.117 by rewriting one column and no coordinates.

The thread through all fourteen: **every one of these was found by comparing an output
against a value that was known in advance.** None would have surfaced from reading the
output and finding it plausible.

---

## Running them

```bash
pip install -r ../requirements.txt

# classical vision, no model and no GPU needed
python 01_color_tracking_and_optical_flow.py
python 02_edges_and_hough_voting.py
python 03_hog_and_haar_detectors.py
python 12_conv_kernels_and_feature_maps.py
python 11_document_layout_audit.py

# trained here, GPU optional
python 04_training_mechanics_xor_softmax_batchnorm.py
python 05_grid_detection_and_pose_assembly.py
python 06_unet_segmentation_and_skip_connections.py
python 07_attention_and_self_supervised_representations.py
python 13_cnn_input_resolution_mismatch.py
python 14_yolo_split_audit_and_submission.py

# a vision model key required
python 08_vlm_field_extraction_audit.py
python 09_vlm_grounding_and_failure_modes.py
python 10_video_keyframe_understanding.py
```

Scripts 08–10 read `GEMINI_API_KEY` or `OPENAI_API_KEY` from `.env` and default to a
small vision model, overridable with `VISION_MODEL`. They send batches of images, so
each one paces itself and retries on a rate limit rather than failing part way
through.

Five scripts also accept a public dataset and repeat their measurements on it. Nothing
is downloaded; each path points at a copy you already have:

```bash
python 03_hog_and_haar_detectors.py --tinyface-root <dir> --cifar10-root <dir>
python 04_training_mechanics_xor_softmax_batchnorm.py --mnist-root <dir>
python 05_grid_detection_and_pose_assembly.py --coco-root <dir>   # annotations/ and val2017/
python 06_unet_segmentation_and_skip_connections.py --voc-root <dir>   # holding VOC2012/
python 07_attention_and_self_supervised_representations.py --cifar10-root <dir>
```

- TinyFace: https://qmul-tinyface.github.io/
- CIFAR-10: https://www.cs.toronto.edu/~kriz/cifar.html
- MNIST: http://yann.lecun.com/exdb/mnist/
- COCO: https://cocodataset.org/#download (val2017 images and annotations only)
- Pascal VOC 2012: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/

Everything else each script needs is generated on the first run into `outputs/`: the
clip, the scenes, the claim forms, the PDF, the feature maps, the training samples and
the detection dataset. That directory is disposable — deleting it costs one rerun.

The four scripts that train a network on the GPU (04, 05, 06, 07) also set cuDNN to
its deterministic algorithms. Without that, cuDNN picks a convolution algorithm per
shape and some of them accumulate in a non-deterministic order: two runs of the same
seed gave mean IoU 0.936 and 0.956 for the same model. With it, two runs of script 06
differ on **0 lines** once the timings are excluded, so every number above is
reproducible on this machine rather than merely typical.
