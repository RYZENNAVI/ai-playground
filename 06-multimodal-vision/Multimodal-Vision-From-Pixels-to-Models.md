# Multimodal vision: from pixels to models

Scripts 01 to 07 build vision from the pixels up, with no pretrained weights: colour, edges,
hand-built descriptors, and then networks trained here for classification, detection,
segmentation and representation learning. Scripts 08 to 10 send rendered images to a hosted
vision model, and script 11 parses a rendered PDF; each scores what comes back against what was
drawn. Scripts 12 to 14 return to training: the arithmetic of a convolution, what an input
resolution costs, and where a detection metric comes from. This document explains what each
script does and the ideas it relies on.

| # | Script | What it shows |
| :---: | :--- | :--- |
| 01 | `01_color_tracking_and_optical_flow.py` | Colour thresholds in BGR and HSV, morphology, connected components, mean shift and CAMSHIFT, Harris corners, block matching and Lucas-Kanade optical flow |
| 02 | `02_edges_and_hough_voting.py` | Gaussian smoothing, Canny stage by stage, and Hough voting for lines, circles and a generalised shape |
| 03 | `03_hog_and_haar_detectors.py` | HOG descriptors, Haar features on an integral image, and AdaBoost on rendered and real faces |
| 04 | `04_training_mechanics_xor_softmax_batchnorm.py` | Why XOR needs a hidden layer, softmax cross-entropy, an MLP against a CNN, and training against evaluation mode |
| 05 | `05_grid_detection_and_pose_assembly.py` | YOLO-style grid detection with IoU anchors, NMS and mAP, and pose assembly with part affinity fields |
| 05b | `05b_loss_spikes_and_step_size.py` | Why every loss term of script 05 rises near epoch 25, tested against batches, Adam's moments and three controls |
| 05c | `05c_curvature_and_stability.py` | Hessian curvature through training, and the step size times the Adam-scaled curvature at each break-up |
| 06 | `06_unet_segmentation_and_skip_connections.py` | Semantic segmentation with a UNet, with and without skip connections, against a flat network |
| 07 | `07_attention_and_self_supervised_representations.py` | Attention by hand, two tokenisers, and six encoders under a linear probe and under fine-tuning |
| 08 | `08_vlm_field_extraction_audit.py` | Key information extraction from form images with a vision-language model, scored field by field |
| 09 | `09_vlm_grounding_and_failure_modes.py` | Visual grounding, dense text reading, and image context across turns |
| 10 | `10_video_keyframe_understanding.py` | Video understanding by keyframe sampling, and what the stride costs |
| 11 | `11_document_layout_audit.py` | Document layout analysis: headings recovered from a PDF, reconciled against the document declared in code |
| 12 | `12_conv_kernels_and_feature_maps.py` | A convolution by hand against `nn.Conv2d`, output sizes, and directional kernels |
| 13 | `13_cnn_input_resolution_mismatch.py` | ResNet-50, designed for 224-pixel inputs, on 32-pixel images, against two small networks |
| 14 | `14_yolo_split_audit_and_submission.py` | A YOLO detector: the split audited before training, and three submission files scored |

## Shared setup

*   Every script generates its own inputs and records the ground truth as it draws it, so
    every score is taken against a known answer. Everything generated goes into `outputs/`,
    which one rerun rebuilds.
*   Five scripts also repeat their measurements on a public dataset. Nothing is downloaded and
    none of the datasets is kept in the repository (together about 3.7 GB); each flag points at
    a copy you already have:

    | Script | Flag | Dataset |
    | :--- | :--- | :--- |
    | `03` | `--tinyface-root`, `--cifar10-root` | TinyFace (https://qmul-tinyface.github.io/), CIFAR-10 python batches (https://www.cs.toronto.edu/~kriz/cifar.html) |
    | `04` | `--mnist-root` | MNIST, the four IDX files uncompressed, in the folder or in `raw/` inside it |
    | `05` | `--coco-root` | COCO val2017 images and annotations (https://cocodataset.org/#download) |
    | `06` | `--voc-root` | Pascal VOC 2012, the folder holding `VOC2012/` (http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) |
    | `07` | `--cifar10-root` | CIFAR-10 |

    For MNIST, `torchvision.datasets.MNIST(root, download=True)` lays the files out the way
    script 04 reads them; they also sit under https://ossci-datasets.s3.amazonaws.com/mnist/.
*   Scripts 08 to 10 read `GEMINI_API_KEY` or `OPENAI_API_KEY` from `.env` and default to
    `gemini-3.1-flash-lite`, overridable with `VISION_MODEL`. They pace their calls and retry on
    a rate limit.
*   Scripts 04 to 07, 05b and 05c set cuDNN to its deterministic algorithms. Without that, two
    runs of the same seed gave mean IoU 0.936 and 0.956 for the same model in script 06; with
    it, two runs differ on 0 lines once the timings are excluded.

## Script 01: Colour tracking and optical flow

In the clip part 1 draws, a blue ellipse moves along a known path, grows, turns, and loses more
than half its light halfway through. Every stage of the classical tracking pipeline is scored
against the mask that drew it.

*   In part 2, a box in the BGR cube and a hue and saturation rule in HSV both reach IoU 0.972
    on the bright frames. On the dimmed frames the BGR box scores 0.000 and HSV 0.984. Dimming
    multiplies B, G and R by one factor, which leaves the ratios, and so hue and saturation,
    alone: the mean colour goes from HSV [106 216 165] to [106 217 73], and only V moves. A
    light that changes colour would move hue as well.
*   In part 3, the raw mask of the first frame holds 41 components, the ellipse and scattered
    specks. Erosion, dilation and a closing leave 1 component at IoU 1.000.
*   In part 4, a two-pass labeller and a flood fill, both written out, agree with OpenCV: 42
    components at 4-connectivity (from 55 provisional labels) and 41 at 8-connectivity (from
    47). The gap is two squares that touch only at a corner. The provisional labels are what the
    equivalence table resolves: a U shape is two arms until the scan reaches the bar.
*   Part 6 back-projects a 16-bin hue histogram onto every frame, and the result matches
    `cv2.calcBackProject` exactly. The saturation gate decides whether it is usable: in the last
    frame 0.489 of the probability mass lands on the object without it and 0.989 with it,
    because a grey pixel's hue is set by noise.
*   Parts 7 and 8 track the object as it grows from 1297 to 4037 pixels:

    | | First third | Last third |
    | :--- | ---: | ---: |
    | Mean shift, fixed window: centre error | 1.66 px | 5.60 px |
    | Mean shift: share of the object inside the window | 85.5% | 42.6% |
    | CAMSHIFT: centre error | 0.03 px | 0.02 px |
    | CAMSHIFT: share of the object inside the window | 100.0% | 100.0% |

*   CAMSHIFT reads size and orientation out of the second moments and follows the object. The
    hand-written mean shift picks the same window as `cv2.meanShift` in 90 of 90 frames. Both
    trackers start from a box taken off the true mask, so these are errors of keeping hold of
    an object, not of finding one.
*   Part 9 finds corners with the structure tensor, the matrix Harris scores and Lucas-Kanade
    inverts. Its eigenvalues
    are (0, 0) on a flat point, (1.43, 0) on an edge (R = -0.102) and (0.97, 0.34) at a corner
    (R = +0.242). All 16 true corners are found, with none on the disk.
*   Parts 10 and 11 recover motion from a texture shifted by a known amount:

    | Method | True shift | Error |
    | :--- | :--- | ---: |
    | Block matching, 16x16, search radius 8 | (3.6, -2.4) | mean 0.58 px |
    | Block matching | (11.3, 6.8) | mean 8.43 px |
    | Lucas-Kanade, one level | (3.6, -2.4) | median 0.010 px |
    | Lucas-Kanade, one level | (11.3, 6.8) | median 12.816 px |
    | Lucas-Kanade, 3-level pyramid | (11.3, 6.8) | median 0.008 px |
    | `cv2.calcOpticalFlowPyrLK` | (11.3, 6.8) | median 0.008 px |

*   Block matching answers whole pixels only, so its floor on the small shift is 0.57 px, and
    the large shift is outside its search square. The pyramid shrinks the 13.2 px shift to
    3.3 px at the top level, inside the range of the first-order expansion.
*   The aperture problem: at the corners of a shifted rectangle the smallest eigenvalue of G is
    45892.4 and the error 0.014 px; at the middles of the sides it is 0.0 and the error 2.150
    px. On the horizontal edges the recovered vector is (0.00, 1.70) for a true (2.6, 1.7): the
    component across the edge is exact and the one along it cannot be observed.

## Script 02: Edges and Hough voting

Part 1 draws four lines 5 px wide and three filled disks at known parameters under Gaussian
noise of σ 25, and the later parts recover them. An edge counts as found within 2 px of a true
boundary.

*   Part 2 shows that a Gaussian window too small for its σ keeps less than half the kernel's
    weight: 3x3 at σ 1.5 keeps 0.479. It still smooths, but it is no longer the Gaussian σ
    describes. Canny itself uses 7x7 at σ 1.4.
*   In part 3, a single threshold on the Sobel magnitude scores precision 0.600 on the noisy
    image and 1.000 after smoothing, but leaves bands about two pixels thick. In part 4,
    non-maximum suppression along the quantised gradient removes 62.2% of the total magnitude,
    thinning 18830 pixels to 6691.
*   Part 5 traces edges between two hysteresis thresholds:

    | Low | High | Edge pixels | Precision | Recall |
    | ---: | ---: | ---: | ---: | ---: |
    | 20 | 50 | 11843 | 0.284 | 1.000 |
    | 150 | 250 | 3045 | 1.000 | 0.913 |
    | 40 | 250 | 3161 | 1.000 | 0.947 |
    | none | 250 | 2616 | 1.000 | 0.786 |

*   The wide pair lets the high threshold decide what an edge is and the low one how far it is
    followed. Against `cv2.Canny` it gives 3270 pixels against 3269, every one within 1 px of
    the other. The later parts use the 40/250 edges.
*   Parts 6 and 7 vote for lines and circles:

    | Search | Votes | Time | Found |
    | :--- | ---: | ---: | :--- |
    | Lines, every angle | 568 980 | 36 ms | the 4 true lines are the top 4 peaks |
    | Lines, lane angles only | 195 982 | 19 ms | both lane lines; the other two are outside the range |
    | Circles, every 6° | 5 865 905 | 409 ms | 2 of 3 disks in the top 10 |
    | Circles, along the gradient | 191 801 | 22 ms | all 3 disks, ranks 1, 2 and 3 |

*   The gradient at a rim points along the radius, so it names the direction of the centre.
    Sampling every angle casts 31 times as many votes, and 8 of its top 10 are small circles on
    no disk. Both savings rest on a prior: the lane range cannot find a line outside it, and
    the gradient vote trusts every pixel's gradient.
*   In part 8, the generalised Hough transform files 279 template edge points into 43 of 72
    gradient-angle bins. The translation-only peak lands 0.07 px from the true reference point.
    In part 9, over 6 scales and 19 rotations (114 hypotheses in 0.1 s) the peak is scale 0.7,
    rotation 25°, 0.33 px off, which is the transform applied. The rotation step equals the 5°
    bin, so an angle between steps would be recovered only to the nearest one.

## Script 03: HOG and Haar detectors

HOG counts gradient directions in cells; the Haar features of the Viola-Jones detector compare
rectangle sums, which an integral image makes cheap.

*   Part 3 compares a star turned by 5°, a quarter of a 20° bin, with the original:

    | Voting | Distance relative to the star's own norm |
    | :--- | ---: |
    | Nearest bin | 0.913 |
    | Split between the two nearest bins | 0.791 |
    | Split between bins and between the four surrounding cells | 0.627 |

*   Split voting makes the change smaller and smoother; it does not make the descriptor
    rotation invariant.
*   In part 4, a 64x128 window gives 15x7 blocks of 36 values, 3780 in all. Over 40 fresh
    windows of each kind, the mean distance to three reference people is 0.456 for people, 0.675
    for clutter and 0.797 for cars, with no overlap between the groups. No classifier is trained
    on HOG here; the trained one is the Haar one.
*   In part 5, a 24x24 window holds 86 400 two-rectangle features and a 16x16 window 17 408, by
    enumeration and by closed form. In part 6, the hand-written integral image matches
    `cv2.integral` exactly. Slicing a rectangle takes 2.9 ms at 1x1 and 27.5 ms at 239x179 over
    2000 positions, while four lookups take 4.3 ms and 4.7 ms: one cost grows with the area and
    the other does not.
*   In part 7, AdaBoost runs 20 rounds over all 17 408 features. The first feature it picks is a
    top/bottom pair at (3, 5), the eye band against the cheeks. On rendered faces:

    | Threshold | Faces found | False alarms | Accuracy |
    | ---: | ---: | ---: | ---: |
    | 0.3 | 100.0% | 7.7% | 96.2% |
    | 0.5 | 99.9% | 0.4% | 99.8% |
    | 0.7 | 93.8% | 0.0% | 96.9% |

*   In part 8, on 3000 TinyFace faces against 3000 CIFAR-10 images at 16x16, the same 20 rounds
    reach 87.1% of faces at 13.3% false alarms, 86.9% accuracy against 50.0% for the larger
    class. The first round's weighted error is 0.222 there against 0.037 on rendered faces. The
    crops come already centred on the face, so this is window classification, not a detector
    scanning whole photographs.

## Script 04: Training mechanics

*   In part 1, a single sigmoid unit reaches 75% on XOR at best, over 100 starting points: both
    XOR pairs share the midpoint (0.5, 0.5), which would have to lie on both sides of one line.
*   Part 2 trains two-layer networks, counted as solved when every output is within 0.1 of its
    target:

    | Hidden units | Initial sd | Solved | Median epochs |
    | ---: | ---: | ---: | ---: |
    | 2 | 0.1 | 54% | 5245 |
    | 2 | 1.0 | 79% | 661 |
    | 4 | 1.0 | 100% | 505 |
    | 8 | 1.0 | 100% | 392 |

*   In the runs that fail, the hidden units saturate so that two inputs with different targets
    get the same hidden values, and the output sits between them at loss 0.125. Weights that
    start near zero start the two units nearly identical.
*   In part 3, the softmax cross-entropy gradient, softmax minus one-hot, matches a central
    difference to 8.11e-10. On logits [1000, 1001, 1002] the textbook softmax returns nan;
    subtracting the largest logit returns [0.09, 0.2447, 0.6652].
*   On MNIST, the 784-256-10 MLP of part 4, written in numpy, reaches 97.84% with 203 530
    parameters, and the CNN of parts 5 and 6 99.03% with 156 010. 200 960 of the MLP's
    parameters sit in its first layer, which shares nothing between positions. The two also
    differ in optimiser, depth, epochs, BatchNorm and Dropout, so this shows that this CNN beats
    this MLP, not that weight sharing alone explains it.
*   In part 6, training and evaluation mode on the same weights change 56 of 10 000 test
    predictions. Dropout at p = 0.3 zeroes 0.299 of a vector and scales the rest by 1/(1-p).
    BatchNorm matches its formulas in both modes to 1e-6.
*   Fed one image at a time, the first 1000 test images score 90.10% with both layer types in
    training mode, 98.40% with BatchNorm alone switched to evaluation and 98.80% with both.
    With a single image, training-mode BatchNorm normalises each channel by that image's own
    statistics and erases how strongly the map responded.
*   `torch.no_grad()` only stops recording gradients: a training-mode network still drops
    activations and updates the running statistics inside it.

## Script 05: Grid detection and pose assembly

*   Part 1 clusters anchors with 1 - IoU as the distance and the median as the centre, the YOLO
    recipe. The mean best IoU is 0.485 for k = 1, 0.713 for k = 3 and 0.769 for k = 5; the
    rest of the script uses k = 3.
*   In part 2, a 160x160 image at stride 16 with 3 anchors holds 300 slots. Encoding a box as a
    cell offset and log(w / anchor_w) and decoding it back costs 3.81e-06 px, and 2 of 12 066
    boxes find their slot taken.
*   Part 4 trains for 30 epochs, to a total loss of 0.086. In part 5, the hand-written per-class
    NMS keeps the same boxes as `torchvision.ops.batched_nms` in 500 of 500 test images, and in
    part 6 the detector reaches mAP@0.5 = 0.974 over 1493 unseen boxes. The objects are rendered
    and overlap by at most 0.1 IoU, so this shows the chain from grid to mAP working, not what a
    small detector reaches on photographs.
*   Part 7 applies the same encoding to 36 334 COCO val2017 boxes letterboxed to 416x416:

    | | Mean best IoU | Boxes whose slot is taken |
    | :--- | ---: | ---: |
    | 3 anchors, one 13x13 grid | 0.461 | 4617 (12.7%) |
    | 9 anchors, grids of 52, 26 and 13 | 0.614 | 860 (2.4%) |

*   For pose assembly, part 8 draws confidence maps and part affinity fields from the
    keypoints, and part 9 scores each candidate limb by the field's line integral and pairs
    greedily per limb. The maps
    come from the labels, so this isolates the association step:

    | Scenes | Rule | Precision | Recall |
    | :--- | :--- | ---: | ---: |
    | 200 rendered, two people overlapping | Affinity field | 98.8% | 78.0% |
    | | Shortest distance | 83.5% | 83.4% |
    | COCO, 150 images, people apart | Affinity field | 100.0% | 89.2% |
    | | Shortest distance | 94.3% | 94.9% |
    | COCO, 112 images, people overlapping | Affinity field | 97.4% | 83.2% |
    | | Shortest distance | 77.8% | 75.7% |

*   Where people stand apart the nearest candidate is usually right. Where they overlap,
    distance falls to 77.8% precision and the field keeps 97.4%: the field rejects a pairing
    whose band carries no direction, and distance never rejects anything.

## Script 05b: Loss spikes and step size

Script 05's loss falls smoothly in its table, yet every term rises together in epochs 25 and
26. This script reproduces the run with every optimiser step recorded, matching it epoch for
epoch, and tests the explanations one by one.

*   Part 3 shows it is not one batch. The heaviest batch of a rising epoch costs 3.20x and 2.24x
    the epoch's median, against 2.16x to 2.44x in quiet epochs. The loss and the update norm
    climb together over hundreds of steps.
*   Part 2 shows it is not Adam's denominator shrinking. From the quiet epochs to the rise, mean
    sqrt(v) moves by x0.91, while the update norm grows by x3.05 against x1.65 for the gradient
    norm. Adam travels x1.82 further per unit of gradient.
*   In part 4, the whole network moves: relative movement per step grows by x2.0 to x5.1 across
    the 17 parameter tensors. The class and objectness terms rise x18.4 and x10.3, the centre
    offsets, behind a sigmoid, x2.9.
*   Parts 5 and 6 each change one setting, from the same seed:

    | Run | Break-up starts at | Loss it left |
    | :--- | ---: | ---: |
    | Batch order 3407, rate 1e-3 (script 05's) | epoch 25 | 0.1247 |
    | Batch order 12345, rate 1e-3 | epoch 30 | 0.1175 |
    | Batch order 3407, rate 5e-4, 60 epochs | epoch 56 | 0.0599 |

*   Another batch order moves the break-up, and half the rate postpones it to a lower loss. At
    30 epochs the half-rate run stands at 0.1326 and looks clean, which is why it runs to 60.
*   Part 7 trains four times longer, and the run breaks up at epochs 25, 54, 86 and 117, 29 to
    32 epochs apart, each from a lower loss (0.1247, 0.0574, 0.0359, 0.0218). All of this points
    at the step size, not the data, but none of it measures the loss surface.

## Script 05c: Curvature and stability

This script measures the curvature of the loss surface epoch by epoch on the same run, with
Hessian-vector products and 12 power iterations on a probe set of 128 training images fixed
before training. The measurement never takes a step, and the 30 epoch losses come out identical
to script 05's. It takes about 45 minutes.

*   Part 2 tracks three curvatures after every epoch: λ_max of the Hessian, the curvature uᵀHu
    along the step the parameters actually took, and λ_max after Adam's scaling diag(1/√v̂).
*   In part 3, λ_max rises from 1450 to a plateau near 3100 by epoch 14, then falls into the
    break-up: 2837 at epoch 24, 2609 at epoch 25. The curvature along the step stays between 0.4
    and 1.0. What grows is the step length, from 0.019 to 0.075.
*   The Adam-scaled curvature reaches its highest value, 7402, in the last epoch before the
    break-up and falls to 5479 two epochs later.
*   Part 5 repeats the measurement at half the step size. The step size times each curvature
    at the first break-up:

    | Run | Break-up | Loss it left | Rate × λ_max | Rate × Adam-scaled |
    | :--- | ---: | ---: | ---: | ---: |
    | rate 1e-3 | epoch 25 | 0.1247 | 2.837 | 7.402 |
    | rate 5e-4 | epoch 56 | 0.0599 | 1.794 | 6.623 |

*   On the Adam-scaled curvature the two runs agree to 11%, on the raw curvature they are 37%
    apart. In part 6, over the four break-ups of the long run, the Adam-scaled product spans
    6.69 to 8.60 while the loss spans 5.7x and the raw curvature falls from 2772 to 1418.
*   Part 7 acts on the warning: a run that halves its rate when the product first passes 1.4x
    its median over epochs 5 to 10 does so at epoch 21 and never breaks up in 35 epochs, ending
    at 0.0626 against 0.0857.
*   The product is a diagnostic, not Adam's stability condition, and the Hessian is of the
    smooth part of the loss. Lowering the rate helps wherever it is applied, so the last run
    shows that acting on the warning is enough, not that the warning names the cause.

## Script 06: Segmentation and skip connections

Part 3 builds three networks that give every pixel a class, and differ in what they do about
resolution. Background covers 89.8% of the rendered dataset, so part 2's constant prediction
scores 89.78% pixel accuracy and 0.180 mean IoU. Part 4 trains all three, and part 5 also scores
them near class boundaries:

| Model | Parameters | Receptive field | Time | Pixel accuracy | Mean IoU | Boundary mean IoU |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| UNet with skips | 1 085 837 | about 89 px | 22 s | 99.71% | 0.964 | 0.957 |
| UNet without skips | 1 525 235 | about 89 px | 27 s | 98.02% | 0.769 | 0.743 |
| Flat, no resampling | 314 261 | about 13 px | 90 s | 92.01% | 0.371 | 0.390 |
| Background everywhere | 0 | n/a | n/a | 89.78% | 0.180 | n/a |

*   The UNet without skips is widened to 30 base channels and has 40% more parameters, and is
    still worse, so this compares two configurations rather than ablating one variable.
*   In part 5, of the interior shape pixels, the UNet without skips gives 9.0% to another shape:
    it finds the shape and names it wrongly. The flat network gives 64.3% to background, since
    with a 13 px field it cannot tell a large patch of colour from what surrounds it. It takes
    four times as long with under a third of the parameters, because every layer runs at full
    resolution.
*   Interior accuracy hides the interior errors: 97.2% of the interior is background.
*   One check without the deterministic setting moved the UNet without skips from 0.769 to
    0.888 mean IoU, so the size of the gap is one draw; its direction held.
*   Part 6 repeats this on Pascal VOC 2012 (1464 training and 1449 validation images at 128x128,
    trained from scratch, the 255 border label ignored):

    | Model | Pixel accuracy | Mean IoU | Classes ever predicted |
    | :--- | ---: | ---: | ---: |
    | UNet with skips | 71.54% | 0.059 | 8 |
    | UNet without skips | 73.12% | 0.045 | 3 |
    | Background everywhere | 73.33% | 0.035 | 1 |

*   Pixel accuracy falls below the baseline while mean IoU rises: predicting an object class
    costs background pixels, and only one of the two numbers pays for it.

## Script 07: Attention and self-supervised representations

*   In part 1, scaled dot-product attention with 4 heads matches `nn.MultiheadAttention` with
    identity projections to 3.73e-07. The largest weight in a row averages 0.510, and 0.852
    without the division by √24. In part 2, the post-norm encoder block with both sub-layers
    zeroed returns LayerNorm(LayerNorm(x)), not x.
*   Part 3 compares two tokenisers with 64 tokens each, three encoder blocks, eight epochs:

    | Tokeniser | Rendered, end to end | Rendered, probe | CIFAR-10, end to end | CIFAR-10, probe |
    | :--- | ---: | ---: | ---: | ---: |
    | Patches, one 4x4 square each | 84.30% | 84.75% | 57.30% | 57.70% |
    | Convolutional stem | 93.15% | 95.90% | 56.70% | 62.35% |

*   The stem's tokens overlap and carry local structure before attention starts. That is worth
    nine points on the rendered shapes; on CIFAR-10 the two measures disagree in direction, one
    run each.
*   Parts 4 to 7 train a supervised ConvNet, an autoencoder, a masked autoencoder and a
    contrastive encoder. Part 8 puts the six encoders through two evaluations that are never
    added together. The linear probe
    freezes the encoder and trains one fresh linear layer, and scores the representation.
    Fine-tuning trains a fresh head together with the encoder, 8 epochs for all, and scores the
    adapted model.

    | Representation | Labels in pretraining | Probe | Fine-tuned | CIFAR-10 probe | CIFAR-10 fine-tuned |
    | :--- | :--- | ---: | ---: | ---: | ---: |
    | Patch-token transformer | all | 84.75% | 88.80% | 57.70% | 56.00% |
    | Convolutional-token transformer | all | 95.90% | 98.40% | 62.35% | 61.00% |
    | Supervised ConvNet | all | 99.85% | 100.00% | 72.45% | 63.00% |
    | Autoencoder | none | 31.45% | 68.95% | 33.25% | 54.35% |
    | Masked autoencoder | none | 44.95% | 89.70% | 38.35% | 60.95% |
    | Contrastive, with projection head | none | 87.65% | 100.00% | 55.05% | 62.25% |
    | Contrastive, no projection head | none | 66.30% | n/a | 49.60% | n/a |

*   The gain from fine-tuning runs opposite to the probe: the least separable features gain
    the most. On the rendered shapes fine-tuning saturates. On CIFAR-10 the three encoders
    trained with labels lose, the supervised ConvNet 9.45 points, while the three label-free
    ones gain. One run per cell, so this shows the two evaluations can disagree in sign.
*   The encoders differ in architecture, width and epochs, so the order is of these six
    configurations, not a ranking of training methods. It fits reconstruction spending its
    capacity on the background it has to redraw.
*   The projection head is discarded after training and still improves the body under it: the
    probe gains 21.4 points on the rendered shapes and 5.45 on CIFAR-10, the best-controlled
    comparison here, from one seed.
*   CIFAR-10 here uses 12 000 training and 2000 held-out images from the official training
    batches, so these are comparisons, not test-set accuracies.

## Script 08: Field-level extraction audit

Part 1 draws a motor claim form with six known values, part 4 asks the model for them as one
JSON object, and part 5 compares each field with what was drawn. Valid JSON with every key
present says nothing about whether the values are right.

*   Five fields carry a trap, listed in part 2. `policy_number` is `IF-4821-77`, with a capital
    I where a 1 would sit. `vehicle_model` is `A6` on a small grey badge, under the line
    `Avant quattro 45 TFSI` in a larger, bolder face. `severity` is one filled box of three.
    `driver_name` is blacked out. `road_surface` is blank, with `Weather: Rain` in the same
    ruled row. `claim_amount` has no trap. The prompt asks for `REDACTED` and `BLANK` as
    explicit values.
*   Part 3 renders the form in English, French and German, with only the labels changed, and
    then as a photograph of each page: rotated 1.4 degrees, blurred, lit unevenly and saved at
    JPEG quality 55.
*   Three runs with `gemini-3.1-flash-lite` at temperature 0 and one through the
    `OPENAI_API_KEY` branch gave the same answers. All replies parsed with all six keys:

    | Field | Render, 3 languages | Photograph, 3 languages |
    | :--- | :--- | :--- |
    | `vehicle_model` | trim returned in all three | trim returned in French and German |
    | `road_surface` | `BLANK` in all three | `RAIN` in all three |
    | the other four | right | right |
    | Fields right | 15/18 (83%) | 13/18 (72%) |

*   On the clean page the model takes the more prominent string. On the photograph it also
    fills the empty field from the row's other value. The policy number, the ticked box, the
    redaction and the amount held in both conditions, which only the per-field breakdown shows.
*   Part 6 names each miss with `classify()`, testing the named traps first: trim taken for the
    badge, empty field filled from elsewhere, redaction read as empty or as content, wrong
    option, character misread, dropped key.

## Script 09: Grounding and failure modes

Three runs with Gemini and one through the OpenAI fallback printed identical output.

*   Part 2 tests visual grounding: the prompt asks for the front wheel's box in pixels and gets
    back `[261, 261, 812, 376]`, where 812 lies outside the 640-wide image. The script scores
    four readings:

    | Reading | Box | IoU |
    | :--- | :--- | ---: |
    | 0-1000 grid as y1 x1 y2 x2 | (167, 110, 241, 341) | 0.304 |
    | pixels as x1 y1 x2 y2 | (261, 261, 812, 376) | 0.000 |
    | pixels as y1 x1 y2 x2 | (261, 261, 376, 812) | 0.000 |
    | 0-1000 grid as x1 y1 x2 y2 | (167, 110, 520, 158) | 0.000 |

*   Read as the pixels the prompt asked for, the reply scores zero. The best reading holds the
    whole wheel inside a box 3.3 times its area, so the IoU is 1 / 3.3.
*   Parts 3 and 4 test dense small text: a departures board of 26 rows at 12 px, asked for every
    airline with a flight in zone A. The gate letter disagrees with the zone in 18 rows, so a
    reader who takes it for the zone lists 7 airlines. The model listed 9 of 9. Part 5's
    collapse check counts whole entries, not words: four airlines end in `Air`, so a word count
    would flag this correct reply.
*   In part 6, four follow-up questions ask for details the first reply never mentioned, with
    the image kept in the history or replaced by its text:

    | Probe | Drawn | Image kept | Image dropped |
    | :--- | :--- | :--- | :--- |
    | gate of row 1 | B5 | B5 | A12 |
    | time of row 1 | 16:20 | 16:20 | 08:15 |
    | airline of row 4 | Kestrel Airways | Kestrel Airw | Norwegian |
    | status of row 2 | Delayed | Delayed | DELAYED |

*   4/4 with the image, 1/4 without, and the one hit is a field with three possible values. A
    transcript stored as plain strings loses the image, and the invented answers come back in
    the same confident shape.

## Script 10: Video by keyframe sampling

An image model stands in for a video model: frames are sampled, each is asked one question, and
the answers are stitched together. Parts 1 and 2 render a clip of 4 s at 30 fps, encode it to
mp4 and read it back from the file. A scrape appears at frame 78 (2.60 s) and stays; a brake
lamp is on for three frames from frame 45.

*   Part 3 samples at a stride of 10: 12 of 120 frames, and none lands on the brake lamp. An
    event shorter than the stride can fall between samples, and no prompt recovers a frame that
    was never sent.
*   In part 4, each frame is asked `DAMAGED` or `CLEAN` in isolation. Three runs gave the same
    12 answers, all matching the frame as drawn. On part 5's timeline the first `DAMAGED` frame
    is 80, 0.07 s late: with every answer right, the estimate can only be late.
*   Part 6 re-reads the same answers at wider strides, with no extra calls:

    | Stride | Calls | Window | Estimate | Error |
    | ---: | ---: | ---: | ---: | ---: |
    | 10 | 12 | 0.33 s | 2.67 s | 0.07 s |
    | 20 | 6 | 0.67 s | 2.67 s | 0.07 s |
    | 30 | 4 | 1.00 s | 3.00 s | 0.40 s |
    | 40 | 3 | 1.33 s | 2.67 s | 0.07 s |

*   The window is what a stride guarantees; where the samples fall inside it is luck. Stride 40
    beats stride 30 here, which is no reason to sample less.
*   Reading frames one at a time can locate an event to within a window and says nothing about
    order or duration, which a model built for video would take the frames together to answer.

## Script 11: Document layout audit

Part 1 declares a two-column document with six sections in code, and part 2 renders it to PDF,
body at 9.5 pt and headings at 14 pt. Three sections get a flaw: a heading drawn just under the
12 pt classifier threshold, one fused onto its paragraph's line, and one drawn character by
character on a baseline staggered by 6 pt. At 6 pt PyMuPDF returns one line per character; at
5 pt it still reads one line.

*   Part 3 reads 58 text lines back. Sorting them top to bottom jumps back to the left column 3
    times, because it interleaves the columns; the reading order as extracted never does. The
    sorted pass is for comparison only.
*   Part 5 reconciles by title, and 3 of 6 headings are recovered: one demoted under the
    threshold, one fused with its paragraph, one shattered into 16 lines.
*   Part 4 counts lines at heading size and reports 19 headings for 6 sections. The count
    suggests more structure than there is, and errors can cancel: one demoted heading and one
    false heading leave it unchanged. Only matching by title names each failure.
*   Part 6 slices by recovered heading and gets 19 chunks. `'Excluded Events'` also holds two
    other sections, the last section sits under the one-letter heading `'s'`, and 15 chunks
    carry five words or fewer.

## Script 12: Convolution from first principles

*   In part 1, a 5x5 binary image and an X-shaped 3x3 kernel, convolved with explicit loops,
    match `nn.Conv2d` with `bias=False` exactly. Like `nn.Conv2d`, the loop does not flip the
    kernel, so strictly it is a cross-correlation. The kernel has 5 ones and no window covers
    all of them, so the maximum is 4.
*   In part 2, the output size is `(H + 2p - k) // s + 1`, checked for four padding and stride
    pairs: padding 1 keeps the 5x5 size, stride 2 about halves it.
*   Part 4's four directional kernels, half -1 and half +1, each sum to zero, so a flat region
    answers zero whatever its brightness. In part 5, the conv output on the 160x160 test image
    of part 3 is (1, 4, 157, 157), and a 2x2 pool keeps 25% of the cells.
*   Part 6 counts what each stage did. ReLU zeroes 2.3% of cells (2,226 of 98,596), each an edge
    running the wrong way. Counting every value below zero gives 26.2%, and 91.4% of those are
    floating-point residue around 1e-7 in flat regions, so the count needs a tolerance.
*   Along one row, the vertical kernels peak at columns 40 and 81, where the edges were drawn.
    The largest response in the whole map, 6.75 against the block's 5.96, is at the cut-off end
    of a brighter stroke (1.00 against 0.90): the kernel scores the size of the step, not the
    edge it was meant for.

## Script 13: Input resolution and network design

Four classes differ only in the gap between two 2x2 dots, 3, 5, 7 or 9 pixels, at random
position and orientation, with the same brightness in every class.

*   Part 2 traces a 32x32 input through ResNet-50's stages: it reaches 1x1 at layer4 where the
    design expects 7x7. Part 3 shows that after the stem, neighbouring cells sit 4 input pixels
    apart, so the 3-pixel gap is 0.75 cells wide and neighbouring classes differ by half a cell.
    Part 4 builds a network sized for the input and one below it, and parts 5 and 6 train and
    score all three:

    | | Parameters | Training time | Accuracy |
    | :--- | ---: | ---: | ---: |
    | ResNet-50 | 23,509,956 | 11 to 32 s | 96.7% to 99.8% |
    | SizedForInput | 27,396 | 1 to 3 s | 100.0% |
    | TooSmall | 152 | 1 to 2 s | 27.3% |

*   Ranges are quoted because the GPU loop is not bit-for-bit deterministic. ResNet-50 trains
    first, so its time includes the warm-up.
*   ResNet-50 scores 1.00 on gap_3, and its few errors fall on gap_7 and gap_9. A coarse map is
    not a small one: no stage holds fewer than 2,048 values (layer4), against 1,024 input
    pixels. The intuition that the mismatch hides the detail does not hold here.
*   `TooSmall`, one convolution with two channels and an 8x pool, lands 2.3 points above chance,
    so the task is not free. It differs from `SizedForInput` in depth, width and pooling at
    once.
*   What the mismatch costs is 858x the parameters and 10x to 15x the training time, and a 1x1
    final map, so the average pool has one cell and nothing downstream can tell where anything
    was.

## Script 14: Detection split audit and submission

160 images of textured plates with four defect classes (scratch, patch, hole, crack), 336
instances, each box recorded as it is painted. Labels are written as VOC XML, and part 2
converts them to YOLO lines and prints 336 of 336 boxes converted. It trains an ultralytics YOLO
model in about 85 s on the GPU.

*   Part 3 shows the split: 128 train, 2 validation and 30 test. Part 4 audits it per class
    before training: validation holds 5 instances, one or two per class.
*   In part 5, the trainer keeps the checkpoint that scored best on those two images:

    | Split | mAP@0.5 | Per class |
    | :--- | ---: | :--- |
    | val | 1.000 | 1.000 for all four |
    | test | 0.836 | scratch 0.784, patch 0.985, hole 0.762, crack 0.812 |

*   Of the 435 ways to pick 2 test images, 247 score 1.000 with the same weights. A perfect
    score on a split this size is not evidence of a perfect model.
*   Average precision is defined over the full ranked list, so scoring asks for detections down
    to confidence 0.001. At the viewing threshold of 0.25 only 58 of 751 test detections remain
    and the score is 0.797, with no warning.
*   Part 6 writes the 751 test predictions to three CSV files, then reads each back and scores
    it:

    | Submission | Rows | mAP@0.5 |
    | :--- | ---: | ---: |
    | as predicted | 751 | 0.8347 |
    | grouped by image_id | 751 | 0.8347 |
    | confidence set to 1.0 | 751 | 0.1173 |

*   The files hold whole-pixel coordinates, which score 0.8347 against 0.8360 before rounding.
    Reordering the rows changes nothing, because average precision ranks by confidence itself.
    A constant confidence makes every row tie, and the ranking falls back to file order.
    Neither edit moves a box: the metric belongs to the submitted list, not only to the
    detector.
