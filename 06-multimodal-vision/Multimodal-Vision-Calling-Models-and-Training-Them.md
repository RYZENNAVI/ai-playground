# Multimodal Vision: Calling Models and Training Them

Seven scripts, in two halves that answer the same question from opposite ends:
**how does the information in a picture become something a program can use?**

The first half hands the picture to a model somebody else trained and audits what
comes back. The second half trains a model on pictures nobody else has. Both halves
fail the same way — the output is well-formed, nothing raises, and the only way to
find out it is wrong is to check it against a value that was known in advance.

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
asserted: the third row of the failure taxonomy in script 01 is a character-level
misread, which is the OCR boundary; and script 02 measures what a model returns when
asked to point.

---

## 1. Field-level extraction audit

`01_vlm_field_extraction_audit.py`

A model that returns valid JSON with every requested key has demonstrated nothing.
This script renders claim forms whose every value it chose itself, asks for those
values back, and scores field by field.

### The form

Six fields, five of which carry a deliberate trap:

```python
TRAPS = {
    "policy_number": "characters that share a shape: the letter I against the digit 1",
    "vehicle_model": "a badge whose last character decides the model",
    "severity": "three boxes, one of them ticked",
    "driver_name": "a field blacked out on the page",
    "road_surface": "a field left blank, with a filled neighbour to borrow from",
}
```

Each trap is something a person reads correctly and an extractor gets wrong in a way
that still looks like an answer:

- **`policy_number`** is `IF-4821-77` — a capital I where a digit 1 would sit.
- **`vehicle_model`** is printed inside `Audi A6 Avant`; only the last character
  separates it from a different model.
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
| Clean render, three languages | **18/18 fields — 100%** |
| Photograph of the same three pages | **15/18 fields — 83%** |

The three failures are the same field, in all three languages:

```
english   photo   road_surface     wanted 'BLANK', got 'RAIN'
french    photo   road_surface     wanted 'BLANK', got 'RAIN'
german    photo   road_surface     wanted 'BLANK', got 'RAIN'
```

The empty field was filled in from its neighbour. Under a clean render the model
answered `BLANK` correctly every time; blur and compression were enough to make the
adjacent value migrate one row up.

Everything else held: the shape-ambiguous policy number, the badge character, the
ticked box and the redaction all survived both conditions.

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

## 2. Grounding and failure modes

`02_vlm_grounding_and_failure_modes.py`

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

## 3. Video by keyframe sampling

`03_video_keyframe_understanding.py`

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
on a third of the calls — which is luck, not a reason to sample less.

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

## 4. Document layout audit

`04_document_layout_audit.py`

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

**The check that catches all of this is one line: recovered heading count against the
count the document is known to have.** Reading the markdown and finding it fluent
catches none of it.

---

## 5. Convolution from first principles

`05_conv_kernels_and_feature_maps.py`

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

The activation zeroed **26.2%** of cells, every one of them an edge running the wrong
way for its kernel. Pooling kept **25%** of the cells.

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

## 6. Input resolution and network design

`06_cnn_input_resolution_mismatch.py`

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

### Two networks, same data

| | Parameters | Training time | Accuracy |
| :--- | ---: | ---: | ---: |
| ResNet-50 | 23,509,956 | 11.9 s | 96.7% – 99.8% |
| SizedForInput | 27,396 | 0.9 s | 100.0% |

(Two runs are quoted for the larger network because this loop is not bit-for-bit
deterministic on GPU; the smaller one reached 100% in both.)

Chance is 25%. Both loops finish and neither raises anything.

### The result, stated as the numbers support it

> the mismatch did NOT hide the fine detail. The stem samples at stride 2, but each
> of its kernels still spans 7 input pixels, so a 3-pixel gap survives in the channel
> values even once the map has gone coarse

What the mismatch actually cost is measurable elsewhere:

- **858× the parameters** and **12–13× the training time**, for the same accuracy
- **a 1×1 final feature map**, so the pooling that follows averages a single cell —
  nothing downstream can ask *where*

"It cannot see the detail" was the intuition. The run says otherwise, and the run is
what gets reported. The usable conclusion is not that the architecture is blind, it is
that **the reason to size a network to its input is what you pay, and what shape comes
out the other end.**

---

## 7. Detection: auditing a split, and pricing a submission

`07_yolo_split_audit_and_submission.py`

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

## What the seven runs settle

1. **A reply that parses is not a result.** Script 01 scored 100% on clean renders and
   83% on photographs of the same pages, with the same JSON shape in both.
2. **Name the kind of mistake, not just the count.** Each of the four kinds in script
   01 has a different fix; a single accuracy number points at none of them.
3. **Coordinates come with an unstated convention.** Script 02 read the same four
   numbers six ways, scoring 0.304 under one and 0.000 under the rest.
4. **The unit a check counts must be the unit the failure repeats.** A word-level
   repetition test called a correct answer degenerate because four airline names
   contained the word Air.
5. **An image is an attachment, not a memory.** 4/4 with the image in the history,
   1/4 without, and the wrong answers were as confident as the right ones.
6. **Sampling buys a bound, not an estimate.** An event shorter than the stride is not
   hard to see; it is not sampled.
7. **Count the structure you recovered against the structure you know is there.**
   Script 04 recovered 3 of 6 headings while reporting 19 heading lines.
8. **A kernel scores contrast, not alignment.** Script 05's strongest response was on
   a brighter diagonal, not on the axis-aligned edge it was built for.
9. **Measure the cost, not the intuition.** Script 06 set out to show a resolution
   mismatch hiding fine detail and found it did not — the cost is 858× the parameters
   and a 1×1 output, not accuracy.
10. **A metric is computed over a list.** Script 07 moved a submission from 0.836 to
    0.117 by rewriting one column and no coordinates.

The thread through all ten: **every one of these was found by comparing an output
against a value that was known in advance.** None would have surfaced from reading the
output and finding it plausible.

---

## Running them

```bash
pip install -r ../requirements.txt
python 05_conv_kernels_and_feature_maps.py      # no key, no network
python 04_document_layout_audit.py              # no key, no network
python 06_cnn_input_resolution_mismatch.py      # no key, GPU optional
python 07_yolo_split_audit_and_submission.py    # no key, GPU optional
python 01_vlm_field_extraction_audit.py         # needs a vision model key
python 02_vlm_grounding_and_failure_modes.py    # needs a vision model key
python 03_video_keyframe_understanding.py       # needs a vision model key
```

Scripts 01–03 read `GEMINI_API_KEY` or `OPENAI_API_KEY` from `.env` and default to a
small vision model, overridable with `VISION_MODEL`. They send batches of images, so
each one paces itself and retries on a rate limit rather than failing part way
through.

Everything each script needs is generated on the first run into `outputs/`: the claim
forms, the scene and board, the clip, the PDF, the feature maps, the training samples
and the detection dataset. That directory is disposable — deleting it costs one rerun.
