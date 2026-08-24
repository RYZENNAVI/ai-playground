"""Audit what survives when a PDF is parsed back into blocks, against a structure known in advance.

Demonstrates why a parsed document needs reconciling before anything is built on it:
    1. Declare a document structure in code, so every heading and paragraph is known.
    2. Render it to a PDF, including the three layouts that break naive parsers.
    3. Read the page back as positioned blocks and spans.
    4. Classify blocks into headings and body the way a layout parser does, by type size.
    5. Reconcile the recovered headings against the declared ones and report the recall.
    6. Slice the document by recovered heading and show what the missed ones cost.

Module 06: Multimodal Vision - Document Layout Audit.
"""

import re
import sys
from pathlib import Path

import pymupdf

sys.stdout.reconfigure(encoding="utf-8")

OUT_DIR = Path(__file__).parent / "outputs" / "layout_audit"
PDF_PATH = OUT_DIR / "policy_handbook.pdf"

PAGE_WIDTH, PAGE_HEIGHT = 595, 842
MARGIN = 56
COLUMN_GAP = 24
COLUMN_WIDTH = (PAGE_WIDTH - 2 * MARGIN - COLUMN_GAP) / 2

BODY_SIZE = 9.5
HEADING_SIZE = 14.0
BODY_FONT = "helv"
HEADING_FONT = "hebo"

# How far a display title's characters ride above and below the baseline. Six
# points is where PyMuPDF stops seeing one line and starts seeing one line per
# character, which is the same shape the artefact takes in real parser output.
BASELINE_STAGGER = 6

# The threshold a size-based classifier uses. Anything at or above it is called a
# heading. Every failure mode below is a way of getting past this one number.
HEADING_SIZE_THRESHOLD = 12.0

# The document, declared before it is drawn. Each entry carries the flaw that its
# heading is rendered with, which is what makes the reconciliation checkable:
#   "clean"   - full heading size and weight
#   "demoted" - drawn a shade above body size, under the classifier threshold
#   "fused"   - drawn on the same line as the paragraph that follows it
#   "art"     - drawn character by character on a staggered baseline
SECTIONS = [
    ("Scope of Cover", "clean", [
        "This handbook states what the policy covers and the conditions attached to each "
        "benefit. Where a benefit is limited, the limit is given as an amount per claim.",
        "Cover starts at the inception date shown on the schedule and ends on the expiry "
        "date, unless cancelled earlier under the terms set out in a later section.",
    ]),
    ("Excluded Events", "clean", [
        "No benefit is payable for loss arising from wear, gradual deterioration, or any "
        "event that was already in progress when the policy incepted.",
        "Loss caused deliberately by the policyholder is excluded in full, and any premium "
        "paid in respect of the affected period is forfeited.",
    ]),
    ("Making a Claim", "demoted", [
        "Notify the claims team within thirty days of the event. A claim reported after "
        "that window is assessed only where the delay itself is shown to be unavoidable.",
        "Supporting evidence must include the incident reference, photographs of the damage, "
        "and a repair estimate from an approved workshop.",
    ]),
    ("Settlement Basis", "fused", [
        "Settlement is on an indemnity basis. The insurer pays the cost of putting the "
        "policyholder back in the position held immediately before the loss, no better.",
        "Where an item cannot be repaired, the insurer pays its market value at the date of "
        "loss, less any excess recorded on the schedule.",
    ]),
    ("Renewal and Cancellation", "clean", [
        "Renewal terms are issued twenty-one days before expiry. Continuing to pay premium "
        "after that date is taken as acceptance of the renewal terms.",
        "The policyholder may cancel at any time. The insurer refunds premium for the unused "
        "period, less an administration charge.",
    ]),
    ("Handling Disputes", "art", [
        "A policyholder who disagrees with a decision may ask for it to be reviewed. The "
        "review is carried out by someone who took no part in the original decision.",
        "If the review does not settle the matter, the policyholder may refer it to the "
        "independent adjudicator named on the schedule.",
    ]),
]


def draw_paragraph(page, text, rect, size=BODY_SIZE, font=BODY_FONT):
    """Fill a rectangle with wrapped text and return the height actually used.

    insert_textbox returns the space left over when the text fits and a negative
    number when it does not, so the caller can lay the next block out underneath.
    """
    leftover = page.insert_textbox(rect, text, fontname=font, fontsize=size, align=0)
    if leftover < 0:
        raise ValueError(f"text did not fit in {rect}: {text[:40]!r}")
    return rect.height - leftover


def draw_art_heading(page, text, x, y):
    """Draw a heading one character at a time on a baseline that rises and falls.

    Display titles are often set this way. Nothing is wrong with the characters,
    but each one is its own positioned glyph, and a parser that groups glyphs into
    lines by their vertical position cannot put a staggered row back into one line.
    """
    cursor = x
    for index, char in enumerate(text):
        offset = -BASELINE_STAGGER if index % 2 == 0 else BASELINE_STAGGER
        page.insert_text(
            (cursor, y + offset), char, fontname=HEADING_FONT, fontsize=HEADING_SIZE
        )
        cursor += pymupdf.get_text_length(char, fontname=HEADING_FONT, fontsize=HEADING_SIZE)


def build_pdf(path):
    """Render SECTIONS into a two-column PDF and return the declared heading list."""
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open()
    page = doc.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)

    columns = [MARGIN, MARGIN + COLUMN_WIDTH + COLUMN_GAP]
    column_index = 0
    y = MARGIN
    declared = []

    for title, flaw, paragraphs in SECTIONS:
        needed = 120
        if y + needed > PAGE_HEIGHT - MARGIN:
            column_index += 1
            y = MARGIN
            if column_index >= len(columns):
                page = doc.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
                column_index = 0
        x = columns[column_index]
        declared.append({"title": title, "flaw": flaw, "page": doc.page_count - 1})

        if flaw == "art":
            draw_art_heading(page, title, x, y + HEADING_SIZE)
            y += HEADING_SIZE + 10
        elif flaw == "fused":
            # The heading and the first sentence share one line, so they end up in
            # one span with one size. The heading is not lost, it is glued.
            merged = f"{title}  {paragraphs[0]}"
            rect = pymupdf.Rect(x, y, x + COLUMN_WIDTH, y + 90)
            y += draw_paragraph(page, merged, rect) + 8
            paragraphs = paragraphs[1:]
        else:
            size = HEADING_SIZE if flaw == "clean" else BODY_SIZE + 1.5
            rect = pymupdf.Rect(x, y, x + COLUMN_WIDTH, y + size * 2)
            draw_paragraph(page, title, rect, size=size, font=HEADING_FONT)
            y += size * 1.6

        for paragraph in paragraphs:
            rect = pymupdf.Rect(x, y, x + COLUMN_WIDTH, y + 110)
            y += draw_paragraph(page, paragraph, rect) + 8

    doc.save(path)
    doc.close()
    return declared


def read_blocks(path, sort):
    """Return one record per extracted line, carrying its largest span size.

    sort=True asks PyMuPDF to order blocks top-to-bottom before returning them,
    which is the wrong order for a two-column page: it interleaves the columns.
    """
    doc = pymupdf.open(path)
    records = []
    for number, page in enumerate(doc):
        page_dict = page.get_text("dict", sort=sort)
        for block in page_dict["blocks"]:
            if block["type"] != 0:
                continue
            for line in block["lines"]:
                text = "".join(span["text"] for span in line["spans"]).strip()
                if not text:
                    continue
                records.append(
                    {
                        "page": number,
                        "text": text,
                        "size": max(span["size"] for span in line["spans"]),
                        "font": line["spans"][0]["font"],
                        "x": round(line["bbox"][0], 1),
                        "y": round(line["bbox"][1], 1),
                    }
                )
    doc.close()
    return records


def classify(records):
    """Split lines into headings and body using the size threshold, and nothing else."""
    for record in records:
        record["kind"] = "heading" if record["size"] >= HEADING_SIZE_THRESHOLD else "body"
    return records


def normalise(text):
    """Strip case and spacing so a scrambled heading can still be compared."""
    return re.sub(r"[^a-z]", "", text.lower())


def find_shattered(headings, target):
    """Return the run of consecutive heading lines that spells out target, if one does.

    A heading broken into one line per character is still present in the output,
    so counting heading lines makes the document look richer in headings than it
    is. Only rejoining the run in order shows what happened.
    """
    for start in range(len(headings)):
        joined = ""
        for end in range(start, len(headings)):
            joined += normalise(headings[end]["text"])
            if joined == target and end > start:
                return headings[start : end + 1]
            if not target.startswith(joined):
                break
    return None


def reconcile(declared, records):
    """Match every declared heading against the recovered lines and label the outcome."""
    headings = [r for r in records if r["kind"] == "heading"]
    body = [r for r in records if r["kind"] == "body"]
    results = []
    for entry in declared:
        target = normalise(entry["title"])

        exact = next((r for r in headings if normalise(r["text"]) == target), None)
        if exact is not None:
            results.append({**entry, "outcome": "recovered", "seen_as": exact["text"]})
            continue

        run = find_shattered(headings, target)
        if run is not None:
            pieces = " ".join(repr(r["text"]) for r in run[:6])
            results.append(
                {**entry, "outcome": "shattered", "seen_as": f"{len(run)} lines: {pieces} ..."}
            )
            continue

        # A demoted heading is still its own line, so it matches exactly before
        # any fused-into-a-paragraph test gets a chance to claim it.
        demoted = next((r for r in body if normalise(r["text"]) == target), None)
        if demoted is not None:
            results.append({**entry, "outcome": "demoted", "seen_as": demoted["text"]})
            continue

        fused = next((r for r in body if r["text"].startswith(entry["title"])), None)
        if fused is not None:
            results.append({**entry, "outcome": "fused", "seen_as": fused["text"][:56] + "..."})
            continue

        results.append({**entry, "outcome": "lost", "seen_as": ""})
    return results


def slice_by_heading(records):
    """Cut the document into one chunk per recovered heading, the usual next step."""
    chunks = []
    for record in records:
        if record["kind"] == "heading":
            chunks.append({"heading": record["text"], "lines": []})
        elif chunks:
            chunks[-1]["lines"].append(record["text"])
    return chunks


def main():
    print("=" * 78)
    print("--- 1. A document structure declared before anything is drawn ---")
    print(f"{len(SECTIONS)} sections, each with a heading and body text")
    for title, flaw, paragraphs in SECTIONS:
        words = sum(len(p.split()) for p in paragraphs)
        print(f"  {title:<28} rendered as {flaw:<8} {words:>3} words of body")

    print()
    print("--- 2. Rendering it to a two-column PDF ---")
    declared = build_pdf(PDF_PATH)
    print(f"wrote {PDF_PATH}")
    print(f"  {PAGE_WIDTH}x{PAGE_HEIGHT} points, two columns of {COLUMN_WIDTH:.0f}")
    print(f"  body {BODY_SIZE}pt {BODY_FONT}, headings {HEADING_SIZE}pt {HEADING_FONT}")

    print()
    print("--- 3. Reading the page back as positioned lines ---")
    records = read_blocks(PDF_PATH, sort=False)
    print(f"{len(records)} text lines recovered")
    for record in records[:4]:
        print(f"  x={record['x']:>6} y={record['y']:>6} {record['size']:>5.1f}pt "
              f"{record['font']:<12} {record['text'][:46]!r}")

    sorted_records = read_blocks(PDF_PATH, sort=True)
    first_column_x = min(r["x"] for r in records)
    flips = sum(
        1
        for earlier, later in zip(sorted_records, sorted_records[1:])
        if earlier["x"] > first_column_x + 10 and later["x"] <= first_column_x + 10
    )
    print(f"sorting the same lines top-to-bottom jumps back to the left column {flips} times")
    print("  a two-column page has two reading orders, and vertical position picks the wrong one")

    print()
    print("--- 4. Classifying lines by type size ---")
    classify(records)
    headings = [r for r in records if r["kind"] == "heading"]
    print(f"{len(headings)} lines are at or above {HEADING_SIZE_THRESHOLD}pt and are called headings:")
    for record in headings[:8]:
        print(f"  {record['size']:>5.1f}pt  {record['text']!r}")
    if len(headings) > 8:
        print(f"  ... and {len(headings) - 8} more, most of them a single character wide")

    print()
    print("--- 5. Reconciling against the declared headings ---")
    results = reconcile(declared, records)
    for result in results:
        seen = f" -> {result['seen_as']!r}" if result["seen_as"] else ""
        print(f"  {result['title']:<28} {result['outcome']:<10}{seen}")
    good = sum(1 for r in results if r["outcome"] == "recovered")
    print(f"heading recall {good}/{len(results)} = {good / len(results):.0%}")
    print(f"  counting heading lines instead would have reported {len(headings)} headings "
          f"for {len(results)} sections, and got both the number and the direction wrong")

    print()
    print("--- 6. What the missed headings cost downstream ---")
    chunks = slice_by_heading(records)
    print(f"slicing by recovered heading gives {len(chunks)} chunks for {len(SECTIONS)} sections")
    substantial = [c for c in chunks if len(" ".join(c["lines"]).split()) > 5]
    for chunk in substantial:
        joined = " ".join(chunk["lines"])
        swallowed = [
            entry["title"]
            for entry in declared
            if entry["title"] != chunk["heading"] and entry["title"] in joined
        ]
        note = f"  <- also holds {', '.join(swallowed)}" if swallowed else ""
        print(f"  {chunk['heading'][:34]!r:<38} {len(joined.split()):>4} words{note}")
    print(f"  the remaining {len(chunks) - len(substantial)} chunks carry five words or fewer, "
          f"one per character of the shattered heading")
    print()
    print("the check that catches all of this is one line: recovered heading count against "
          "the count the document is known to have")
    print("=" * 78)


if __name__ == "__main__":
    main()
