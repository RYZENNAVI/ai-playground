"""Write a plugin the way a platform expects one, and let its schema do the checking.

Demonstrates that a plugin is a typed handler, not just a function that fetches:
    1. Write the feed pages this plugin reads, so every later number is reproducible.
    2. Declare the input and output schema the editor validates connections against.
    3. Call the handler with arguments that satisfy the input schema.
    4. Call it with arguments that do not, and watch the schema refuse before any work.
    5. Check the rows that come back against the declared output schema.
    6. Read a page with a field missing, through a permissive mapper and a strict one.
    7. Read what paging costs, and put the limit in the contract instead of the loop.

Module 09: Low-Code Platforms - Plugin Input/Output Contract.
"""

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

DATA_DIR = Path(__file__).resolve().parent / "data"
FEED_DIR = DATA_DIR / "review_feed"
NS = {"atom": "http://www.w3.org/2005/Atom", "im": "http://itunes.apple.com/rss"}

# The transport here is the local filesystem. A hosted plugin would put an HTTP
# call in fetch_page and change nothing else: the contract is what the platform
# reads, and the contract says nothing about where the bytes come from.
PAGES = {
    1: [
        ("Fast and stable", "5", "2026-05-04T08:12:00Z", "Ada Whitfield",
         "Order entry is quick and the charts finally load on a weak connection."),
        ("Login keeps failing", "1", "2026-05-04T09:40:00Z", "Bruno Kestrel",
         "Face unlock fails every morning and the password screen rejects a valid password."),
        ("Fine for the price", "3", "2026-05-03T19:05:00Z", "Cleo Marchetti",
         "It does what it says. Nothing about it stands out either way."),
    ],
    2: [
        ("Lost my watchlist", "2", "2026-05-03T11:20:00Z", "Dario Penn",
         "The update wiped my watchlist and support has not replied in four days."),
        ("Good research tab", "4", "2026-05-02T15:48:00Z", "Esme Lindqvist",
         "The research tab is genuinely useful, though it buries the export button."),
        ("Charts unreadable at night", "2", "2026-05-02T22:31:00Z", "Falk Osei",
         "Dark mode turns the candles grey on grey, so I trade from a laptop instead."),
    ],
    # The third page carries one entry with no rating element at all, which is
    # what a real feed does when a reviewer leaves a comment without a score.
    3: [
        ("Alerts arrive late", None, "2026-05-01T07:02:00Z", "Greta Amara",
         "Price alerts land two or three minutes after the move has already happened."),
        ("Solid since the rewrite", "5", "2026-05-01T18:26:00Z", "Hugo Vance",
         "Whatever changed in the last release fixed the freeze on the options chain."),
    ],
}

# What the platform reads to draw the node and validate every wire into it.
PLUGIN_SCHEMA = {
    "name": "review_feed",
    "input": {
        "app_id": {"type": "string", "required": True},
        "page": {"type": "integer", "required": True},
    },
    "output": {
        "items": {"type": "array", "of": {
            "title": "string",
            "rating": "integer",
            "author": "string",
            "updated": "string",
            "content": "string",
        }},
        "page": {"type": "integer"},
    },
}

PY_TYPES = {"string": str, "integer": int, "array": list}


def write_feed_pages():
    """Write one Atom file per page, and report whether anything changed.

    The files are rewritten byte for byte on every run, so the script is
    idempotent and the counts printed further down cannot drift between runs.
    """
    FEED_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for page, entries in PAGES.items():
        lines = ['<?xml version="1.0" encoding="utf-8"?>',
                 '<feed xmlns="http://www.w3.org/2005/Atom" '
                 'xmlns:im="http://itunes.apple.com/rss">',
                 '  <title>ABC Trade customer reviews</title>',
                 f'  <updated>2026-05-04T12:00:00Z</updated>']
        for title, rating, updated, author, content in entries:
            lines.append("  <entry>")
            lines.append(f"    <title>{title}</title>")
            if rating is not None:
                lines.append(f"    <im:rating>{rating}</im:rating>")
            lines.append(f"    <updated>{updated}</updated>")
            lines.append(f"    <author><name>{author}</name></author>")
            lines.append(f"    <content type='text'>{content}</content>")
            lines.append("  </entry>")
        lines.append("</feed>")
        path = FEED_DIR / f"page{page}.atom"
        body = "\n".join(lines) + "\n"
        changed = not path.exists() or path.read_text(encoding="utf-8") != body
        path.write_text(body, encoding="utf-8")
        written.append((path.name, len(entries), changed))
    return written


def fetch_page(app_id, page):
    """Return the raw bytes of one feed page for one application id."""
    path = FEED_DIR / f"page{page}.atom"
    if not path.exists():
        raise FileNotFoundError(f"{app_id} has no page {page}")
    return path.read_text(encoding="utf-8")


def map_permissively(entry):
    """Map an entry by reading whatever elements happen to be there.

    Every field is fetched with a default, so a feed that stops sending one
    produces a row that still looks complete. The row is the wrong shape and
    nothing on the way out says so.
    """
    def text(path):
        node = entry.find(path, NS)
        return node.text if node is not None else None

    return {
        "title": text("atom:title"),
        "rating": text("im:rating"),
        "author": text("atom:author/atom:name"),
        "updated": text("atom:updated"),
        "content": text("atom:content"),
    }


def map_strictly(entry):
    """Map an entry against the declared row shape, naming any field that is absent."""
    row, missing = {}, []
    lookup = {"title": "atom:title", "rating": "im:rating",
              "author": "atom:author/atom:name", "updated": "atom:updated",
              "content": "atom:content"}
    for field, path in lookup.items():
        node = entry.find(path, NS)
        if node is None or node.text is None:
            missing.append(field)
            continue
        row[field] = int(node.text) if field == "rating" else node.text
    if missing:
        raise ValueError(f"entry is missing {missing}")
    return row


def validate_args(schema, args):
    """Return the reasons a call does not satisfy the declared input schema."""
    problems = []
    for name, rule in schema["input"].items():
        if name not in args:
            if rule.get("required"):
                problems.append(f"{name} is required and was not passed")
            continue
        expected = PY_TYPES[rule["type"]]
        if isinstance(args[name], bool) or not isinstance(args[name], expected):
            problems.append(f"{name} should be {rule['type']}, got "
                            f"{type(args[name]).__name__}")
    for name in args:
        if name not in schema["input"]:
            problems.append(f"{name} is not a declared input")
    return problems


def validate_output(schema, result):
    """Return the reasons a return value does not satisfy the declared output schema."""
    problems = []
    for name, rule in schema["output"].items():
        if name not in result:
            problems.append(f"{name} is missing from the return value")
            continue
        if not isinstance(result[name], PY_TYPES[rule["type"]]):
            problems.append(f"{name} should be {rule['type']}")
    for index, row in enumerate(result.get("items", [])):
        for field, kind in schema["output"]["items"]["of"].items():
            if field not in row:
                problems.append(f"items[{index}] has no {field}")
            elif not isinstance(row[field], PY_TYPES[kind]):
                problems.append(f"items[{index}].{field} should be {kind}, got "
                                f"{type(row[field]).__name__}")
    return problems


def handler(args, mapper=map_strictly, skip_invalid=False):
    """Run the plugin: check the arguments, fetch one page, map it to declared rows.

    skip_invalid decides what an unmappable entry costs. Dropping it keeps the
    page readable, but only if the count of what was dropped comes back too:
    a plugin that skips silently is the permissive mapper one level up.
    """
    problems = validate_args(PLUGIN_SCHEMA, args)
    if problems:
        raise ValueError("; ".join(problems))
    root = ET.fromstring(fetch_page(args["app_id"], args["page"]))
    items, skipped = [], []
    for entry in root.findall("atom:entry", NS):
        try:
            items.append(mapper(entry))
        except ValueError as error:
            if not skip_invalid:
                raise
            title = entry.find("atom:title", NS)
            skipped.append((title.text if title is not None else "?", str(error)))
    return {"items": items, "page": args["page"], "skipped": skipped}


def read_pages(app_id, page_limit):
    """Read pages until the source runs out or the declared limit is reached.

    page_limit belongs to the caller, not to the body of a loop. A plugin that
    hardcodes how far it will walk hides its own cost from whoever wires it up.
    """
    rows, skipped, fetched = [], [], 0
    for page in range(1, page_limit + 1):
        try:
            result = handler({"app_id": app_id, "page": page}, skip_invalid=True)
        except FileNotFoundError:
            break
        fetched += 1
        rows.extend(result["items"])
        skipped.extend(result["skipped"])
    return rows, skipped, fetched


def main():
    print("--- 1. The feed pages this plugin reads ---")
    for name, count, changed in write_feed_pages():
        state = "written" if changed else "unchanged"
        print(f"  {name:<12} {count} entries  {state}")
    print(f"  files live in {FEED_DIR.relative_to(DATA_DIR.parent)}; "
          f"rerunning rewrites them identically")

    print("\n--- 2. The schema the editor validates wires against ---")
    for side in ("input", "output"):
        for name, rule in PLUGIN_SCHEMA[side].items():
            detail = rule["type"]
            if "of" in rule:
                detail += " of {" + ", ".join(
                    f"{k}: {v}" for k, v in rule["of"].items()) + "}"
            flag = " (required)" if rule.get("required") else ""
            print(f"  {side:<6} {name:<7} {detail}{flag}")

    print("\n--- 3. A call that satisfies the input schema ---")
    result = handler({"app_id": "ABC-Trade", "page": 1})
    print(f"  returned {len(result['items'])} rows from page {result['page']}")
    for row in result["items"]:
        print(f"    {row['rating']}  {row['title'][:34]:<36} {row['author']}")

    print("\n--- 4. Calls that do not ---")
    for bad in ({"app_id": "ABC-Trade"},
                {"app_id": "ABC-Trade", "page": "1"},
                {"app_id": "ABC-Trade", "page": 1, "sort": "recent"}):
        problems = validate_args(PLUGIN_SCHEMA, bad)
        print(f"  {str(bad)[:52]:<54} {problems}")
    print("  each of these is refused before a single page is fetched")

    print("\n--- 5. The rows, checked against the output schema ---")
    for page in (1, 2):
        result = handler({"app_id": "ABC-Trade", "page": page})
        problems = validate_output(PLUGIN_SCHEMA, result)
        print(f"  page {page}: {len(result['items'])} rows, "
              f"{len(problems)} schema violation(s)")

    print("\n--- 6. The page whose first entry has no rating ---")
    loose = handler({"app_id": "ABC-Trade", "page": 3}, mapper=map_permissively)
    print(f"  permissive mapper returned {len(loose['items'])} rows and raised nothing")
    for row in loose["items"]:
        print(f"    rating={row['rating']!r:<6} {row['title']}")
    problems = validate_output(PLUGIN_SCHEMA, loose)
    print(f"  the output schema finds {len(problems)}: {problems[:2]}")
    try:
        handler({"app_id": "ABC-Trade", "page": 3}, mapper=map_strictly)
    except ValueError as error:
        print(f"  strict mapper stops instead, and names the field: {error}")
    print("  one of these hands the next node a rating of None; the other hands it nothing")

    print("\n--- 7. What paging costs the caller ---")
    for limit in (1, 3, 20):
        rows, skipped, fetched = read_pages("ABC-Trade", limit)
        print(f"  page_limit={limit:<3} fetched {fetched} page(s), {len(rows)} rows, "
              f"{len(skipped)} entry(s) skipped")
    for title, reason in skipped:
        print(f"    skipped {title!r}: {reason}")
    print("  the source holds 3 pages, so a limit of 20 costs 3 fetches here and")
    print("  20 against a source that keeps answering; the number belongs in the")
    print("  contract, where whoever wires the node can see it")


if __name__ == "__main__":
    main()
