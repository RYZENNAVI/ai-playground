"""Write a plugin the way a platform expects one, and let its schema do the checking.

Demonstrates that a plugin is a typed handler, not just a function that fetches:
    1. Write the feed pages this plugin reads, so every later number is reproducible.
    2. Declare the input and output schema the editor validates connections against.
    3. Call the handler with arguments that satisfy the input schema.
    4. Call it with arguments that do not, and watch the schema refuse before any work.
    5. Check the rows that come back against the declared output schema.
    6. Read a page with a field missing, through a permissive mapper and a strict one.
    7. Read what paging costs, and where the page limit does and does not show.

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
        # The handler reports what it dropped, so the report is an output port too.
        # Leave it out and the editor has nothing to wire it to.
        "skipped": {"type": "array", "of": {
            "title": "string",
            "reason": "string",
        }},
    },
}

PY_TYPES = {"string": str, "integer": int, "array": list}


def matches_type(value, kind):
    """Say whether a value has the declared type, with bool refused as an integer.

    bool is a subclass of int in Python, so isinstance(True, int) is True. A
    schema that says integer means a count or a score, not a flag, and both
    directions of the contract have to hold that line or neither does.
    """
    return not isinstance(value, bool) and isinstance(value, PY_TYPES[kind])


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

    Every field is fetched with a default and nothing is converted, so a
    missing rating comes back as None and a present one comes back as the
    string the XML held. Both are the wrong shape for the declared row, and
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
        if not matches_type(args[name], rule["type"]):
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
        if not matches_type(result[name], rule["type"]):
            problems.append(f"{name} should be {rule['type']}")
    for name, rule in schema["output"].items():
        # A value that is not a list was already reported above; walking it would
        # crash the checker on exactly the return value it exists to describe.
        if "of" not in rule or not matches_type(result.get(name), "array"):
            continue
        for index, row in enumerate(result[name]):
            if not isinstance(row, dict):
                problems.append(f"{name}[{index}] should be a row, got "
                                f"{type(row).__name__}")
                continue
            for field, kind in rule["of"].items():
                if field not in row:
                    problems.append(f"{name}[{index}] has no {field}")
                elif not matches_type(row[field], kind):
                    problems.append(f"{name}[{index}].{field} should be {kind}, got "
                                    f"{type(row[field]).__name__}")
    # validate_args refuses an undeclared input; an undeclared output is the same
    # mistake from the other side - a port the handler fills that no wire can reach.
    for name in result:
        if name not in schema["output"]:
            problems.append(f"{name} is returned but not a declared output")
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
            skipped.append({"title": title.text if title is not None else "?",
                            "reason": str(error)})
    return {"items": items, "page": args["page"], "skipped": skipped}


def read_pages(app_id, page_limit):
    """Call the one-page handler until the source runs out or page_limit is reached.

    The limit is an argument to this loop, which sits outside the plugin. It is
    not in PLUGIN_SCHEMA, so a canvas drawing the node from that schema never
    shows it. Returns pages read and requests attempted separately: the request
    that finds the end of the source is still a request.
    """
    rows, skipped, pages_read, attempts = [], [], 0, 0
    for page in range(1, page_limit + 1):
        attempts += 1
        try:
            result = handler({"app_id": app_id, "page": page}, skip_invalid=True)
        except FileNotFoundError:
            break
        pages_read += 1
        rows.extend(result["items"])
        skipped.extend(result["skipped"])
    return rows, skipped, pages_read, attempts


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
    without_skipped = {**PLUGIN_SCHEMA, "output": {
        k: v for k, v in PLUGIN_SCHEMA["output"].items() if k != "skipped"}}
    print(f"  the same page against a schema that forgets 'skipped': "
          f"{validate_output(without_skipped, result)}")

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
        rows, skipped, pages_read, attempts = read_pages("ABC-Trade", limit)
        print(f"  page_limit={limit:<3} {pages_read} page(s) read in {attempts} request(s), "
              f"{len(rows)} rows, {len(skipped)} entry(s) skipped")
    for entry in skipped:
        print(f"    skipped {entry['title']!r}: {entry['reason']}")
    print(f"  the source holds {len(PAGES)} pages, so a limit of 20 reads {pages_read} and "
          f"sends {attempts}")
    print("  requests here - the last one only finds the end - and would send all 20")
    print("  against a source that keeps answering")
    print("  page_limit is an argument to the calling loop, not an input in PLUGIN_SCHEMA,")
    print("  so nobody wiring this node on a canvas sees it. Making it visible would mean")
    print("  declaring it as an input and moving the paging into the plugin itself")


if __name__ == "__main__":
    main()
