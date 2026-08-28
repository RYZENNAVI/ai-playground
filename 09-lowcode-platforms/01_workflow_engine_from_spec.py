"""Run a visual workflow from its exported definition, without the visual editor.

Demonstrates that a drag-and-drop workflow is a typed graph plus a node registry:
    1. Load three workflow definitions and print the node types they are built from.
    2. Resolve every port reference, then repeat it on a definition whose one edited
       reference points at a field no upstream node emits.
    3. Order the nodes topologically, and refuse a graph that contains a cycle.
    4. Execute the main workflow node by node, printing what each node contributed.
    5. Read what the code nodes actually do, including one regular expression whose
       character class excludes four letters instead of one word.
    6. Run the same batch body as isolated iterations and as a state-carrying loop.
    7. Read the branch the selector wrote, and drop the items it marked.
    8. Follow both sub-workflow calls, bounded by a call-depth guard.

Module 09: Low-Code Platforms - Workflow Engine.
"""

import json
import re
import sys
from copy import deepcopy
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

SPEC_DIR = Path(__file__).resolve().parent / "data" / "workflows"
MAX_CALL_DEPTH = 3

# What the two plugin nodes return. A plugin is just a function the platform knows
# the input and output shape of; where the rows come from is the plugin's business.
NEWS_FIXTURE = [
    {"title": "Index closes higher on rate relief",
     "body": "The benchmark index gained 1.2 percent. Turnover stayed thin all session.",
     "published": "2026-05-04 18:20"},
    {"title": "Broker cuts commission on index funds",
     "body": "A mid-sized broker cut its fee to four basis points. Rivals have not followed.",
     "published": "2026-05-04 09:05"},
    {"title": "Quarterly filings land next week",
     "body": "Seventeen listed brokers file next Tuesday. Analysts expect flat revenue.",
     "published": "2026-05-03 21:40"},
    {"title": "Settlement window shortens in June",
     "body": "The exchange confirmed a shorter settlement window. Members must certify by June.",
     "published": "2026-05-04 11:55"},
]

REVIEW_FIXTURE = [
    {"title": "Fast and stable", "rating": 5,
     "body": "Order entry is quick and the charts finally load on a weak connection."},
    {"title": "Login keeps failing", "rating": 1,
     "body": "Face unlock fails every morning and the password screen rejects a valid password."},
    {"title": "Fine for the price", "rating": 3,
     "body": "It does what it says. Nothing about it stands out either way."},
    {"title": "Lost my watchlist", "rating": 2,
     "body": "The update wiped my watchlist and support has not replied in four days."},
    {"title": "Good research tab", "rating": 4,
     "body": "The research tab is genuinely useful, though it buries the export button."},
]

# One paragraph per scene, in the shape a model is asked to produce them.
SCENE_TEXT = (
    "Scene1: A red kite rises over an empty green field at dawn.\n"
    "Scene2: The same field at noon, seen from beneath a bending tree.\n"
    "Scene3: Evening arrives and the kite is a dark speck against orange cloud.\n"
)


# --- Plugin registry -------------------------------------------------------

def plugin_news_feed(topic):
    """Return market articles for a topic. Stands in for a remote data source."""
    return {"items": [dict(row, topic=topic) for row in NEWS_FIXTURE]}


def plugin_review_feed(app_id):
    """Return store reviews for one application id."""
    return {"items": [dict(row, app_id=app_id) for row in REVIEW_FIXTURE]}


PLUGINS = {"news_feed": plugin_news_feed, "review_feed": plugin_review_feed}


# --- Code node registry ----------------------------------------------------

def code_same_calendar_day(today, published):
    """Return 1 when a published timestamp falls on the reference date.

    The date is compared as two integers rather than as text, so a timestamp
    written as '2026-05-04 09:05' and a date written as '2026-05-04' still meet.
    """
    day = published.split(" ")[0]
    return {"same_day": 1 if day == today else 0}


def code_keep_marked(values, marks):
    """Keep the values whose parallel mark is the string 'keep'."""
    return {"kept": [v for v, m in zip(values, marks) if m == "keep"]}


def code_split_by_verdict(verdicts, digests):
    """Split digests into two lists according to the verdict beside each one.

    The comparison is a string equality against a fixed vocabulary. Whatever
    produced the verdict has to keep emitting exactly these words; script 02
    is about what happens when it stops.
    """
    positive = [d for v, d in zip(verdicts, digests) if v == "positive"]
    negative = [d for v, d in zip(verdicts, digests) if v == "negative"]
    return {"positive": positive, "negative": negative}


CODE_FNS = {
    "same_calendar_day": code_same_calendar_day,
    "keep_marked": code_keep_marked,
    "split_by_verdict": code_split_by_verdict,
}


# --- Model node registry ---------------------------------------------------

STOPWORDS = {"the", "a", "an", "and", "on", "in", "at", "of", "to", "its", "it",
             "is", "are", "has", "have", "not", "for", "by", "all", "next"}


def model_node(title, args):
    """Answer a model node with a deterministic local handler.

    Nothing here calls a model. This script is about the engine around the node,
    so every handler has to return the same thing on every run; script 02 puts a
    real model behind this same node type and measures what it returns.
    """
    if title == "Digest":
        first = args["body"].split(".")[0].strip()
        return {"text": f"{args['title']} - {first}."}
    if title == "Classify":
        rating = args.get("rating", 3)
        verdict = "positive" if rating >= 4 else "negative" if rating <= 2 else "neutral"
        return {"verdict": verdict, "digest": args["title"]}
    if title == "Hotwords":
        counts = {}
        for line in args["lines"]:
            for word in re.findall(r"[a-z]{4,}", line.lower()):
                if word not in STOPWORDS:
                    counts[word] = counts.get(word, 0) + 1
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        return {"text": ", ".join(w for w, c in ranked[:6])}
    if title.startswith("Summarise"):
        lines = args["lines"]
        head = lines[0] if lines else "nothing recorded"
        return {"text": f"{len(lines)} item(s); first is: {head}"}
    raise KeyError(f"no handler registered for model node {title!r}")


# --- Reference resolution --------------------------------------------------

def declared_ports(spec):
    """Map every node id in a definition to the ports it says it emits."""
    ports = {}
    for node in spec["nodes"]:
        ports[node["id"]] = list(node.get("ports", []))
        for inner in node.get("body", []):
            ports[inner["id"]] = list(inner.get("ports", []))
    return ports


def resolve(ref, ctx, item=None):
    """Turn one reference string into a value.

    Three forms exist, and the platform's editor writes all three:
    'literal:x' is a constant, 'item.field' reads the current batch element,
    and 'node.port' reads a port another node has already produced.
    """
    if ref.startswith("literal:"):
        return ref[len("literal:"):]
    source, _, port = ref.partition(".")
    if source == "item":
        if item is None:
            raise KeyError("'item' referenced outside a batch body")
        return item[port]
    return ctx[source][port]


def validate(spec, library):
    """Return every reference in one definition that cannot be satisfied.

    This is the check the editor is expected to run before the graph is allowed
    to start. A reference that names a port nobody declares is a mistake that
    the canvas will happily draw and the runtime will only find on the way past.
    """
    ports = declared_ports(spec)
    known = set(ports) | {"item"}
    problems = []

    def check(node, in_batch):
        for name, ref in node.get("inputs", {}).items():
            if ref.startswith("literal:"):
                continue
            source, _, port = ref.partition(".")
            if source == "item":
                if not in_batch:
                    problems.append(f"{node['id']}.{name} reads 'item' outside a batch")
                continue
            if source not in known:
                problems.append(f"{node['id']}.{name} points at unknown node {source}")
            elif port not in ports[source]:
                problems.append(
                    f"{node['id']}.{name} wants {source}.{port}, but {source} emits "
                    f"{ports[source] or ['nothing']}")
        if node["type"] == "subworkflow":
            target = library.get(node["workflow"])
            if target is None:
                problems.append(f"{node['id']} calls missing workflow {node['workflow']!r}")
            else:
                missing = set(target["inputs"]) - set(node.get("inputs", {}))
                for key in sorted(missing):
                    problems.append(f"{node['id']} calls {target['id']} without {key!r}")

    for node in spec["nodes"]:
        check(node, in_batch=False)
        for inner in node.get("body", []):
            check(inner, in_batch=True)
        for name, ref in node.get("collect", {}).items():
            source, _, port = ref.partition(".")
            if port not in ports.get(source, []):
                problems.append(f"{node['id']} collects {ref}, which is not emitted")
    for edge in spec["edges"]:
        for side in ("from", "to"):
            if edge[side] not in ports:
                problems.append(f"edge {edge['from']}->{edge['to']} has unknown {side}")
    return problems


def topological_order(spec):
    """Order the top-level nodes so every node runs after its predecessors.

    Returns (order, cycle_nodes). A non-empty second element means the graph
    cannot run at all: the remaining nodes are each waiting on another one.
    """
    incoming = {node["id"]: 0 for node in spec["nodes"]}
    outgoing = {node["id"]: [] for node in spec["nodes"]}
    for edge in spec["edges"]:
        outgoing[edge["from"]].append(edge["to"])
        incoming[edge["to"]] += 1

    ready = sorted(n for n, c in incoming.items() if c == 0)
    order = []
    while ready:
        current = ready.pop(0)
        order.append(current)
        for target in outgoing[current]:
            incoming[target] -= 1
            if incoming[target] == 0:
                ready.append(target)
        ready.sort()
    return order, sorted(set(incoming) - set(order))


# --- Execution -------------------------------------------------------------

def run_node(node, ctx, item=None, trace=None):
    """Execute one node and return the dictionary of ports it produces."""
    kind = node["type"]
    args = {k: resolve(v, ctx, item) for k, v in node.get("inputs", {}).items()}

    if kind == "start":
        return ctx["100001"]
    if kind == "end":
        return args
    if kind == "plugin":
        return PLUGINS[node["plugin"]](**args)
    if kind == "code":
        return CODE_FNS[node["fn"]](**args)
    if kind == "text":
        return {"text": node["template"].format(**args)}
    if kind == "model":
        if item is not None:
            args = dict(item, **args)
        return model_node(node["title"], args)
    if kind == "selector":
        for case in node["cases"]:
            if ctx[case["when"].split(".")[0]][case["when"].split(".")[1]] == case["equals"]:
                return {"branch": case["then"]}
        return {"branch": node["otherwise"]}
    if kind == "batch":
        return run_batch(node, ctx, trace)
    raise KeyError(f"unknown node type {kind!r}")


def run_batch(node, ctx, trace=None):
    """Run the body once per element, each iteration starting from a clean slate.

    Every iteration gets its own context. Nothing an iteration writes is visible
    to the next one, which is what makes the elements safe to process in any
    order - and also why anything that must hold across them has to be written
    into the data before the batch splits it up.
    """
    collected = {name: [] for name in node["collect"]}
    for element in resolve(node["over"], ctx):
        local = dict(ctx)
        for inner in node["body"]:
            local[inner["id"]] = run_node(inner, local, item=element, trace=trace)
        for name, ref in node["collect"].items():
            collected[name].append(resolve(ref, local))
    return collected


def run_workflow(spec_id, library, inputs, depth=0, trace=None):
    """Execute one workflow definition and return the ports its end node reads.

    depth counts nested sub-workflow calls. A platform needs this guard because
    a definition may call another one that calls back into it; the canvas shows
    two tidy boxes either way.
    """
    if depth > MAX_CALL_DEPTH:
        raise RecursionError(f"call depth {depth} exceeded at {spec_id!r}")
    spec = library[spec_id]
    order, cycle = topological_order(spec)
    if cycle:
        raise ValueError(f"{spec_id} cannot run: {cycle} form a cycle")

    nodes = {node["id"]: node for node in spec["nodes"]}
    ctx = {"100001": dict(inputs)}
    result = {}
    for node_id in order:
        node = nodes[node_id]
        if node["type"] == "start":
            continue
        if node["type"] == "subworkflow":
            args = {k: resolve(v, ctx) for k, v in node["inputs"].items()}
            ctx[node_id] = run_workflow(node["workflow"], library, args, depth + 1, trace)
        else:
            ctx[node_id] = run_node(node, ctx, trace=trace)
        if node["type"] == "end":
            result = ctx[node_id]
        if trace is not None:
            trace.append((depth, spec_id, node_id, node["title"], ctx[node_id]))
    return result


# --- Code node reshaping ---------------------------------------------------

def split_scenes_excluding(text):
    """Split on scene headings with a character class, which is the wrong tool.

    '[^Scene]' is not 'anything but the word Scene'. It is 'anything but the
    letters S, c, e and n', so the capture stops at the first of those letters
    in the body text.
    """
    return [m.group(1).strip() for m in re.finditer(r"Scene\d+:([^Scene]+)", text)]


def split_scenes_by_separator(text):
    """Split on the heading itself, so the body text cannot terminate a match."""
    parts = re.split(r"Scene\d+:", text)
    return [part.strip() for part in parts if part.strip()]


# --- Reporting -------------------------------------------------------------

def load_library():
    """Read every workflow definition in the data directory."""
    library = {}
    for path in sorted(SPEC_DIR.glob("*.json")):
        spec = json.loads(path.read_text(encoding="utf-8"))
        library[spec["id"]] = spec
    return library


def census(spec):
    """Count node types in one definition, body nodes included."""
    counts = {}
    for node in spec["nodes"]:
        counts[node["type"]] = counts.get(node["type"], 0) + 1
        for inner in node.get("body", []):
            counts[inner["type"]] = counts.get(inner["type"], 0) + 1
    return counts


def main():
    library = load_library()

    print("--- 1. Three definitions, and the node types they are built from ---")
    for spec_id, spec in library.items():
        counts = census(spec)
        total = sum(counts.values())
        print(f"  {spec['name']:<16} {total:>2} nodes  {len(spec['edges']):>2} edges  "
              f"{', '.join(f'{k}x{v}' for k, v in sorted(counts.items()))}")
    print(f"  every node type above resolves to a handler: "
          f"{len(PLUGINS)} plugins, {len(CODE_FNS)} code functions, 1 model handler")

    print("\n--- 2. Every port reference, checked before anything runs ---")
    for spec_id, spec in library.items():
        problems = validate(spec, library)
        print(f"  {spec['name']:<16} {len(problems)} unsatisfied reference(s)")
    broken = deepcopy(library["market_sentiment"])
    for node in broken["nodes"]:
        if node["id"] == "123474":
            node["inputs"]["values"] = "136482.summary"
    print("  edit one reference from 136482.digest to 136482.summary:")
    for problem in validate(broken, library):
        print(f"    {problem}")
    print("  the canvas draws that edge exactly the same either way")

    print("\n--- 3. Execution order, and a graph that has none ---")
    order, cycle = topological_order(library["market_sentiment"])
    titles = {n["id"]: n["title"] for n in library["market_sentiment"]["nodes"]}
    print("  " + " -> ".join(titles[n] for n in order))
    looped = deepcopy(library["market_sentiment"])
    looped["edges"].append({"from": "900001", "to": "107368"})
    order2, cycle2 = topological_order(looped)
    print(f"  add one edge End -> FetchNews: {len(order2)} nodes can start, "
          f"{len(cycle2)} wait forever {cycle2}")

    print("\n--- 4. The main workflow, node by node ---")
    trace = []
    result = run_workflow("market_sentiment", library,
                          {"topic": "brokerage", "today": "2026-05-04"}, trace=trace)
    for depth, spec_id, node_id, title, output in trace:
        shape = ", ".join(
            f"{k}={len(v)} item(s)" if isinstance(v, list) else f"{k}={str(v)[:38]!r}"
            for k, v in output.items())
        print(f"  {'  ' * depth}{title:<20} {shape}")

    print("\n--- 5. What the code nodes are for ---")
    print("  none of them holds business logic; each one reshapes data for the next node:")
    for name, fn in CODE_FNS.items():
        print(f"    {name:<20} {fn.__doc__.splitlines()[0]}")
    bad = split_scenes_excluding(SCENE_TEXT)
    good = split_scenes_by_separator(SCENE_TEXT)
    print(f"  splitting 3 scenes with a [^Scene] character class: {len(bad)} part(s)")
    for part in bad:
        print(f"    {part!r}")
    print(f"  splitting the same text on the heading itself: {len(good)} part(s)")
    for part in good:
        print(f"    {part[:52]!r}")

    print("\n--- 6. A batch body, run isolated and run carrying state ---")
    news = plugin_news_feed("brokerage")["items"]
    isolated = []
    for element in news:
        local = {"100001": {"today": "2026-05-04"}}
        local["130992"] = code_same_calendar_day("2026-05-04", element["published"])
        isolated.append(local["130992"]["same_day"])
    carried, seen = [], 0
    for element in news:
        seen += code_same_calendar_day("2026-05-04", element["published"])["same_day"]
        carried.append(seen)
    print(f"  isolated iterations: {isolated}   (any order gives this)")
    print(f"  carried across them: {carried}   (only this order gives this)")
    print("  a running total is the second kind, so it cannot live inside a batch;")
    print("  anything that has to hold across elements is written in before the split")

    print("\n--- 7. The branch the selector wrote ---")
    batch_output = next(o for _, _, nid, _, o in trace if nid == "136482")
    for element, branch in zip(news, batch_output["branch"]):
        print(f"  {branch:<5} {element['published']}  {element['title'][:44]}")
    kept = code_keep_marked(batch_output["digest"], batch_output["branch"])["kept"]
    print(f"  {len(kept)} of {len(news)} digests survive the branch")

    print("\n--- 8. The two sub-workflow calls ---")
    depths = sorted({(d, s) for d, s, _, _, _ in trace})
    for depth, spec_id in depths:
        print(f"  depth {depth}: {library[spec_id]['name']}")
    print(f"  guard stops at depth {MAX_CALL_DEPTH}; a definition that calls itself "
          f"raises instead of hanging")
    print(f"\n  hotwords: {result['hotwords']}")
    print("  report:")
    for line in result["report"].splitlines():
        print(f"    {line}")


if __name__ == "__main__":
    main()
