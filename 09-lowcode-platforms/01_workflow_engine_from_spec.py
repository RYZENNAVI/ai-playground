"""This script runs a drag-and-drop workflow from its exported JSON definition, without
the visual editor, by resolving the port references between its nodes and executing them
in topological order.

The run shows that such a workflow is a typed graph plus a node registry:
    1. Load three workflow definitions and print the node types they are built from.
    2. Check every port reference before anything runs. Then break one reference under
       'inputs', and break the batch's 'over' and the selector's 'when', which sit
       outside 'inputs'.
    3. Order the nodes topologically, then add one edge that closes a cycle.
    4. Execute the main workflow node by node, printing what each node produced.
    5. List what each code node does. Then split a scene text with a regular expression
       whose character class excludes four letters instead of one word. No workflow
       here uses that split.
    6. Run one node of the batch body over every ordering of the four articles, once
       isolated and once carrying a running total.
    7. Read the branch the selector wrote for each article in step 4, and what
       KeepMarked kept.
    8. Follow both sub-workflow calls, then run a workflow that calls itself and let the
       call-depth guard stop it.

Nothing here calls a model. Every model node is answered by a fixed local handler, so
each run prints the same thing; script 02 puts a real model behind the same node type.
A batch gives every iteration its own copy of the context, so nothing one element writes
reaches the next. Anything that must hold across elements has to be written into the
data before the batch splits it.
"""

import json
import re
import sys
from copy import deepcopy
from itertools import permutations
from pathlib import Path

# Print UTF-8 even when the output is piped or redirected on Windows.
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

# Three scene paragraphs under numbered headings, for the regex in step 5.
SCENE_TEXT = (
    "Scene1: A red kite rises over an empty green field at dawn.\n"
    "Scene2: The same field at noon, seen from beneath a bending tree.\n"
    "Scene3: Evening arrives and the kite is a dark speck against orange cloud.\n"
)


def plugin_news_feed(topic):
    """Return market articles for a topic. Stands in for a remote data source."""
    return {"items": [dict(row, topic=topic) for row in NEWS_FIXTURE]}


def plugin_review_feed(app_id):
    """Return store reviews for one application id."""
    return {"items": [dict(row, app_id=app_id) for row in REVIEW_FIXTURE]}


PLUGINS = {"news_feed": plugin_news_feed, "review_feed": plugin_review_feed}


def code_same_calendar_day(today, published):
    """Return 1 when a published timestamp falls on the reference date.

    Cuts '2026-05-04 09:05' to its date part first, so it can equal '2026-05-04'.
    """
    day = published.split(" ")[0]
    return {"same_day": 1 if day == today else 0}


def code_keep_marked(values, marks):
    """Keep the values whose parallel mark is the string 'keep'."""
    return {"kept": [v for v, m in zip(values, marks) if m == "keep"]}


def code_split_by_verdict(verdicts, digests):
    """Split digests into a positive and a negative list by the verdict beside each one.

    Any other word, 'neutral' included, lands in neither list (script 02 counts them).
    """
    positive = [d for v, d in zip(verdicts, digests) if v == "positive"]
    negative = [d for v, d in zip(verdicts, digests) if v == "negative"]
    return {"positive": positive, "negative": negative}


CODE_FNS = {
    "same_calendar_day": code_same_calendar_day,
    "keep_marked": code_keep_marked,
    "split_by_verdict": code_split_by_verdict,
}


# Hotwords only counts words of four letters or more, so shorter stopwords never reach
# this check.
STOPWORDS = {"have", "next"}


def model_node(title, args):
    """Answer a model node with a fixed local handler instead of a model."""
    if title == "Digest":
        # A bare split(".") would cut "gained 1.2 percent" after the 1. A sentence
        # end is a period followed by whitespace or the end of the text; a decimal
        # point is followed by a digit.
        first = re.split(r"\.(?=\s|$)", args["body"].strip())[0].strip()
        return {"text": f"{args['title']}: {first}."}
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


def declared_ports(spec):
    """Map every node id in a definition to the ports it says it emits."""
    ports = {}
    for node in spec["nodes"]:
        ports[node["id"]] = list(node.get("ports", []))
        for inner in node.get("body", []):
            ports[inner["id"]] = list(inner.get("ports", []))
    return ports


def resolve(ref, ctx, item=None):
    """Turn one reference into a value: 'literal:x', 'item.field' or 'node.port'.

    'item' is the current batch element; 'node.port' is a port already produced.
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
    """Return every reference in one definition that no declared port satisfies.

    Without this check, such a reference fails only when the run reaches its node.
    """
    ports = declared_ports(spec)
    known = set(ports) | {"item"}
    problems = []

    def check_ref(node_id, label, ref, in_batch):
        if ref.startswith("literal:"):
            return
        source, _, port = ref.partition(".")
        if source == "item":
            if not in_batch:
                problems.append(f"{node_id}.{label} reads 'item' outside a batch")
            return
        if source not in known:
            problems.append(f"{node_id}.{label} points at unknown node {source}")
        elif port not in ports[source]:
            problems.append(
                f"{node_id}.{label} wants {source}.{port}, but {source} emits "
                f"{ports[source] or ['nothing']}")

    def check(node, in_batch):
        for name, ref in node.get("inputs", {}).items():
            check_ref(node["id"], name, ref, in_batch)
        # A batch says what it walks and a selector says what it tests, and neither
        # of those sits under 'inputs'. They are references all the same, so a typo
        # in one fails at run time unless it is checked here too.
        if node["type"] == "batch":
            check_ref(node["id"], "over", node["over"], in_batch)
        if node["type"] == "selector":
            for pos, case in enumerate(node["cases"]):
                check_ref(node["id"], f"cases[{pos}].when", case["when"], in_batch)
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
    """Order the top-level nodes so each runs after its predecessors (Kahn's algorithm).

    Returns (order, stuck); stuck lists the nodes that never become ready.
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
    """Run the body once per element, each iteration on its own copy of the context."""
    collected = {name: [] for name in node["collect"]}
    for element in resolve(node["over"], ctx):
        local = dict(ctx)
        for inner in node["body"]:
            local[inner["id"]] = run_node(inner, local, item=element, trace=trace)
        for name, ref in node["collect"].items():
            collected[name].append(resolve(ref, local))
    return collected


def run_workflow(spec_id, library, inputs, depth=0, trace=None):
    """Execute one workflow definition and return the values its end node reads.

    depth counts nested sub-workflow calls; past MAX_CALL_DEPTH the call raises.
    """
    if depth > MAX_CALL_DEPTH:
        raise RecursionError(f"call depth {depth} exceeded at {spec_id!r}")
    spec = library[spec_id]
    order, cycle = topological_order(spec)
    if cycle:
        raise ValueError(f"{spec_id} cannot run: {cycle} are stuck on a cycle")

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


def split_scenes_excluding(text):
    """Split on scene headings with a character class, which is the wrong tool.

    '[^Scene]' excludes the letters S, c, e and n, not the word, so captures stop early.
    """
    return [m.group(1).strip() for m in re.finditer(r"Scene\d+:([^Scene]+)", text)]


def split_scenes_by_separator(text):
    """Split on the heading itself, so the body text cannot terminate a match."""
    parts = re.split(r"Scene\d+:", text)
    return [part.strip() for part in parts if part.strip()]


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
    # 1. Load the definitions and count node types
    library = load_library()

    print("--- 1. Three definitions, and the node types they are built from ---")
    for spec_id, spec in library.items():
        counts = census(spec)
        total = sum(counts.values())
        print(f"  {spec['name']:<16} {total:>2} nodes  {len(spec['edges']):>2} edges  "
              f"{', '.join(f'{k}x{v}' for k, v in sorted(counts.items()))}")
    print(f"  handlers registered: {len(PLUGINS)} plugins, {len(CODE_FNS)} code functions,"
          f" 1 model handler (step 4 runs all three definitions on them)")

    # 2. Check every port reference

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
    bent = deepcopy(library["market_sentiment"])
    for node in bent["nodes"]:
        if node["id"] == "136482":
            node["over"] = "107368.rows"
            for inner in node["body"]:
                if inner["type"] == "selector":
                    inner["cases"][0]["when"] = "130992.same_date"
    print("  edit the batch's 'over' and the selector's 'when' instead:")
    for problem in validate(bent, library):
        print(f"    {problem}")
    print("  neither sits under 'inputs', so a check of 'inputs' alone misses both")
    print("  run_workflow never calls validate: this is the editor's gate, not the")
    print("  runtime's, so a definition handed straight to the engine skips it")

    # 3. Order the nodes, then close a cycle

    print("\n--- 3. Execution order, and a graph that has none ---")
    order, cycle = topological_order(library["market_sentiment"])
    titles = {n["id"]: n["title"] for n in library["market_sentiment"]["nodes"]}
    print("  " + " -> ".join(titles[n] for n in order))
    looped = deepcopy(library["market_sentiment"])
    looped["edges"].append({"from": "900001", "to": "107368"})
    order2, cycle2 = topological_order(looped)
    print(f"  add one edge End -> FetchNews: {len(order2)} of "
          f"{len(looped['nodes'])} nodes can run, "
          f"{len(cycle2)} never become ready {cycle2}")

    # 4. Run the main workflow

    print("\n--- 4. The main workflow, node by node ---")
    trace = []
    result = run_workflow("market_sentiment", library,
                          {"topic": "brokerage", "today": "2026-05-04"}, trace=trace)
    for depth, spec_id, node_id, title, output in trace:
        shape = ", ".join(
            f"{k}={len(v)} item(s)" if isinstance(v, list) else f"{k}={str(v)[:38]!r}"
            for k, v in output.items())
        print(f"  {'  ' * depth}{title:<20} {shape}")

    # 5. Code nodes, and a character class that is the wrong tool

    print("\n--- 5. What the code nodes are for ---")
    print("  each one reshapes data for the next node rather than deciding anything a")
    print("  model would decide; the rule each applies is a fixed comparison:")
    for name, fn in CODE_FNS.items():
        print(f"    {name:<20} {fn.__doc__.splitlines()[0]}")
    bad = split_scenes_excluding(SCENE_TEXT)
    good = split_scenes_by_separator(SCENE_TEXT)
    bad_chars = sum(len(part) for part in bad)
    good_chars = sum(len(part) for part in good)
    print(f"  splitting 3 scenes with a [^Scene] character class: {len(bad)} part(s), "
          f"{bad_chars} characters kept")
    for part in bad:
        print(f"    {part!r}")
    print(f"  splitting the same text on the heading itself: {len(good)} part(s), "
          f"{good_chars} characters kept")
    for part in good:
        print(f"    {part[:52]!r}")
    print(f"  the part count is the same either way, so counting parts says nothing;"
          f" the character class kept {bad_chars} of {good_chars} characters, each"
          f" capture stopping at the first S, c, e or n in the body")

    # 6. One batch body node, isolated and carrying state
    print("\n--- 6. A batch body, run isolated and run carrying state ---")
    news = plugin_news_feed("brokerage")["items"]
    isolated = [code_same_calendar_day("2026-05-04", element["published"])["same_day"]
                for element in news]
    carried, seen = [], 0
    for element in news:
        seen += code_same_calendar_day("2026-05-04", element["published"])["same_day"]
        carried.append(seen)
    marks_by_article = [set() for _ in news]
    totals_by_article = [set() for _ in news]
    totals_by_order = []
    for perm in permutations(range(len(news))):
        running, totals = 0, []
        for i in perm:
            mark = code_same_calendar_day("2026-05-04", news[i]["published"])["same_day"]
            running += mark
            marks_by_article[i].add(mark)
            totals_by_article[i].add(running)
            totals.append(running)
        totals_by_order.append(tuple(totals))
    orders = len(totals_by_order)
    matching = totals_by_order.count(tuple(carried))
    print(f"  isolated iterations: {isolated}")
    print(f"  carried across them: {carried}")
    print(f"  over all {orders} orderings of the same {len(news)} articles, what each"
          f" article receives:")
    for element, marks, totals in zip(news, marks_by_article, totals_by_article):
        print(f"    {element['title'][:38]:<38} isolated {sorted(marks)}"
              f"  carried {sorted(totals)}")
    print(f"  {matching} of the {orders} orderings give the carried totals {carried}")
    print("  an isolated mark reads only its own article; a running total depends on the")
    print("  order, so it cannot live inside a batch. Anything that has to hold across")
    print("  elements is written in before the split")

    # 7. The branch the selector wrote
    print("\n--- 7. The branch the selector wrote ---")
    batch_output = next(o for _, _, nid, _, o in trace if nid == "136482")
    kept = next(o for _, _, nid, _, o in trace if nid == "123474")["kept"]
    for element, branch in zip(news, batch_output["branch"]):
        print(f"  {branch:<5} {element['published']}  {element['title'][:44]}")
    print(f"  {len(kept)} of {len(batch_output['digest'])} digests survive the branch"
          f" (KeepMarked's output in step 4)")

    # 8. Sub-workflow calls and the call-depth guard

    print("\n--- 8. The two sub-workflow calls ---")
    depths = sorted({(d, s) for d, s, _, _, _ in trace})
    for depth, spec_id in depths:
        print(f"  depth {depth}: {library[spec_id]['name']}")
    self_call = {
        "id": "self_call", "name": "SelfCall", "inputs": ["topic"],
        "nodes": [
            {"id": "100001", "type": "start", "title": "Start", "ports": ["topic"]},
            {"id": "200001", "type": "subworkflow", "title": "CallAgain",
             "workflow": "self_call", "inputs": {"topic": "100001.topic"},
             "ports": ["topic"]},
            {"id": "900001", "type": "end", "title": "End",
             "inputs": {"topic": "200001.topic"}},
        ],
        "edges": [{"from": "100001", "to": "200001"}, {"from": "200001", "to": "900001"}],
    }
    try:
        run_workflow("self_call", dict(library, self_call=self_call), {"topic": "x"})
    except RecursionError as err:
        print(f"  a definition that calls itself: RecursionError: {err}")
    else:
        print("  a definition that calls itself: finished without an error")
    print(f"  depths 0 to {MAX_CALL_DEPTH} run, and the call into depth"
          f" {MAX_CALL_DEPTH + 1} raises; without the guard,")
    print(f"  Python's recursion limit ({sys.getrecursionlimit()} here) raises instead,"
          f" with a message that names no workflow")
    print(f"\n  hotwords: {result['hotwords']}")
    print("  report:")
    for line in result["report"].splitlines():
        print(f"    {line}")


if __name__ == "__main__":
    main()
