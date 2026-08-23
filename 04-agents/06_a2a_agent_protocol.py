"""Let one agent discover another at runtime and delegate a task to it.

Demonstrates agent-to-agent delegation without a shared codebase:
    1. Publish a capability card at a well-known path on the provider.
    2. Start the provider and wait for it to accept connections.
    3. Discover the provider by reading its card, not by hardcoding its routes.
    4. Build a task from the schema the card declares and submit it.
    5. Act on the returned artifact to make a decision the caller owns.
    6. Submit a task the card's schema forbids, and read the rejection.
    7. Submit a task without the bearer token the card declared, and read the rejection.

Module 04: Agents - Agent-to-Agent Protocol.
"""

import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from json import dumps, loads

sys.stdout.reconfigure(encoding="utf-8")

HOST = "127.0.0.1"
PORT = 8931
PROVIDER_URL = f"http://{HOST}:{PORT}"

# 1. The capability card. A caller that has never seen this code should be able
# to read this document and learn the endpoint, the accepted inputs and the
# authentication scheme, which is the entire premise of runtime discovery.
AGENT_CARD = {
    "name": "RoomAvailabilityAgent",
    "version": "1.0",
    "description": "Reports which rooms are free on a given date and for how many people.",
    "endpoints": {"task_submit": "/api/tasks/availability"},
    "input_schema": {
        "type": "object",
        "properties": {
            "date": {"type": "string", "format": "date"},
            "attendees": {"type": "integer", "minimum": 1, "maximum": 60},
        },
        "required": ["date", "attendees"],
    },
    "authentication": {"methods": ["bearer"]},
}

# The provider's private data. The caller never sees this table, only the
# artifact derived from it - which is what makes this delegation rather than a
# shared library call.
ROOMS = {
    "2026-04-14": [{"room": "Cedar", "seats": 12}, {"room": "Aspen", "seats": 40}],
    "2026-04-15": [{"room": "Cedar", "seats": 12}],
    "2026-04-16": [],
}


def build_provider():
    """Return a FastAPI application that serves the card and handles tasks.

    The import sits inside the function so the module still loads for anyone who
    only wants to read the client half, and so a missing web framework produces
    a clear error at the point of use instead of at import time.
    """
    from fastapi import FastAPI, Header, HTTPException
    from pydantic import BaseModel

    app = FastAPI()

    class TaskRequest(BaseModel):
        task_id: str
        params: dict

    @app.get("/.well-known/agent.json")
    async def get_card() -> dict:
        return AGENT_CARD

    @app.post(AGENT_CARD["endpoints"]["task_submit"])
    async def handle_task(request: TaskRequest, authorization: str | None = Header(default=None)) -> dict:
        # The card declares bearer auth, so this is what makes that declaration
        # true rather than decorative: a caller that read the card and skipped
        # the token gets rejected here, not waved through.
        if authorization != "Bearer local-demo-token":
            raise HTTPException(status_code=401, detail="missing or invalid bearer token")

        date = request.params.get("date")
        attendees = request.params.get("attendees")

        if not isinstance(attendees, int) or not 1 <= attendees <= 60:
            raise HTTPException(status_code=400, detail="attendees must be an integer between 1 and 60")
        if date not in ROOMS:
            raise HTTPException(status_code=400, detail=f"no availability data for {date}")

        fitting = [room for room in ROOMS[date] if room["seats"] >= attendees]
        return {
            "task_id": request.task_id,
            "status": "completed",
            "artifact": {"date": date, "attendees": attendees, "rooms": fitting},
        }

    return app


def start_provider() -> threading.Thread:
    """Step 2. Run the provider in a background thread and wait until it answers.

    Polling the card endpoint is the readiness check rather than a fixed sleep,
    because a sleep long enough to be safe on a cold start is wasted on every
    later run, and a short one turns into an intermittent failure.
    """
    import uvicorn

    config = uvicorn.Config(build_provider(), host=HOST, port=PORT, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        try:
            urllib.request.urlopen(f"{PROVIDER_URL}/.well-known/agent.json", timeout=1).read()
            return thread
        except (urllib.error.URLError, ConnectionError):
            time.sleep(0.2)
    raise RuntimeError("the provider did not start within 15 seconds")


def discover(base_url: str) -> dict:
    """Step 3. Fetch the card and return it as the caller's only knowledge."""
    with urllib.request.urlopen(f"{base_url}/.well-known/agent.json", timeout=5) as response:
        return loads(response.read())


def submit_task(base_url: str, card: dict, params: dict, token: str | None = "local-demo-token") -> tuple[int, dict]:
    """Step 4. Post a task to whatever path the card named, and return the reply.

    The endpoint comes out of the card rather than a constant in this file. That
    is the difference that matters: the provider can move its route, and the
    caller follows without being redeployed, because it re-reads the card.

    `token` defaults to the value that satisfies the bearer scheme the card
    declared. Step 7 passes None to send the request with no Authorization
    header at all, to show that scheme is enforced rather than descriptive.
    """
    payload = dumps({"task_id": str(uuid.uuid4()), "params": params}).encode()
    headers = {"Content-Type": "application/json"}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(f"{base_url}{card['endpoints']['task_submit']}", data=payload, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, loads(error.read())


def plan_workshop(base_url: str, card: dict, date: str, attendees: int) -> None:
    """Step 5. Turn the returned artifact into a decision this agent owns.

    The provider reports facts about rooms and stops there. Whether a workshop
    goes ahead is not its call, and keeping that split is what lets either side
    change its rules without renegotiating with the other.
    """
    status, reply = submit_task(base_url, card, {"date": date, "attendees": attendees})
    if status != 200:
        print(f"  {date} for {attendees}: request rejected ({reply.get('detail')})")
        return

    rooms = reply["artifact"]["rooms"]
    if rooms:
        best = min(rooms, key=lambda room: room["seats"])
        print(f"  {date} for {attendees}: confirmed in {best['room']} ({best['seats']} seats)")
    else:
        print(f"  {date} for {attendees}: cancelled, no room fits")


def main() -> None:
    print("--- 1. The capability card this provider publishes ---")
    print(f"  name:     {AGENT_CARD['name']}")
    print(f"  endpoint: {AGENT_CARD['endpoints']['task_submit']}")
    print(f"  requires: {', '.join(AGENT_CARD['input_schema']['required'])}")

    print("\n--- 2. Starting the provider ---")
    start_provider()
    print(f"  answering on {PROVIDER_URL}")

    print("\n--- 3. Discovering it from the caller's side ---")
    card = discover(PROVIDER_URL)
    print(f"  discovered {card['name']} v{card['version']}")
    print(f"  learned endpoint: {card['endpoints']['task_submit']}")
    print(f"  learned auth:     {card['authentication']['methods']}")

    print("\n--- 4-5. Delegating three planning decisions ---")
    plan_workshop(PROVIDER_URL, card, "2026-04-14", 30)
    plan_workshop(PROVIDER_URL, card, "2026-04-15", 30)
    plan_workshop(PROVIDER_URL, card, "2026-04-16", 8)

    print("\n--- 6. A task the card's schema forbids ---")
    status, reply = submit_task(PROVIDER_URL, card, {"date": "2026-04-14", "attendees": 500})
    print(f"  status {status}: {reply.get('detail')}")
    print(f"  the card said attendees max is {card['input_schema']['properties']['attendees']['maximum']}")

    print("\n--- 7. A task without the bearer token the card declared ---")
    status, reply = submit_task(PROVIDER_URL, card, {"date": "2026-04-14", "attendees": 5}, token=None)
    print(f"  status {status}: {reply.get('detail')}")
    print(f"  the card declared authentication methods: {card['authentication']['methods']}")


if __name__ == "__main__":
    main()
