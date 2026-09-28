"""Discover another agent from its agent card and delegate a task to it, A2A style.

A2A (Agent2Agent) is a protocol for one agent to find and use another. The
provider publishes an agent card at a well-known URL. The caller reads the
card to learn where to send a task, what inputs it takes and how to
authenticate. Here both sides run locally: a FastAPI provider that knows which
rooms are free, and a caller that decides whether a workshop goes ahead. The
card and the task format are simplified. A2A's own card lists skills and
security schemes, and its tasks go through the SendMessage operation.

The run prints six parts:
    1. The capability card. The endpoint, required inputs and auth scheme
       the provider publishes.
    2. Starting the provider. It runs in a background thread and is ready
       once the card URL answers.
    3. Discovering it. The caller fetches the card and reads the endpoint
       and the auth scheme from it.
    4. Delegating three planning decisions. The provider returns the rooms
       that fit, and the caller confirms the smallest one or cancels.
    5. A task the card's schema forbids. 500 attendees against the card's
       maximum of 60, rejected with 400.
    6. A task without a bearer token. Rejected with 401.
"""

import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from json import dumps, loads

# Print UTF-8 even when the output is piped or redirected on Windows.
sys.stdout.reconfigure(encoding="utf-8")

HOST = "127.0.0.1"
PORT = 8931
PROVIDER_URL = f"http://{HOST}:{PORT}"
CARD_PATH = "/.well-known/agent-card.json"

# 1. The capability card. A caller that has never seen this code learns the
# endpoint, the accepted inputs and the authentication scheme from it.
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

# The provider's private data. The caller only sees the rooms a task returns.
ROOMS = {
    "2026-04-14": [{"room": "Cedar", "seats": 12}, {"room": "Aspen", "seats": 40}],
    "2026-04-15": [{"room": "Cedar", "seats": 12}],
    "2026-04-16": [],
}


def build_provider():
    """Return a FastAPI application that serves the card and handles tasks."""
    from fastapi import FastAPI, Header, HTTPException
    from pydantic import BaseModel

    app = FastAPI()

    class TaskRequest(BaseModel):
        task_id: str
        params: dict

    @app.get(CARD_PATH)
    async def get_card() -> dict:
        return AGENT_CARD

    @app.post(AGENT_CARD["endpoints"]["task_submit"])
    async def handle_task(request: TaskRequest, authorization: str | None = Header(default=None)) -> dict:
        # The card declares bearer auth, and this check enforces it.
        if authorization != "Bearer local-demo-token":
            raise HTTPException(status_code=401, detail="missing or invalid bearer token")

        date = request.params.get("date")
        attendees = request.params.get("attendees")

        limits = AGENT_CARD["input_schema"]["properties"]["attendees"]
        if not isinstance(attendees, int) or not limits["minimum"] <= attendees <= limits["maximum"]:
            raise HTTPException(
                status_code=400,
                detail=f"attendees must be an integer between {limits['minimum']} and {limits['maximum']}",
            )
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
    """Step 2. Run the provider in a background thread and poll the card until it
    answers, so no fixed sleep has to guess the start-up time."""
    import uvicorn

    config = uvicorn.Config(build_provider(), host=HOST, port=PORT, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        try:
            urllib.request.urlopen(f"{PROVIDER_URL}{CARD_PATH}", timeout=1).read()
            return thread
        except (urllib.error.URLError, ConnectionError):
            time.sleep(0.2)
    raise RuntimeError("the provider did not start within 15 seconds")


def discover(base_url: str) -> dict:
    """Step 3. Fetch the card and return it as the caller's only knowledge."""
    with urllib.request.urlopen(f"{base_url}{CARD_PATH}", timeout=5) as response:
        return loads(response.read())


def submit_task(base_url: str, card: dict, params: dict, token: str | None = "local-demo-token") -> tuple[int, dict]:
    """Post a task to the path the card names and return (status, reply).
    token=None sends no Authorization header."""
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
    """Step 4. Ask the provider which rooms fit, then decide here: the smallest
    room that fits, or cancel. The provider only reports rooms."""
    status, reply = submit_task(base_url, card, {"date": date, "attendees": attendees})
    if status != 200:
        print(f"  {date} for {attendees}: request rejected ({reply.get('detail')})")
        return

    rooms = reply["artifact"]["rooms"]
    returned = ", ".join(f"{room['room']} ({room['seats']})" for room in rooms) or "none"
    if rooms:
        best = min(rooms, key=lambda room: room["seats"])
        decision = f"confirmed in {best['room']}"
    else:
        decision = "cancelled, no room fits"
    print(f"  {date} for {attendees}: provider returned {returned} -> {decision}")


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
    print(f"  discovered {card['name']} v{card['version']} at {CARD_PATH}")
    print(f"  learned endpoint: {card['endpoints']['task_submit']}")
    print(f"  learned auth:     {card['authentication']['methods']}")

    print("\n--- 4. Delegating three planning decisions ---")
    plan_workshop(PROVIDER_URL, card, "2026-04-14", 10)
    plan_workshop(PROVIDER_URL, card, "2026-04-15", 30)
    plan_workshop(PROVIDER_URL, card, "2026-04-16", 8)

    print("\n--- 5. A task the card's schema forbids ---")
    status, reply = submit_task(PROVIDER_URL, card, {"date": "2026-04-14", "attendees": 500})
    print(f"  status {status}: {reply.get('detail')}")
    print(f"  the card said attendees max is {card['input_schema']['properties']['attendees']['maximum']}")

    print("\n--- 6. A task without a bearer token ---")
    status, reply = submit_task(PROVIDER_URL, card, {"date": "2026-04-14", "attendees": 5}, token=None)
    print(f"  status {status}: {reply.get('detail')}")
    print(f"  the card declared authentication methods: {card['authentication']['methods']}")


if __name__ == "__main__":
    main()
