"""Serve the protocol a workflow platform exposes, then call it three ways.

Demonstrates what a hosted workflow looks like from outside, and how a client
stops knowing which deployment it is talking to:
    1. Start a local server that answers the three endpoints such a platform exposes.
    2. Call the blocking endpoint and read the single response body.
    3. Call the streaming endpoint and read the events as they arrive.
    4. Print the request headers the way a debugging client does, and look at them.
    5. Send a request that forgets to carry the user's own words.
    6. Let a client probe five payload shapes until one stops failing.
    7. Break the connection and watch that probe rewrite the cause of the failure.

Module 09: Low-Code Platforms - Platform API Protocol.
"""

import json
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from textwrap import dedent

sys.stdout.reconfigure(encoding="utf-8")

HOST = "127.0.0.1"
API_KEY = "local-development-key"
SERVER_FILE = Path(__file__).resolve().parent / "data" / "mock_platform_server.py"

# The server module is written out at run time so the whole protocol lives in one
# file. It answers the three endpoints and speaks the event stream a workflow
# platform sends: one started event, one per node, one finished event.
SERVER_SOURCE = dedent('''
    """A minimal stand-in for a hosted workflow deployment, for local runs only."""

    import json
    import sys
    import time

    from fastapi import FastAPI, Header, HTTPException, Request
    from fastapi.responses import StreamingResponse

    API_KEY = "local-development-key"
    app = FastAPI()

    # The one input variable this deployment declares. A caller that sends some
    # other key sends a request the deployment has no way to read.
    INPUT_VARIABLE = "question"

    ANSWERS = {
        "why do price alerts arrive late":
            "Alerts read one-second quote buckets, so they trail the print.",
        "how do i restore a watchlist":
            "Support can restore the profile snapshot for thirty days.",
    }


    def answer_for(question):
        key = question.strip().lower().rstrip("?")
        return ANSWERS.get(key, f"No configured answer for {question!r}.")


    def check(authorization):
        if authorization != f"Bearer {API_KEY}":
            raise HTTPException(status_code=401, detail={"code": "invalid_api_key",
                                                         "message": "bad credential"})


    def stream_run(question):
        def event(payload):
            return f"data: {json.dumps(payload)}\\n\\n"

        yield event({"event": "workflow_started", "workflow_run_id": "run-0001",
                     "task_id": "task-0001"})
        for node in ("Start", "Retrieve", "Answer"):
            time.sleep(0.05)
            yield event({"event": "node_finished",
                         "data": {"title": node, "elapsed_time": 0.05}})
        yield event({"event": "workflow_finished",
                     "data": {"outputs": {"answer": answer_for(question)}}})


    @app.post("/v1/workflows/run")
    async def run_workflow(request: Request, authorization: str = Header(None)):
        check(authorization)
        body = await request.json()
        inputs = body.get("inputs") or {}
        if INPUT_VARIABLE not in inputs:
            raise HTTPException(
                status_code=400,
                detail={"code": "app_unavailable",
                        "message": f"input variable {INPUT_VARIABLE!r} is required"})
        question = inputs[INPUT_VARIABLE]
        if body.get("response_mode") == "streaming":
            return StreamingResponse(stream_run(question),
                                     media_type="text/event-stream")
        return {"workflow_run_id": "run-0001", "task_id": "task-0001",
                "data": {"status": "succeeded",
                         "outputs": {"answer": answer_for(question)}}}


    @app.post("/v1/chat-messages")
    async def chat_messages(request: Request, authorization: str = Header(None)):
        check(authorization)
        body = await request.json()
        if "query" not in body:
            raise HTTPException(status_code=400,
                                detail={"code": "not_chat_app",
                                        "message": "this deployment is not a chat app"})
        return {"conversation_id": "conv-0001", "message_id": "msg-0001",
                "answer": answer_for(body["query"])}


    @app.post("/v1/completion-messages")
    async def completion_messages(request: Request, authorization: str = Header(None)):
        check(authorization)
        body = await request.json()
        inputs = body.get("inputs") or {}
        if INPUT_VARIABLE not in inputs:
            raise HTTPException(
                status_code=400,
                detail={"code": "app_unavailable",
                        "message": f"input variable {INPUT_VARIABLE!r} is required"})
        return {"message_id": "msg-0002", "answer": answer_for(inputs[INPUT_VARIABLE])}


    if __name__ == "__main__":
        import uvicorn

        uvicorn.run(app, host="127.0.0.1", port=int(sys.argv[1]), log_level="error")
''').strip() + "\n"


def free_port():
    """Ask the operating system for a port nobody is using."""
    with socket.socket() as probe:
        probe.bind((HOST, 0))
        return probe.getsockname()[1]


def start_server(port):
    """Write the server module, start it, and wait until it answers."""
    SERVER_FILE.write_text(SERVER_SOURCE, encoding="utf-8")
    process = subprocess.Popen([sys.executable, str(SERVER_FILE), str(port)],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            with socket.create_connection((HOST, port), timeout=0.5):
                return process
        except OSError:
            time.sleep(0.2)
    process.terminate()
    raise RuntimeError("the local server did not come up")


class PlatformClient:
    """A client for the three endpoints, holding the key it authenticates with."""

    def __init__(self, base_url, api_key):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.headers = {"Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                        "Accept": "application/json"}

    def post(self, path, payload, stream=False, timeout=30):
        """Send one request and return (status, parsed body or open response)."""
        request = urllib.request.Request(
            f"{self.base_url}{path}", method="POST",
            data=json.dumps(payload).encode("utf-8"), headers=self.headers)
        try:
            response = urllib.request.urlopen(request, timeout=timeout)
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read().decode("utf-8"))
        except urllib.error.URLError as error:
            return None, {"transport_error": str(error.reason)}
        if stream:
            return response.status, response
        return response.status, json.loads(response.read().decode("utf-8"))

    def run_blocking(self, question):
        """Run the workflow and wait for the whole answer."""
        return self.post("/v1/workflows/run",
                         {"inputs": {"question": question},
                          "response_mode": "blocking", "user": "demo"})

    def run_streaming(self, question):
        """Run the workflow and read its events as they are produced.

        A line that does not parse is skipped rather than raised on, because a
        stream carries keep-alive lines and blank lines between the events.
        """
        status, response = self.post(
            "/v1/workflows/run",
            {"inputs": {"question": question}, "response_mode": "streaming",
             "user": "demo"}, stream=True)
        events = []
        if status != 200:
            return status, events
        for raw in response:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data: "):
                continue
            try:
                events.append(json.loads(line[6:]))
            except json.JSONDecodeError:
                continue
        return status, events

    def completion_dropping_input(self, question):
        """Call the completion endpoint with an empty inputs object.

        question is a parameter of this method and appears nowhere in the body
        it sends. The endpoint is reached and the key is accepted; the user's
        words are the one thing that does not travel.
        """
        return self.post("/v1/completion-messages",
                         {"inputs": {}, "response_mode": "blocking", "user": "demo"})

    def completion_probing(self, question, timeout=30):
        """Try five payload shapes until one of them stops failing.

        Every failure is read as the wrong shape, so a failure that has nothing
        to do with shape gets the same treatment: move on to the next one, and
        when the list is exhausted, report that the shapes are exhausted.
        """
        shapes = [{}, {"text": question}, {"query": question},
                  {"question": question}, {"prompt": question}]
        attempts = []
        for shape in shapes:
            status, body = self.post("/v1/completion-messages",
                                     {"inputs": shape, "response_mode": "blocking",
                                      "user": "demo"}, timeout=timeout)
            attempts.append((list(shape), status, body))
            if status == 200:
                return attempts, body
        return attempts, {"error": True,
                          "message": "every input format failed; check the "
                                     "application configuration and API key"}


def redacted(headers):
    """Return the headers with the credential replaced."""
    return {**headers, "Authorization": "Bearer ***"}


def describe(body):
    """Render an error body as its code, falling back to the whole body."""
    detail = body.get("detail", body) if isinstance(body, dict) else body
    if isinstance(detail, dict):
        return detail.get("code") or detail.get("transport_error") or str(detail)
    return str(detail)


def main():
    port = free_port()
    server = start_server(port)
    client = PlatformClient(f"http://{HOST}:{port}", API_KEY)
    question = "Why do price alerts arrive late?"
    try:
        print(f"--- 1. A local server on port {port} ---")
        print(f"  wrote {SERVER_FILE.name} and started it as a subprocess")
        print("  it answers /v1/workflows/run, /v1/chat-messages and "
              "/v1/completion-messages")
        print("  it declares exactly one input variable: 'question'")

        print("\n--- 2. The blocking call ---")
        status, body = client.run_blocking(question)
        print(f"  HTTP {status}  run {body['workflow_run_id']}  "
              f"status {body['data']['status']}")
        print(f"  answer: {body['data']['outputs']['answer']}")
        print("  one request, one response, and nothing observable in between")

        print("\n--- 3. The same run, streamed ---")
        started = time.time()
        status, events = client.run_streaming(question)
        for event in events:
            detail = event.get("data", {}).get("title") or event.get("workflow_run_id", "")
            print(f"  {event['event']:<18} {detail}")
        print(f"  {len(events)} events in {time.time() - started:.2f}s; the node "
              f"events are the only view of what ran")
        print(f"  answer: {events[-1]['data']['outputs']['answer']}")

        print("\n--- 4. The headers a client prints while debugging ---")
        print(f"  as written: {client.headers}")
        print(f"  redacted  : {redacted(client.headers)}")
        print("  the first form is one line in a debug print and one copy of the")
        print("  credential in every log this process ever writes")

        print("\n--- 5. A request that carries no question ---")
        status, body = client.completion_dropping_input(question)
        print(f"  HTTP {status}  {describe(body)}")
        print("  the endpoint is right and the key is right; the payload has an")
        print("  empty inputs object, so the deployment refuses for its own missing")
        print("  input variable rather than for anything the caller can see")

        print("\n--- 6. A client probing for the shape ---")
        attempts, result = client.completion_probing(question)
        for shape, status, body in attempts:
            note = "accepted" if status == 200 else describe(body)
            print(f"  inputs={str(shape):<14} HTTP {status}  {note}")
        print(f"  {len(attempts)} request(s) to arrive at a key the deployment names")
        print(f"  in its own configuration: answer -> {result.get('answer')}")

        print("\n--- 7. The same probe against a server that is gone ---")
        server.terminate()
        server.wait(timeout=10)
        attempts, result = client.completion_probing(question, timeout=2)
        for shape, status, body in attempts:
            print(f"  inputs={str(shape):<14} HTTP {status}  {describe(body)}")
        print(f"  reported to the caller: {result['message']!r}")
        print("  five transport failures in a row, and the message that comes back")
        print("  names the application configuration and the API key instead")
    finally:
        if server.poll() is None:
            server.terminate()
            server.wait(timeout=10)
        print(f"\n  server stopped, exit code {server.returncode}")


if __name__ == "__main__":
    main()
