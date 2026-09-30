"""This script serves the HTTP API of a hosted workflow platform from a local FastAPI
server, then calls it three ways: a blocking request, a stream of server-sent events
(SSE), and a client that probes payload shapes until one stops failing.

It shows what a hosted workflow looks like from outside, and how a client stops knowing
which deployment it is talking to. The run prints 7 parts:
    1. A local server answering the three endpoints.
    2. The blocking call.
    3. The same run, streamed.
    4. The headers a client prints while debugging.
    5. Two requests that both come back HTTP 400.
    6. A client probing for the shape.
    7. The same probe against a server that is gone.
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

# Print UTF-8 even when the output is piped or redirected on Windows.
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

    # What kind of application this deployment is. All three endpoints exist on
    # the platform; only the one matching this type answers for this key.
    APP_TYPE = "workflow"

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
        if APP_TYPE != "chat":
            raise HTTPException(
                status_code=400,
                detail={"code": "not_chat_app",
                        "message": f"this deployment is a {APP_TYPE}, not a chat app"})
        if "query" not in body:
            raise HTTPException(status_code=400,
                                detail={"code": "invalid_param",
                                        "message": "'query' is required"})
        return {"conversation_id": "conv-0001", "message_id": "msg-0001",
                "answer": answer_for(body["query"])}


    @app.post("/v1/completion-messages")
    async def completion_messages(request: Request, authorization: str = Header(None)):
        check(authorization)
        body = await request.json()
        if APP_TYPE != "completion":
            raise HTTPException(
                status_code=400,
                detail={"code": "not_completion_app",
                        "message":
                            f"this deployment is a {APP_TYPE}, not a completion app"})
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
        Only 'data: ' lines are kept; blank and keep-alive lines are skipped."""
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

    def run_dropping_input(self, question):
        """Send the user's question nowhere: the inputs object goes out empty."""
        return self.post("/v1/workflows/run",
                         {"inputs": {}, "response_mode": "blocking", "user": "demo"})

    def chat_on_a_workflow_deployment(self, question):
        """Send a body correct for the chat endpoint to a workflow deployment."""
        return self.post("/v1/chat-messages", {"query": question, "user": "demo"})

    def run_probing(self, question, timeout=30):
        """Try five payload shapes in turn, reading every failure as the wrong shape."""
        shapes = [{}, {"text": question}, {"query": question},
                  {"question": question}, {"prompt": question}]
        attempts = []
        for shape in shapes:
            status, body = self.post("/v1/workflows/run",
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
        # 1. A local server answering the three endpoints
        print(f"--- 1. A local server on port {port}, answering the three endpoints ---")
        print(f"  wrote {SERVER_FILE.name} and started it as a subprocess")
        print("  it answers /v1/workflows/run, /v1/chat-messages and "
              "/v1/completion-messages")
        print("  it is a workflow deployment and declares exactly one input")
        print("  variable, 'question'; the other two endpoints exist and refuse")

        # 2. The blocking call
        print("\n--- 2. The blocking call ---")
        status, body = client.run_blocking(question)
        print(f"  HTTP {status}  run {body['workflow_run_id']}  "
              f"status {body['data']['status']}")
        print(f"  answer: {body['data']['outputs']['answer']}")
        print("  one request, one response, and nothing observable in between")

        # 3. The same run, streamed
        print("\n--- 3. The same run, streamed ---")
        started = time.time()
        status, events = client.run_streaming(question)
        for event in events:
            detail = event.get("data", {}).get("title") or event.get("workflow_run_id", "")
            print(f"  {event['event']:<18} {detail}")
        print(f"  {len(events)} events in {time.time() - started:.2f}s; the node "
              f"events are the only view of what ran")
        print(f"  answer: {events[-1]['data']['outputs']['answer']}")

        # 4. The headers a client prints while debugging
        print("\n--- 4. The headers a client prints while debugging ---")
        print(f"  as written: {client.headers}")
        print(f"  redacted  : {redacted(client.headers)}")
        print("  the first form puts the credential into stdout, and into any log")
        print("  that collects it")

        # 5. Two requests that both come back HTTP 400
        print("\n--- 5. Two requests that both come back HTTP 400 ---")
        status, body = client.run_dropping_input(question)
        print(f"  right endpoint, empty inputs : HTTP {status}  {describe(body)}")
        print("  the endpoint is right and the key is right; the payload has an")
        print("  empty inputs object, so the deployment refuses for its missing input")
        print(f"  variable. The body's message says {body['detail']['message']!r};")
        print("  the status code alone does not")
        status, body = client.chat_on_a_workflow_deployment(question)
        print(f"  wrong endpoint, valid body   : HTTP {status}  {describe(body)}")
        print("  this payload carries the question and is the correct shape for a")
        print("  chat application; the deployment behind this key is a workflow, so")
        print("  the endpoint itself is the mistake and no payload repairs it")
        print("  the two share a status code and agree on nothing else: the status")
        print("  says that a request failed, and only the code in the body says why")

        # 6. A client probing for the shape
        print("\n--- 6. A client probing for the shape ---")
        attempts, result = client.run_probing(question)
        for shape, status, body in attempts:
            note = "accepted" if status == 200 else describe(body)
            print(f"  inputs={str(shape):<14} HTTP {status}  {note}")
        print(f"  {len(attempts)} request(s) to arrive at a key the deployment names")
        answer = result["data"]["outputs"]["answer"]
        print(f"  in its own configuration: answer -> {answer}")

        # 7. The same probe against a server that is gone
        print("\n--- 7. The same probe against a server that is gone ---")
        server.terminate()
        server.wait(timeout=10)
        attempts, result = client.run_probing(question, timeout=2)
        for shape, status, body in attempts:
            print(f"  inputs={str(shape):<14} HTTP {status}  {describe(body)}")
        print(f"  reported to the caller: {result['message']!r}")
        print(f"  {len(attempts)} transport failures in a row, and the message that "
              f"comes back")
        print("  names the application configuration and the API key instead")
    finally:
        if server.poll() is None:
            server.terminate()
            server.wait(timeout=10)
        print(f"\n  server stopped, exit code {server.returncode}")


if __name__ == "__main__":
    main()
