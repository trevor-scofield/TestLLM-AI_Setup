import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse

# Local host settings for the bridge server.
HOST = "127.0.0.1"
PORT = 8000


def choose_action(state):
    """Choose an action from the incoming game state.

    This is a placeholder policy for testing the connection between Godot and
    Python. Later, you can replace this with a trained model, an LLM policy,
    or any custom decision logic.
    """
    """Return an action for the current game state.

    This starter version uses a tiny heuristic so you can test the connection
    before swapping in a trained model or an LLM-based policy.
    """
    if not state:
        return "idle"

    player = state.get("player", {})
    target = state.get("target", {})

    player_x = player.get("x", 0)
    target_x = target.get("x", player_x)

    if player_x < target_x:
        return "move_right"
    if player_x > target_x:
        return "move_left"

    if state.get("health", 0) <= 0:
        return "respawn"

    return "idle"


class AgentHandler(BaseHTTPRequestHandler):
    # Handle simple health checks so you can confirm the server is up.
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            self._send_json({"status": "ok"})
            return
        self._send_json({"error": "not found"}, status=404)

    # Handle incoming state updates from Godot and return the chosen action.
    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path in {"/step", "/act", "/action"}:
            body = self._read_body()
            try:
                payload = json.loads(body or "{}")
            except json.JSONDecodeError:
                self._send_json({"error": "invalid json"}, status=400)
                return

            # This is the main handoff point: Godot sends state, Python chooses an action.
            action = choose_action(payload)
            self._send_json({"action": action})
            return

        if parsed.path == "/reset":
            self._send_json({"status": "reset"})
            return

        self._send_json({"error": "not found"}, status=404)

    # Read the JSON payload sent by Godot.
    def _read_body(self):
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return ""
        return self.rfile.read(length).decode("utf-8")

    # Send JSON back to Godot so the game can apply the chosen action.
    def _send_json(self, payload, status=200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # Silence default logging noise from the HTTP server.
    def log_message(self, format, *args):
        return


def run_server():
    # Start the local bridge server so Godot can talk to Python over HTTP.
    server = HTTPServer((HOST, PORT), AgentHandler)
    print(f"Agent bridge listening on http://{HOST}:{PORT}")
    server.serve_forever()


if __name__ == "__main__":
    run_server()
