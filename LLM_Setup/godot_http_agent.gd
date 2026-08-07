extends Node

# A small HTTP client node that sends game state to the Python bridge.
var http_request := HTTPRequest.new()
var url := "http://127.0.0.1:8000/step"

func _ready():
    # Create the request node and connect the completion signal.
    add_child(http_request)
    http_request.request_completed.connect(_on_request_completed)
    _send_state()

func _send_state():
    # Build a simple JSON payload representing the current game state.
    # Replace this with your actual Godot variables later.
    var state = {
        "player": {"x": 5, "y": 2},
        "target": {"x": 10, "y": 2},
        "health": 100
    }
    var headers = ["Content-Type: application/json"]
    var body = JSON.stringify(state)

    # Send the state to the Python agent bridge.
    http_request.request(url, headers, HTTPClient.METHOD_POST, body)

func _on_request_completed(result, response_code, headers, body):
    # Read the action returned by Python and print it for debugging.
    var response = JSON.parse_string(body.get_string_from_utf8())
    print("Agent action:", response.get("action", "idle"))

    # Repeat the loop after a short delay.
    get_tree().create_timer(0.1).timeout.connect(_send_state)
