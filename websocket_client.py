# websocket_client.py
import socketio
import threading
import time

# WebSocket URL
SERVER_URL = "https://frostingbunbun.ru"

class WebSocketClient:
    def __init__(self, on_message_callback, log_callback):
        self.sio = socketio.Client()
        self.sio.on("stt_transcription_update", on_message_callback)
        self.is_connected = False
        self.connection_in_progress = False
        self.log_callback = log_callback  # Callback to log messages to the UI

    def connect_socket(self):
        """Connect to WebSocket in a separate thread."""
        def connect_thread():
            self.connection_in_progress = True  # Set flag to true to prevent reconnection attempts

            attempt_count = 0
            max_retries = 3  # Maximum number of retry attempts

            while attempt_count < max_retries:
                try:
                    # If it's the first connection attempt
                    if attempt_count == 0:
                        self.log_callback(f"Attempting to connect to {SERVER_URL}...")
                    else:
                        self.log_callback(f"Retrying to connect to {SERVER_URL} (Attempt {attempt_count + 1})...")

                    self.sio.connect(SERVER_URL)
                    self.is_connected = True
                    self.connection_in_progress = False  # Reset flag after successful connection
                    self.log_callback(f"Connected to {SERVER_URL}")
                    break  # Exit the loop after a successful connection
                except Exception as e:
                    self.connection_in_progress = False  # Reset flag even on failure
                    self.log_callback(f"Connection failed: {e}")
                    attempt_count += 1
                    if attempt_count < max_retries:
                        self.log_callback("Retrying...")
                        time.sleep(1)  # Wait before retrying
                    else:
                        self.log_callback(f"Failed to connect after {max_retries} attempts.")
                        break

        # Start the connection attempt in a new thread
        threading.Thread(target=connect_thread, daemon=True).start()

    def disconnect_socket(self):
        """Disconnect from WebSocket."""
        if self.sio.connected:
            self.sio.disconnect()
            self.is_connected = False
            self.log_callback("Disconnected from server")

    def send_message(self, message):
        """Send a message via WebSocket."""
        if self.is_connected:
            self.sio.emit("stt_transcription", message)
            self.log_callback(f"Sent: {message}")