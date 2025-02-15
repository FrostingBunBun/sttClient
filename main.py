# main.py
import tkinter as tk
from tkinter import simpledialog
from config import load_config, save_config
from transcriber import WhisperTranscriber
from websocket_client import WebSocketClient
from ui import STTClientUI
import threading
import sounddevice as sd

class STTClientApp:
    def __init__(self, root):
        self.root = root
        self.config = load_config()

        # Initialize UI
        self.ui = STTClientUI(
            root,
            self.config,
            on_send_text=self.send_text,
            on_try_connect_again=self.try_connect_again,
            on_change_keybind=self.change_keybind,
            on_toggle_stt=self.toggle_stt
        )

        # Initialize WebSocket client with the UI's log method
        self.websocket_client = WebSocketClient(self.on_message, self.ui.log)

        # Initialize Whisper Transcriber
        self.transcriber = WhisperTranscriber()
        self.transcriber.set_keybind(self.config.get("keybind", "space"))

        # Initial connection attempt
        self.try_connect_again()

    def send_text(self):
        """Send manual text input."""
        message = self.ui.text_input.get().strip()
        if message:
            self.websocket_client.send_message(message)
            self.ui.text_input.delete(0, tk.END)

    def on_message(self, data):
        """Handle incoming messages."""
        # print()
        # self.ui.log(f"Received: {data}")

    def try_connect_again(self):
        """Try to connect to the server again."""
        self.websocket_client.connect_socket()

    def change_keybind(self):
        """Change the keybind for starting/stopping recording."""
        keybind = simpledialog.askstring("Change Keybind", "Enter a new keybind (e.g., 'a', 'b', 'ctrl', 'shift'):")
        if keybind:
            self.config["keybind"] = keybind
            self.transcriber.set_keybind(keybind)
            self.ui.keybind_label.config(text=f"Current Keybind: {keybind}")
            save_config(self.config)

    def toggle_stt(self):
        """Start/Stop STT and load/unload the model."""
        if self.ui.listening:
            # Stop listening
            self.ui.listening = False
            self.ui.stt_button.config(text="Start STT")
            self.transcriber.unload_model()
        else:
            # Start listening
            self.ui.listening = True
            self.ui.stt_button.config(text="Stop STT")
            self.transcriber.load_model()

            # Deselect the text input field
            self.root.focus()

            # Start recording in a separate thread
            threading.Thread(target=self.start_stt, daemon=True).start()

    def start_stt(self):
        """Start microphone stream and process audio in real-time."""
        self.ui.log("Starting STT...")
        self.ui.log("Listening...")

        # Get selected mic index
        mic_index = int(self.ui.selected_mic.get().split(":")[0])
        self.config["mic_index"] = mic_index
        save_config(self.config)
        sd.default.device = mic_index

        # Continuous recording as long as the listening flag is True
        while self.ui.listening:
            recording = self.transcriber.record_audio()
            transcription = self.transcriber.save_temp_audio(recording)
            if transcription.strip():
                # self.ui.log(f"Transcription: {transcription}")
                self.websocket_client.send_message(transcription)

if __name__ == "__main__":
    root = tk.Tk()
    app = STTClientApp(root)
    root.mainloop()