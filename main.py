import tkinter as tk
import socketio
import threading
import sounddevice as sd
import queue
import vosk
import json
import numpy as np

# Default WebSocket URLs
LOCAL_SERVER = "http://127.0.0.1:5000"
DEPLOYED_SERVER = "https://forstingbunbun.ru"

# Load VOSK Model (Download: https://alphacephei.com/vosk/models)
MODEL_PATH = "vosk-model-en-us-0.42-gigaspeech"

# Set up VOSK recognizer
vosk.SetLogLevel(-1)  # Suppress logs
model = vosk.Model(MODEL_PATH)
recognizer = vosk.KaldiRecognizer(model, 16000)
audio_queue = queue.Queue()

# Silence detection threshold
SILENCE_THRESHOLD = 0.02  # Threshold for silence detection (adjustable)

class STTClientApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Live STT WebSocket Client")
        self.root.geometry("500x400")

        # UI Elements
        self.label = tk.Label(root, text="Enter Text (or use mic):")
        self.label.pack(pady=5)

        self.text_input = tk.Entry(root, width=50)
        self.text_input.pack(pady=5)

        self.send_button = tk.Button(root, text="Send", command=self.send_text)
        self.send_button.pack(pady=5)

        # Server toggle checkbox
        self.test_mode = tk.BooleanVar(value=True)
        self.checkbox = tk.Checkbutton(root, text="Use Testing Server", variable=self.test_mode, command=self.toggle_server)
        self.checkbox.pack(pady=5)

        # Mic Selection Dropdown
        self.mic_label = tk.Label(root, text="Select Microphone:")
        self.mic_label.pack(pady=5)

        self.mic_list = [f"{i}: {name}" for i, name in enumerate(sd.query_devices())]
        self.selected_mic = tk.StringVar()
        self.selected_mic.set(self.mic_list[0][0])  # Default to first mic

        self.mic_dropdown = tk.OptionMenu(root, self.selected_mic, *self.mic_list)
        self.mic_dropdown.pack(pady=5)

        # STT Toggle Button
        self.listening = False
        self.stt_button = tk.Button(root, text="Start STT", command=self.toggle_stt)
        self.stt_button.pack(pady=5)

        # Log area
        self.log_area = tk.Text(root, height=10, width=60, state=tk.DISABLED)
        self.log_area.pack(pady=5)

        # WebSocket client
        self.sio = socketio.Client()
        self.sio.on("stt_transcription_update", self.on_message)

        # Connect WebSocket
        self.connect_socket()

    def toggle_server(self):
        """Switch between local and deployed server"""
        self.disconnect_socket()
        self.connect_socket()

    def connect_socket(self):
        """Connect to WebSocket"""
        server_url = LOCAL_SERVER if self.test_mode.get() else DEPLOYED_SERVER
        try:
            self.sio.connect(server_url)
            self.log(f"Connected to {server_url}")
        except Exception as e:
            self.log(f"Connection failed: {e}")

    def disconnect_socket(self):
        """Disconnect from WebSocket"""
        if self.sio.connected:
            self.sio.disconnect()
            self.log("Disconnected from server")

    def send_text(self):
        """Send manual text input"""
        message = self.text_input.get().strip()
        if message:
            self.sio.emit("stt_transcription", message)
            self.log(f"Sent: {message}")
            self.text_input.delete(0, tk.END)

    def on_message(self, data):
        """Handle incoming messages"""
        self.log(f"Received: {data}")

    def log(self, message):
        """Log messages in UI"""
        self.log_area.config(state=tk.NORMAL)
        self.log_area.insert(tk.END, message + "\n")
        self.log_area.config(state=tk.DISABLED)
        self.log_area.yview(tk.END)

    def toggle_stt(self):
        """Start/Stop STT"""
        if self.listening:
            self.listening = False
            self.stt_button.config(text="Start STT")
        else:
            self.listening = True
            self.stt_button.config(text="Stop STT")
            threading.Thread(target=self.start_stt, daemon=True).start()

    def start_stt(self):
        """Start microphone stream and process audio in real-time"""
        self.log("Listening...")

        def callback(indata, frames, time, status):
            """Process incoming audio in chunks"""
            if status:
                self.log(f"STT Error: {status}")
                return
            audio_queue.put(indata.copy())

        # Get selected mic index
        mic_index = int(self.selected_mic.get().split(":")[0])

        # Start audio stream with selected microphone
        with sd.InputStream(device=mic_index, callback=callback, channels=1, samplerate=16000, blocksize=1024, dtype='int16'):
            while self.listening:
                self.process_audio()  # Process audio from queue

    def process_audio(self):
        """Process audio from queue using VOSK with silence threshold"""
        while not audio_queue.empty():
            audio_chunk = audio_queue.get()
            volume_norm = np.linalg.norm(audio_chunk) * 10

            # Apply silence detection
            if volume_norm < SILENCE_THRESHOLD:
                # If volume is below the threshold, skip processing this chunk
                return

            if recognizer.AcceptWaveform(audio_chunk.tobytes()):
                result = json.loads(recognizer.Result())
                if "text" in result and result["text"].strip():
                    filtered_text = self.filter_text(result["text"])
                    if filtered_text:
                        self.log(f"STT: {filtered_text}")
                        self.sio.emit("stt_transcription", filtered_text)

    def filter_text(self, text):
        """Filter out unwanted filler words or phrases like 'the', 'uh', etc."""
        stop_words = {}
        words = text.split()

        # Filter out stop words
        filtered_words = [word for word in words if word.lower() not in stop_words]

        # Rebuild the sentence
        return " ".join(filtered_words) if filtered_words else None


if __name__ == "__main__":
    root = tk.Tk()
    app = STTClientApp(root)
    root.mainloop()
