import tkinter as tk
from tkinter import simpledialog
import socketio
import threading
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import tempfile
import os
from faster_whisper import WhisperModel
from pynput import keyboard
from pynput.keyboard import Key, Listener
import json

# Default WebSocket URLs
LOCAL_SERVER = "http://127.0.0.1:5000"
DEPLOYED_SERVER = "https://frostingbunbun.ru"

# Configuration file to save keybind and mic selection
CONFIG_FILE = "stt_config.json"

class WhisperTranscriber:
    def __init__(self, model_size="medium", sample_rate=16000):
        self.model_size = model_size
        self.sample_rate = sample_rate
        self.model = None  # Model will be loaded on demand
        self.is_recording = False
        self.listener = None

    def load_model(self):
        """Load the Whisper model."""
        if self.model is None:
            self.model = WhisperModel(self.model_size, device="cuda", compute_type="float32")
            print("Whisper model loaded.")

    def unload_model(self):
        """Unload the Whisper model."""
        if self.model is not None:
            self.model = None
            print("Whisper model unloaded.")

    def on_press(self, key):
        """Handle key press events."""
        if key == self.keybind:
            if not self.is_recording:
                self.is_recording = True
                print("Recording started")

    def on_release(self, key):
        """Handle key release events."""
        if key == self.keybind:
            if self.is_recording:
                self.is_recording = False
                print("Recording stopped")
                return False

    def set_keybind(self, keybind):
        """Set the keybind for starting/stopping recording."""
        self.keybind = keybind

    def record_audio(self):
        """Record audio while the keybind is pressed."""
        print("Recording audio...")
        recording = np.array([], dtype='float32').reshape(0, 1)
        frames_per_buffer = int(self.sample_rate * 0.1)

        with Listener(on_press=self.on_press, on_release=self.on_release) as listener:
            self.listener = listener
            while True:
                if self.is_recording:
                    chunk = sd.rec(frames_per_buffer, samplerate=self.sample_rate, channels=1, dtype='float32')
                    sd.wait()
                    recording = np.vstack([recording, chunk])
                if not self.is_recording and len(recording) > 0:
                    break
            listener.join()
        
        print(f"Recorded {len(recording)} frames.")
        return recording
    
    def save_temp_audio(self, recording):
        """Save recorded audio to a temporary file and transcribe it."""
        print("Saving temp audio file for transcription...")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav:
            write(temp_wav.name, self.sample_rate, recording)  # Save recording as WAV
            file_path = temp_wav.name  # Get file path

        segments, info = self.model.transcribe(file_path, beam_size=5)
        print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

        full_transcription = ""
        for segment in segments:
            print(f"Segment text: {segment.text}")
            full_transcription += segment.text + " "

        os.remove(file_path)  # Delete temp file after transcription
        print("Transcription complete.")
        return full_transcription

class STTClientApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Live STT WebSocket Client")
        self.root.geometry("500x400")

        # Load configuration
        self.config = self.load_config()

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

        self.mic_list = [f"{i}: {device['name']}" for i, device in enumerate(sd.query_devices())]
        self.selected_mic = tk.StringVar()
        self.selected_mic.set(self.mic_list[self.config.get("mic_index", 0)])  # Default to saved mic

        self.mic_dropdown = tk.OptionMenu(root, self.selected_mic, *self.mic_list)
        self.mic_dropdown.pack(pady=5)

        # Keybind Selection
        self.keybind_label = tk.Label(root, text=f"Current Keybind: {self.config.get('keybind', 'space')}")
        self.keybind_label.pack(pady=5)

        self.keybind_button = tk.Button(root, text="Change Keybind", command=self.change_keybind)
        self.keybind_button.pack(pady=5)

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

        # Initialize Whisper Transcriber
        self.transcriber = WhisperTranscriber()
        self.transcriber.set_keybind(self.config.get("keybind", Key.space))

    def load_config(self):
        """Load configuration from file."""
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, "r") as f:
                return json.load(f)
        return {}

    def save_config(self):
        """Save configuration to file."""
        with open(CONFIG_FILE, "w") as f:
            json.dump(self.config, f)

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
            print(f"Connected to {server_url}")  # Debugging: Confirm successful connection
        except Exception as e:
            self.log(f"Connection failed: {e}")
            print(f"Connection failed: {e}")  # Debugging: If connection fails


    def disconnect_socket(self):
        """Disconnect from WebSocket"""
        if self.sio.connected:
            self.sio.disconnect()
            self.log("Disconnected from server")

    def send_text(self):
        """Send manual text input"""
        message = self.text_input.get().strip()
        if message:
            print(f"Sending message: {message}")  # Debugging: Show the message being sent
            self.sio.emit("stt_transcription", message)
            self.log(f"Sent: {message}")
            self.text_input.delete(0, tk.END)
        else:
            print("No message to send.")  # Debugging: If no message is entered

    def on_message(self, data):
        """Handle incoming messages"""
        self.log(f"Received: {data}")

    def log(self, message):
        """Log messages in UI"""
        self.log_area.config(state=tk.NORMAL)
        self.log_area.insert(tk.END, message + "\n")
        self.log_area.config(state=tk.DISABLED)
        self.log_area.yview(tk.END)

    def change_keybind(self):
        """Change the keybind for starting/stopping recording."""
        keybind = simpledialog.askstring("Change Keybind", "Enter a new keybind (e.g., 'space', 'ctrl', 'alt'):")
        if keybind:
            try:
                self.config["keybind"] = getattr(Key, keybind)
                self.transcriber.set_keybind(self.config["keybind"])
                self.keybind_label.config(text=f"Current Keybind: {keybind}")
                self.save_config()
            except AttributeError:
                self.log("Invalid keybind. Please try again.")

    def toggle_stt(self):
        """Start/Stop STT and load/unload the model."""
        if self.listening:
            self.listening = False
            self.stt_button.config(text="Start STT")
            self.transcriber.unload_model()  # Unload the model
        else:
            self.listening = True
            self.stt_button.config(text="Stop STT")
            self.transcriber.load_model()  # Load the model
            threading.Thread(target=self.start_stt, daemon=True).start()

    def start_stt(self):
        """Start microphone stream and process audio in real-time."""
        print("Starting STT...")
        self.log("Listening...")

        # Get selected mic index
        mic_index = int(self.selected_mic.get().split(":")[0])
        self.config["mic_index"] = mic_index
        self.save_config()
        sd.default.device = mic_index

        while self.listening:
            recording = self.transcriber.record_audio()
            transcription = self.transcriber.save_temp_audio(recording)
            if transcription.strip():
                print(f"Transcription: {transcription}")
                self.log(f"STT: {transcription}")
                self.sio.emit("stt_transcription", transcription)



if __name__ == "__main__":
    root = tk.Tk()
    app = STTClientApp(root)
    root.mainloop()