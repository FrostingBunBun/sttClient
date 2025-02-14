import sys
import pyaudio
import numpy as np
import torch
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QTextEdit, QComboBox, QLabel, QProgressBar
from PyQt6.QtCore import QThread, pyqtSignal
from whisper import load_model


global_state = {}

def get_microphones():
    p = pyaudio.PyAudio()
    devices = [(p.get_device_info_by_index(i)['name'].encode('utf-8', 'ignore').decode(), i) for i in range(p.get_device_count())]
    p.terminate()
    return devices

class AudioStreamer(QThread):
    text_ready = pyqtSignal(str)
    volume_ready = pyqtSignal(int)
    log_ready = pyqtSignal(str)

    def __init__(self, mic_index):
        super().__init__()
        self.mic_index = mic_index
        self.running = True

    def run(self):
        p = pyaudio.PyAudio()
        try:
            stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True,
                            input_device_index=self.mic_index, frames_per_buffer=1024)
            self.log_ready.emit(f"Microphone {self.mic_index} opened successfully.")
        except Exception as e:
            self.log_ready.emit(f"Error opening microphone: {str(e)}")
            return

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = load_model("base", device=device)
        self.log_ready.emit(f"Model loaded on {device}. Recording started.")

        audio_buffer = np.array([], dtype=np.float32)
        
        while self.running:
            data = stream.read(1024, exception_on_overflow=False)
            audio_np = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            volume = int(np.linalg.norm(audio_np) * 10)
            self.volume_ready.emit(volume)
            
            audio_buffer = np.concatenate((audio_buffer, audio_np))
            if len(audio_buffer) >= 16000 * 0.5:  # Process every half second
                try:
                    result = model.transcribe(audio_buffer, word_timestamps=True)
                    words = " ".join([word["text"] for word in result["segments"]])
                    self.text_ready.emit(words)
                    self.log_ready.emit("Transcribed words: " + words)
                except Exception as e:
                    self.log_ready.emit(f"Transcription error: {str(e)}")
                audio_buffer = np.array([], dtype=np.float32)

        stream.stop_stream()
        stream.close()
        p.terminate()
        self.log_ready.emit("Recording stopped.")

    def stop(self):
        self.running = False


def start_recording():
    mic_index = global_state["mic_select"].currentData()
    if mic_index is None:
        global_state["log_display"].append("No microphone selected!")
        return
    
    global_state["recorder"] = AudioStreamer(mic_index)
    global_state["recorder"].text_ready.connect(global_state["text_display"].append)
    global_state["recorder"].volume_ready.connect(global_state["volume_bar"].setValue)
    global_state["recorder"].log_ready.connect(global_state["log_display"].append)
    global_state["recorder"].start()
    global_state["start_button"].setEnabled(False)
    global_state["stop_button"].setEnabled(True)

def stop_recording():
    if global_state.get("recorder"):
        global_state["recorder"].stop()
        global_state["recorder"].wait()
        global_state["recorder"] = None
    global_state["start_button"].setEnabled(True)
    global_state["stop_button"].setEnabled(False)

def setup_ui():
    window = QWidget()
    window.setWindowTitle("Speech to Text Client")
    window.setGeometry(100, 100, 500, 400)
    layout = QVBoxLayout()
    
    global_state["mic_select"] = QComboBox()
    for name, index in get_microphones():
        global_state["mic_select"].addItem(name, index)
    layout.addWidget(QLabel("Select Microphone:"))
    layout.addWidget(global_state["mic_select"])
    
    global_state["volume_bar"] = QProgressBar()
    global_state["volume_bar"].setRange(0, 100)
    layout.addWidget(QLabel("Input Volume:"))
    layout.addWidget(global_state["volume_bar"])
    
    global_state["text_display"] = QTextEdit()
    global_state["text_display"].setReadOnly(True)
    layout.addWidget(QLabel("Transcribed Text:"))
    layout.addWidget(global_state["text_display"])
    
    global_state["start_button"] = QPushButton("Start")
    global_state["start_button"].clicked.connect(start_recording)
    layout.addWidget(global_state["start_button"])
    
    global_state["stop_button"] = QPushButton("Stop")
    global_state["stop_button"].setEnabled(False)
    global_state["stop_button"].clicked.connect(stop_recording)
    layout.addWidget(global_state["stop_button"])
    
    global_state["log_display"] = QTextEdit()
    global_state["log_display"].setReadOnly(True)
    layout.addWidget(QLabel("Debug Log:"))
    layout.addWidget(global_state["log_display"])
    
    window.setLayout(layout)
    return window

if __name__ == "__main__":
    app = QApplication(sys.argv)
    global_state["app"] = app
    global_state["window"] = setup_ui()
    global_state["window"].show()
    sys.exit(app.exec())
