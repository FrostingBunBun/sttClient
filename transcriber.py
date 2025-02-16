import time
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
import tempfile
import os
from faster_whisper import WhisperModel
from pynput import keyboard

# Threshold for key press time (in seconds)
KEY_PRESS_THRESHOLD = 0.5  # Ignore key presses shorter than 0.5 seconds

# Word replacement dictionary
WORD_REPLACEMENTS = {
    "you": "u",
    "what": "wat",
    "okay": "oke",
}

class WhisperTranscriber:
    def __init__(self, model_size="medium", sample_rate=16000):
        self.model_size = model_size
        self.sample_rate = sample_rate
        self.model = None  # Model will be loaded on demand
        self.is_recording = False
        self.listener = None
        self.press_start_time = None  # Time when the key was pressed
        self.recording_in_progress = False  # Flag to prevent multiple recordings
        self.ignore_recording = False  # Flag to ignore recordings with short key presses

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
        try:
            if str(key.char) == self.keybind:
                if not self.is_recording and not self.recording_in_progress:
                    self.press_start_time = time.time()  # Record the start time of the press
                    self.is_recording = True
                    self.recording_in_progress = True  # Set flag to prevent multiple recordings
                    self.ignore_recording = False  # Reset ignore flag
                    print("Recording started")
        except AttributeError:
            # Handle special keys like space, ctrl, etc.
            if str(key) == self.keybind:
                if not self.is_recording and not self.recording_in_progress:
                    self.press_start_time = time.time()  # Record the start time of the press
                    self.is_recording = True
                    self.recording_in_progress = True  # Set flag to prevent multiple recordings
                    self.ignore_recording = False  # Reset ignore flag
                    print("Recording started")

    def on_release(self, key):
        """Handle key release events."""
        try:
            if str(key.char) == self.keybind and self.press_start_time:
                press_duration = time.time() - self.press_start_time
                if press_duration < KEY_PRESS_THRESHOLD:
                    print(f"Key {key.char} pressed too quickly ({press_duration:.2f}s). Ignoring.")
                    self.ignore_recording = True  # Set flag to ignore this recording
                self.is_recording = False
                self.recording_in_progress = False  # Reset flag
                return False
        except AttributeError:
            if str(key) == self.keybind and self.press_start_time:
                press_duration = time.time() - self.press_start_time
                if press_duration < KEY_PRESS_THRESHOLD:
                    print(f"Key {key} pressed too quickly ({press_duration:.2f}s). Ignoring.")
                    self.ignore_recording = True  # Set flag to ignore this recording
                self.is_recording = False
                self.recording_in_progress = False  # Reset flag
                return False

    def set_keybind(self, keybind):
        """Set the keybind for starting/stopping recording."""
        self.keybind = keybind

    def record_audio(self):
        """Record audio while the keybind is pressed."""
        print("Recording audio...")
        recording = np.array([], dtype='float32').reshape(0, 1)
        frames_per_buffer = int(self.sample_rate * 0.1)

        with keyboard.Listener(on_press=self.on_press, on_release=self.on_release) as listener:
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

        # If the recording should be ignored, return an empty array
        if self.ignore_recording:
            print("Ignoring recording due to short key press.")
            return np.array([], dtype='float32').reshape(0, 1)

        return recording

    def replace_words(self, text):
        """Replace specific words in the text based on the replacement dictionary."""
        for word, replacement in WORD_REPLACEMENTS.items():
            text = text.replace(word, replacement)
        return text

    def capitalize_first_letter(self, text):
        print("Original:", text)
        print("ASCII Codes:", [ord(c) for c in text])  # Debug each character
    
        text = text.lstrip()  # Remove leading spaces
    
        if len(text) > 0:
            capitalized = text[0].upper() + text[1:]
            print("Modified:", capitalized)
            return capitalized
        return text



    def save_temp_audio(self, recording):
        """Save recorded audio to a temporary file and transcribe it."""
        if len(recording) == 0:
            return ""  # Return empty string if recording is ignored

        print("Saving temp audio file for transcription...")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav:
            write(temp_wav.name, self.sample_rate, recording)  # Save recording as WAV
            file_path = temp_wav.name  # Get file path

        segments, info = self.model.transcribe(file_path, beam_size=5)
        print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

        full_transcription = ""
        for segment in segments:
            full_transcription += segment.text + " "

        os.remove(file_path)  # Delete temp file after transcription
        print("Transcription complete.")

        # Normalize text to lowercase
        full_transcription = full_transcription.lower()

        # Replace words
        full_transcription = self.replace_words(full_transcription)

        # Capitalize the first letter
        full_transcription = self.capitalize_first_letter(full_transcription)

        return full_transcription