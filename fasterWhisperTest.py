import sounddevice as sd
import numpy as np
from pynput import keyboard
from scipy.io.wavfile import write
import tempfile
import os
from faster_whisper import WhisperModel



import torch
print(torch.__version__)
print(torch.cuda.is_available())



class WhisperTranscriber:
    def __init__(self, model_size="large-v3", sample_rate=48000):
        self.model_size = model_size
        self.sample_rate = sample_rate
        self.model = WhisperModel(model_size, device='cuda', compute_type='float32')
        self.is_recording = False

    def on_press(self, key):
        if key == keyboard.Key.space:
            if not self.is_recording:
                self.is_recording = True
                print("Recording started")

    def on_release(self, key):
        if key == keyboard.Key.space:
            if self.is_recording:
                self.is_recording = False
                print("Recording stopped")
                return False

    def record_audio(self):
        recording = np.array([], dtype='float64').reshape(0, 2)
        frames_per_buffer = int(self.sample_rate * 0.1)

        with keyboard.Listener(on_press=self.on_press, on_release=self.on_release) as listener:
            while True:
                if self.is_recording:
                    chunk = sd.rec(frames_per_buffer, samplerate=self.sample_rate, channels=2, dtype='float64')
                    sd.wait()
                    recording = np.vstack([recording, chunk])
                if not self.is_recording and len(recording) > 0:
                    break
            listener.join()
        
        return recording
    
    def save_temp_audio(self, recording):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav:
            write(temp_wav.name, self.sample_rate, recording)  # Save recording as WAV
            file_path = temp_wav.name  # Get file path

        segments, info = self.model.transcribe(file_path, beam_size=5)
        print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

        full_transcription = ""
        for segment in segments:
            print(segment.text)
            full_transcription += segment.text + " "

        os.remove(file_path)  # Delete temp file after transcription
        return full_transcription


    def run(self):
        print("Hold the spacebar to start recording...")
        while True:
            recording = self.record_audio()
            transcription = self.save_temp_audio(recording)  # Corrected function call
            print(transcription)  # Display the transcription
            print("\nPress the spacebar to start recording again, or press Ctrl+C to exit")


if __name__ == "__main__":
    sd.default.device = 8

    print("Using microphone:", sd.query_devices(sd.default.device['input'])['name'])

    transcriber = WhisperTranscriber()
    transcriber.run()