import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
import queue

# Set up the model (GPU or CPU, FP16 or INT8)
model = WhisperModel("large-v3", device="cuda", compute_type="float16")  # For GPU
# If using CPU:
# model = WhisperModel("large-v3", device="cpu", compute_type="int8")

# Create a queue to hold audio data
audio_queue = queue.Queue()

# Callback function to capture audio from the microphone
def callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    audio_queue.put(indata.copy())

# Initialize the microphone stream
mic_stream = sd.InputStream(callback=callback, channels=1, samplerate=16000, blocksize=1024, dtype="int16")
mic_stream.start()

# Real-time transcription loop
while True:
    # Get the latest audio chunk from the queue
    audio_data = audio_queue.get()

    # Convert the audio data to float32, as required by the model
    audio_data = np.array(audio_data, dtype=np.float32)

    # Run the transcription using the faster-whisper model
    segments, _ = model.transcribe(audio_data, beam_size=3, vad_filter=True)

    # Output the transcription results
    for segment in segments:
        print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
