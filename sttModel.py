import sounddevice as sd
import numpy as np
import tempfile
import os
from scipy.io.wavfile import write
from faster_whisper import WhisperModel
import queue
import time
import threading

class STTModel:
    def __init__(self, model_size="large-v3", sample_rate=16000):
        self.sample_rate = sample_rate
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.selected_mic = None
        self.chunk_duration = 5  # Duration in seconds for each chunk
        self.transcription_callback = None
        
        print("[DEBUG] Initializing STT Model...")
        print(f"[DEBUG] Sample rate: {sample_rate}")
        self.model = WhisperModel(model_size, device='cuda', compute_type='float32')
        print("[DEBUG] Whisper model loaded successfully")

    def set_transcription_callback(self, callback):
        """Set callback function to handle transcriptions"""
        self.transcription_callback = callback

    def list_microphones(self):
        print("\n[DEBUG] Listing all input devices:")
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                print(f"[DEBUG] Device {i}: {device['name']} (Channels: {device['max_input_channels']})")
        
        microphones = [f"{i}: {name['name']}" for i, name in enumerate(devices) if name['max_input_channels'] > 0]
        return microphones

    def set_microphone(self, mic_index):
        self.selected_mic = int(mic_index)
        print(f"\n[DEBUG] Setting microphone:")
        print(f"[DEBUG] Selected index: {self.selected_mic}")
        device_info = sd.query_devices(self.selected_mic)
        print(f"[DEBUG] Device details: {device_info}")
        
        # Set the correct sample rate for the device
        self.sample_rate = int(device_info['default_samplerate'])
        print(f"[DEBUG] Adjusted sample rate to: {self.sample_rate}")

    def process_audio_chunk(self, recorded_chunks):
        """Process and transcribe an audio chunk"""
        if not recorded_chunks:
            return None

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav:
            file_path = temp_wav.name

        try:
            # Combine and normalize audio chunks
            audio_data = np.concatenate(recorded_chunks)
            audio_data = audio_data / np.max(np.abs(audio_data))
            
            # Save as WAV file
            write(file_path, self.sample_rate, audio_data)
            
            # Transcribe
            segments, info = self.model.transcribe(file_path, beam_size=5)
            transcription = " ".join([segment.text for segment in segments])
            
            # Clean up
            os.remove(file_path)
            
            return transcription.strip()
        except Exception as e:
            print(f"[DEBUG] Error processing audio chunk: {e}")
            if os.path.exists(file_path):
                os.remove(file_path)
            return None

    def continuous_recording(self):
        """Continuously record and process audio in chunks"""
        recorded_chunks = []
        chunk_start_time = time.time()
        
        def audio_callback(indata, frames, time, status):
            if status:
                print(f"[DEBUG] Recording error in callback: {status}")
            if self.is_recording:
                # Convert to mono if necessary
                if indata.shape[1] > 1:
                    audio_data = np.mean(indata, axis=1)
                else:
                    audio_data = indata[:, 0]
                recorded_chunks.append(audio_data)

        try:
            stream = sd.InputStream(
                device=self.selected_mic,
                channels=2,
                samplerate=self.sample_rate,
                callback=audio_callback
            )
            
            print("[DEBUG] Starting continuous audio stream...")
            with stream:
                while self.is_recording:
                    current_time = time.time()
                    elapsed_time = current_time - chunk_start_time
                    
                    # Process chunk when duration is reached
                    if elapsed_time >= self.chunk_duration and recorded_chunks:
                        print(f"[DEBUG] Processing chunk of {len(recorded_chunks)} chunks")
                        
                        # Process in a separate thread to avoid blocking
                        chunks_to_process = recorded_chunks.copy()
                        recorded_chunks.clear()
                        
                        def process_chunk():
                            transcription = self.process_audio_chunk(chunks_to_process)
                            if transcription and self.transcription_callback:
                                self.transcription_callback(transcription)
                        
                        threading.Thread(target=process_chunk, daemon=True).start()
                        chunk_start_time = current_time
                    
                    sd.sleep(100)  # Small sleep to prevent busy waiting
                
            print("[DEBUG] Recording stopped")
            
        except Exception as e:
            print(f"[DEBUG] Error in continuous recording: {e}")

    def start_stt(self):
        """Start continuous STT process"""
        print("[DEBUG] Starting continuous STT")
        self.is_recording = True
        
        # Start recording in a separate thread
        threading.Thread(target=self.continuous_recording, daemon=True).start()

    def stop_stt(self):
        """Stop STT process"""
        print("[DEBUG] Stopping STT")
        self.is_recording = False