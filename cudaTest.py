import nemo.collections.asr as nemo_asr

# Load the QuartzNet model (fast & accurate)
asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name="stt_en_quartznet15x5")

# Convert audio file to text (ensure 16kHz mono WAV)
text = asr_model.transcribe(["your_audio_file.wav"])[0]
print("Transcribed Text:", text)
