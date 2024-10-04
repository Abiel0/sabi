import torch
import torchaudio
import numpy as np
from transformers import VitsModel, AutoTokenizer

# Check if CUDA is available and set the device accordingly
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Define the model name
model_name = "facebook/mms-tts-eng"

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = VitsModel.from_pretrained(model_name).to(device)

# Prepare the text input
text = "Hello, I am Abiel from India and I love Jesus Christ."

# Tokenize the input
inputs = tokenizer(text, return_tensors="pt").to(device)

# Generate speech
with torch.no_grad():
    output = model(**inputs).waveform

# Convert the output tensor to a numpy array
audio = output.cpu().numpy().squeeze()

# Normalize the audio
audio = audio / np.abs(audio).max()

# Save the audio to a file
sample_rate = model.config.sampling_rate
torchaudio.save("natural_tts_output.wav", torch.tensor(audio).unsqueeze(0), sample_rate)

print("Speech generated and saved as 'natural_tts_output.wav'")

# Optional: Play the audio (if you're running this in an environment with audio output)
# import IPython.display as ipd
# ipd.display(ipd.Audio(audio, rate=sample_rate))
