import base64

import modal
import fastapi
import librosa
import torch.nn as nn
import torchaudio.transforms as t
import torch
from model import AudioCNN
from pydantic import BaseModel

app = modal.App("audio-cnn-inference")
image = (modal.Image.debian_slim()
         .pip_install_from_requirements("requirements.txt")
         .add_local_python_source("model")
         )
model_volume = modal.Volume.from_name("esc-model")

class AudioProcessor:
    def __init__(self):
        self.transform = nn.Sequential(
            t.MelSpectrogram(
                sample_rate=22050,
                n_fft=1024,
                hop_length=512,
                n_mels=128,
                f_min=0,
                f_max=11025
            ),
            t.AmplitudeToDB()
        )

    def process_audio_chunk(self, audio_chunk):
        waveform = torch.from_numpy(audio_chunk).float()
        waveform = waveform.unsqueeze(0)
        spec = self.transform(waveform)
        return spec.unsqueeze(0)

class InferenceRequest(BaseModel):
    audio_data: str

@app.cls(image=image, gpu="A10G", volumes={"/models": model_volume}, scaledown_window=15)
class AudioClassifier:
    @modal.enter()
    def load_model(self):
        print("Loading model...")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint = torch.load('/models/best_model.pth', map_location=self.device)

        self.classes = checkpoint["classes"]
        self.model = AudioCNN(num_classes=len(self.classes))
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        self.audio_processor = AudioProcessor()

        print("Model loaded.")

    @modal.fastapi_endpoint(method="POST")
    def inference(self, request: InferenceRequest):
        audio_bytes = base64.b64decode(request.audio_data)

