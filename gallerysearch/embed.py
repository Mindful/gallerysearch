from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import numpy as np
from gallerysearch.gather import GalleryPair


class CLIPEmbedder:
    def __init__(self, model_name: str):
        # Load the CLIP model and processor
        self.model_name = model_name
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def embed_image(self, target: GalleryPair) -> np.ndarray:
        # Open and preprocess the image
        image = Image.open(target.img_file).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt", padding=True)

        # Get the image embeddings
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)

        return image_features.cpu().detach().numpy().flatten()

    def embed_text(self, text: str) -> np.ndarray:
        # Preprocess the text
        inputs = self.processor(text=text, return_tensors="pt", padding=True)

        # Get the text embeddings
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)

        return text_features.cpu().detach().numpy().flatten().reshape(1, -1)
