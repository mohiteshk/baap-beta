import torch
from transformers import CLIPProcessor, CLIPModel
from core.config import config

class VisionTextModel:
    def __init__(self, device):
        self.device = device
        self.model_id = config.get("model_id", "openai/clip-vit-base-patch32")
        print("Loading AI Models...")
        
        try:
            self.processor = CLIPProcessor.from_pretrained(self.model_id, local_files_only=True)
            self.model = CLIPModel.from_pretrained(self.model_id, torch_dtype=torch.float16, local_files_only=True).to(self.device)
            print("✅ Model loaded instantly from local disk cache.")
        except Exception:
            print("⚠️ Local cache not found. Downloading model... (This will only happen once)")
            self.processor = CLIPProcessor.from_pretrained(self.model_id)
            self.model = CLIPModel.from_pretrained(self.model_id, torch_dtype=torch.float16).to(self.device)

    def get_text_embedding(self, prompt: str) -> list:
        """Processes text and returns a flattened 1D list embedding."""
        inputs = self.processor(text=[prompt], return_tensors="pt", padding=True).to(self.device)
        
        with torch.no_grad():
            text_outputs = self.model.text_model(
                input_ids=inputs.input_ids, 
                attention_mask=inputs.attention_mask
            )
            pooled_output = text_outputs.pooler_output
            text_features = self.model.text_projection(pooled_output)
            text_features = text_features.to(torch.float32)
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
            return text_features.cpu().numpy().flatten().tolist()

    def get_image_embeddings(self, frames: list) -> list:
        """Processes a batch of PIL images and returns a list of embeddings."""
        inputs = self.processor(images=frames, return_tensors="pt").to(self.device, torch.float16)
        
        with torch.no_grad():
            image_features = self.model.get_image_features(pixel_values=inputs.pixel_values)
            
            if not isinstance(image_features, torch.Tensor):
                if hasattr(image_features, 'pooler_output'):
                    image_features = image_features.pooler_output
                elif hasattr(image_features, 'image_embeds'):
                    image_features = image_features.image_embeds
                else:
                    image_features = image_features[0]
                    
            image_features = image_features.to(torch.float32)
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            return image_features.cpu().numpy().tolist()