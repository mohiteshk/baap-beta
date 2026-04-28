import os
import torch
from transformers import CLIPProcessor, CLIPModel
from core.config import config

class VisionTextModel:
    def __init__(self, device):
        self.device = device
        self.model_id = config.get("model_id", "openai/clip-vit-base-patch32")
        self.model_dir = config.get("model_dir", "./ai_models")
        os.makedirs(self.model_dir, exist_ok=True)
        
        print(f"Loading AI Models (Target Dir: {self.model_dir})...")
        
        try:
            # 1. Attempt to load strictly from local cache first
            self.processor = CLIPProcessor.from_pretrained(
                self.model_id, 
                cache_dir=self.model_dir,
                local_files_only=True
            )
            self.model = CLIPModel.from_pretrained(
                self.model_id, 
                torch_dtype=torch.float16, 
                cache_dir=self.model_dir,
                local_files_only=True
            ).to(self.device)
            print("✅ Model loaded instantly from local disk cache.")
            
        except Exception:
            print("⚠️ Local cache not found or incomplete. Downloading model...")
            try:
                # 2. Download the model to the specific directory
                self.processor = CLIPProcessor.from_pretrained(
                    self.model_id, 
                    cache_dir=self.model_dir
                )
                self.model = CLIPModel.from_pretrained(
                    self.model_id, 
                    torch_dtype=torch.float16, 
                    cache_dir=self.model_dir
                ).to(self.device)
                print(f"✅ Model downloaded successfully to: {self.model_dir}")
                
            except Exception as download_error:
                print("\n❌ FAILED to download the model. Network error or invalid model ID:")
                print(download_error)
                raise

    def get_text_embedding(self, prompt: str) -> list:
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