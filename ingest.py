import os
# --- AMD ROCm 6700 XT STABILITY FIXES ---
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
os.environ["MIOPEN_DISABLE_CACHE"] = "1"
os.environ["PYTORCH_HIP_ALLOC_CONF"] = "garbage_collection_threshold:0.8,max_split_size_mb:512"

import cv2
# Prevent OpenCV from colliding with PyTorch threads
cv2.setNumThreads(0) 

import torch

# Disable unstable optimized kernels for RDNA2
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
torch.backends.cudnn.enabled = False
# ----------------------------------------

import chromadb
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

# --- CONFIGURATION ---
VIDEO_FOLDER = "./my_drone_videos"
DB_FOLDER = "./chroma_db"
FPS_TO_EXTRACT = 1

print("Loading CLIP Model onto GPU in Float16 (Safe Mode)...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "openai/clip-vit-base-patch32"

processor = CLIPProcessor.from_pretrained(model_id)
model = CLIPModel.from_pretrained(model_id, torch_dtype=torch.float16).to(device)

print("Initializing Chroma Database...")
chroma_client = chromadb.PersistentClient(path=DB_FOLDER)
collection = chroma_client.get_or_create_collection(
    name="drone_footage",
    metadata={"hnsw:space": "cosine"}
)

def process_video(video_path):
    filename = os.path.basename(video_path)
    print(f"\nProcessing: {filename}")
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if fps == 0 or total_frames == 0:
        print(f"Skipping {filename} - couldn't read video properties.")
        return

    frame_interval = int(fps * FPS_TO_EXTRACT)
    current_frame = 0
    
    with tqdm(total=total_frames) as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if current_frame % frame_interval == 0:
                timestamp_sec = current_frame / fps
                
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_frame)
                
                # Convert inputs to float16
                inputs = processor(images=pil_img, return_tensors="pt").to(device, torch.float16)
                
                #with torch.no_grad():
                #    # The Fix: Use standard model call and extract image_embeds safely
                #    #outputs = model(**inputs)
                #    #image_features = outputs.image_embeds
                #    
                #    # Safely extract ONLY the image features, completely ignoring text
                #    image_features = model.get_image_features(pixel_values=inputs.pixel_values)
                #
                #    # Convert back to float32 for normalization and math
                #    image_features = image_features.to(torch.float32)
                #    image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
                #    embedding = image_features.cpu().numpy().tolist()[0]
                
                with torch.no_grad():
                    # 1. Ask CLIP for the image features
                    image_features = model.get_image_features(pixel_values=inputs.pixel_values)
                    
                    # 2. SAFETY CATCH: If it returned a Python object instead of a raw tensor, unwrap it
                    if not isinstance(image_features, torch.Tensor):
                        if hasattr(image_features, 'pooler_output'):
                            image_features = image_features.pooler_output
                        elif hasattr(image_features, 'image_embeds'):
                            image_features = image_features.image_embeds
                        else:
                            image_features = image_features[0]
                            
                    # 3. Convert back to float32 for normal math and saving
                    image_features = image_features.to(torch.float32)
                    image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
                    embedding = image_features.cpu().numpy().tolist()[0]
                doc_id = f"{filename}_{timestamp_sec:.2f}"
                collection.add(
                    embeddings=[embedding],
                    metadatas=[{"filename": filename, "timestamp": timestamp_sec}],
                    ids=[doc_id]
                )
                
            current_frame += 1
            pbar.update(1)
            
    cap.release()

if not os.path.exists(VIDEO_FOLDER):
    os.makedirs(VIDEO_FOLDER)
    print(f"Created folder '{VIDEO_FOLDER}'. Please put a test video in there!")
else:
    valid_extensions = ('.mp4', '.mov', '.avi', '.mkv')
    video_files = [f for f in os.listdir(VIDEO_FOLDER) if f.lower().endswith(valid_extensions)]
    
    if not video_files:
        print(f"No videos found in '{VIDEO_FOLDER}'. Add some and run again.")
    else:
        for vf in video_files:
            process_video(os.path.join(VIDEO_FOLDER, vf))
        print("\nIngestion Complete! Your AI Knowledge Base is ready.")