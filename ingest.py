import os
import json
import cv2
from tqdm import tqdm

# --- SILENCE HUGGING FACE WARNINGS ---
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

# --- AMD ROCm 6700 XT STABILITY FIXES ---
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
os.environ["MIOPEN_DISABLE_CACHE"] = "1"
os.environ["PYTORCH_HIP_ALLOC_CONF"] = "garbage_collection_threshold:0.8,max_split_size_mb:512"

import cv2
cv2.setNumThreads(0) 

import torch
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
torch.backends.cudnn.enabled = False

import chromadb
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

# --- LOAD CONFIGURATION ---
CONFIG_FILE = "config.json"
if not os.path.exists(CONFIG_FILE):
    raise FileNotFoundError(f"Missing {CONFIG_FILE}. Please create it first.")

with open(CONFIG_FILE, 'r') as f:
    config = json.load(f)

video_folders = config.get('video_folders', ['./my_drone_videos'])
fps_to_extract = config.get('fps_to_extract', 1)
BATCH_SIZE = config.get('batch_size', 32) # The Turbo Charger

# --- INITIALIZATION ---
print("Loading AI Models...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "openai/clip-vit-base-patch32"

try:
    processor = CLIPProcessor.from_pretrained(model_id, local_files_only=True)
    model = CLIPModel.from_pretrained(model_id, torch_dtype=torch.float16, local_files_only=True).to(device)
    print("✅ Model loaded instantly from local disk cache.")
except Exception:
    print("⚠️ Local cache not found. Downloading model...")
    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id, torch_dtype=torch.float16).to(device)

print("Connecting to Knowledge Base...")
chroma_client = chromadb.PersistentClient(path=config['db_folder'])
collection = chroma_client.get_or_create_collection(
    name="drone_footage",
    metadata={"hnsw:space": "cosine"}
)

def process_batch(frames, metadatas):
    """Feeds a massive block of frames to the AMD GPU all at once."""
    if not frames: return
    
    # 1. Process all images into one massive tensor block
    inputs = processor(images=frames, return_tensors="pt").to(device, torch.float16)
    
    with torch.no_grad():
        # 2. GPU crunches all 32 frames simultaneously
        image_features = model.get_image_features(pixel_values=inputs.pixel_values)
        
        if not isinstance(image_features, torch.Tensor):
            if hasattr(image_features, 'pooler_output'):
                image_features = image_features.pooler_output
            elif hasattr(image_features, 'image_embeds'):
                image_features = image_features.image_embeds
            else:
                image_features = image_features[0]
                
        # 3. Math and normalization
        image_features = image_features.to(torch.float32)
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        embeddings = image_features.cpu().numpy().tolist()
    
    # 4. Save entire batch to ChromaDB instantly
    ids = [f"{m['filename']}_{m['timestamp']:.2f}" for m in metadatas]
    collection.add(
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )

def process_video(video_path):
    filename = os.path.basename(video_path)
    
    existing_records = collection.get(where={"filename": filename}, limit=1)
    if len(existing_records['ids']) > 0:
        print(f"⏭️  Skipping: {filename} (Already in Knowledge Base)")
        return

    print(f"\nProcessing: {filename}")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if fps == 0 or total_frames == 0:
        print(f"⚠️ Skipping {filename} - couldn't read video properties.")
        return

    frame_interval = int(fps / fps_to_extract) if fps_to_extract > 0 else int(fps)
    if frame_interval < 1: frame_interval = 1
    
    current_frame = 0
    frame_buffer = []
    meta_buffer = []
    
    with tqdm(total=total_frames, desc=filename) as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            if current_frame % frame_interval == 0:
                timestamp_sec = current_frame / fps
                
                # Convert to RGB and keep it in RAM
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_frame)
                
                frame_buffer.append(pil_img)
                meta_buffer.append({"filename": filename, "timestamp": timestamp_sec})
                
                # If we hit 32 frames, unleash the GPU!
                if len(frame_buffer) >= BATCH_SIZE:
                    process_batch(frame_buffer, meta_buffer)
                    frame_buffer.clear()
                    meta_buffer.clear()
                
            current_frame += 1
            pbar.update(1)
            
    # Clean up any leftover frames in the buffer at the end of the video
    if len(frame_buffer) > 0:
        process_batch(frame_buffer, meta_buffer)
        
    cap.release()

if __name__ == "__main__":
    valid_extensions = ('.mp4', '.mov', '.avi', '.mkv')
    videos_found = 0
    
    for folder in video_folders:
        if not os.path.exists(folder):
            print(f"⚠️ Folder not found: {folder}")
            continue
            
        print(f"\n📂 Scanning folder: {folder}")
        video_files = [f for f in os.listdir(folder) if f.lower().endswith(valid_extensions)]
        
        for vf in video_files:
            videos_found += 1
            process_video(os.path.join(folder, vf))
            
    if videos_found == 0:
        print("\nNo videos found in any of your configured folders!")
    else:
        print("\n✅ Knowledge Base Sync Complete!")