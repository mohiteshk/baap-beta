import os
import subprocess
import shutil

# --- AMD ROCm 6700 XT STABILITY FIXES ---
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
os.environ["MIOPEN_DISABLE_CACHE"] = "1"
os.environ["PYTORCH_HIP_ALLOC_CONF"] = "garbage_collection_threshold:0.8,max_split_size_mb:512"

import torch
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
torch.backends.cudnn.enabled = False

import chromadb
from transformers import CLIPProcessor, CLIPModel

# --- CONFIGURATION ---
VIDEO_FOLDER = "./my_drone_videos"
DB_FOLDER = "./chroma_db"
LUT_FILE = "dji.cube"
CLIP_DURATION = 4.0 # Make each clip 4 seconds long

print("Loading AI Editor Modules...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "openai/clip-vit-base-patch32"

processor = CLIPProcessor.from_pretrained(model_id)
model = CLIPModel.from_pretrained(model_id, torch_dtype=torch.float16).to(device)

chroma_client = chromadb.PersistentClient(path=DB_FOLDER)
collection = chroma_client.get_collection(name="drone_footage")

def create_montage(prompt, num_clips=3):
    print(f"\n🎬 Searching for: '{prompt}'...")
    inputs = processor(text=[prompt], return_tensors="pt", padding=True).to(device)
    
    with torch.no_grad():
        text_outputs = model.text_model(
            input_ids=inputs.input_ids, 
            attention_mask=inputs.attention_mask
        )
        pooled_output = text_outputs.pooler_output
        text_features = model.text_projection(pooled_output)
        
        text_features = text_features.to(torch.float32)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        embedding = text_features.cpu().numpy().flatten().tolist()
    
    results = collection.query(query_embeddings=[embedding], n_results=num_clips)
    
    # Setup temporary folder for our cut clips
    temp_dir = "./temp_render"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    
    concat_list_path = os.path.join(temp_dir, "concat.txt")
    
    print("\n✂️ Cutting clips and applying LUT...")
    
    with open(concat_list_path, "w") as f:
        for i in range(num_clips):
            meta = results['metadatas'][0][i]
            video_file = meta['filename']
            timestamp = meta['timestamp']
            
            # Start the clip 1.5 seconds before the exact AI match
            start_time = max(0, timestamp - 1.5)
            
            input_path = os.path.join(VIDEO_FOLDER, video_file)
            output_clip = os.path.join(temp_dir, f"clip_{i}.mp4")
            
            # FFmpeg Command: Cut clip, apply LUT, re-encode audio/video
            ffmpeg_cmd = [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-ss", str(start_time),
                "-i", input_path,
                "-t", str(CLIP_DURATION),
                "-vf", f"lut3d={LUT_FILE}",
                "-c:v", "libx264", "-preset", "fast", "-crf", "22",
                "-c:a", "aac",
                output_clip
            ]
            
            print(f"   -> Processing Clip {i+1} from {video_file}...")
            subprocess.run(ffmpeg_cmd, check=True)
            
            # Add to concat file for final stitching
            # FFmpeg requires a specific format: file 'path/to/file.mp4'
            f.write(f"file 'clip_{i}.mp4'\n")
            
    print("\n🧵 Stitching clips together...")
    final_output = f"{prompt.replace(' ', '_')}_montage.mp4"
    
    concat_cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "concat", "-safe", "0",
        "-i", concat_list_path,
        "-c", "copy",
        final_output
    ]
    subprocess.run(concat_cmd, check=True)
    
    # Cleanup
    shutil.rmtree(temp_dir)
    print(f"\n✅ DONE! Your video is ready: {final_output}")

if __name__ == "__main__":
    if not os.path.exists(LUT_FILE):
        print(f"⚠️ ERROR: Could not find '{LUT_FILE}' in the current folder.")
        print("Please copy your DJI .cube file here and name it 'dji.cube'.")
    else:
        user_prompt = input("\n🎥 What kind of montage do you want to make? (e.g., 'trees'): ")
        create_montage(user_prompt, num_clips=3)