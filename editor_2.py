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
MUSIC_FILE = "music.mp3"

CLIP_DURATION = 4.0      # Length of each raw clip
FADE_DURATION = 1.0      # How long the crossfade lasts
MIN_GAP_SECONDS = 10.0   # Minimum distance between clips from the same video

print("Loading AI Editor Modules...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "openai/clip-vit-base-patch32"

processor = CLIPProcessor.from_pretrained(model_id)
model = CLIPModel.from_pretrained(model_id, torch_dtype=torch.float16).to(device)

chroma_client = chromadb.PersistentClient(path=DB_FOLDER)
collection = chroma_client.get_collection(name="drone_footage")

def create_pro_montage(prompt, num_clips=4):
    print(f"\n🎬 Searching Knowledge Base for: '{prompt}'...")
    inputs = processor(text=[prompt], return_tensors="pt", padding=True).to(device)
    
    with torch.no_grad():
        text_outputs = model.text_model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
        pooled_output = text_outputs.pooler_output
        text_features = model.text_projection(pooled_output)
        text_features = text_features.to(torch.float32)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        embedding = text_features.cpu().numpy().flatten().tolist()
    
    # FIX 1: Fetch way more results than we need (20) to allow for filtering
    results = collection.query(query_embeddings=[embedding], n_results=20)
    
    selected_clips = []
    print("\n🧠 Applying Diversity Filter (Rejecting overlaps)...")
    
    for i in range(len(results['ids'][0])):
        meta = results['metadatas'][0][i]
        
        # Check if this clip is too close to one we already picked
        is_too_similar = False
        for sc in selected_clips:
            if sc['filename'] == meta['filename'] and abs(sc['timestamp'] - meta['timestamp']) < MIN_GAP_SECONDS:
                is_too_similar = True
                break
                
        if not is_too_similar:
            selected_clips.append(meta)
            print(f"   + Selected: {meta['filename']} at {int(meta['timestamp'])}s")
            
        if len(selected_clips) == num_clips:
            break

    if len(selected_clips) < num_clips:
        print(f"⚠️ Could only find {len(selected_clips)} diverse clips for this prompt.")
        num_clips = len(selected_clips)
        if num_clips == 0: return

    # Setup temp folder
    temp_dir = "./temp_render"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    
    print("\n✂️ Extracting clips, forcing 1080p/30fps, and applying LUT...")
    
    for i in range(num_clips):
        meta = selected_clips[i]
        start_time = max(0, meta['timestamp'] - 1.5)
        input_path = os.path.join(VIDEO_FOLDER, meta['filename'])
        output_clip = os.path.join(temp_dir, f"clip_{i}.mp4")
        
        # FIX 3: Force uniform resolution/fps so transitions don't crash, and strip drone audio (-an)
        # Using scale+pad ensures vertical/weird aspect ratios get letterboxed perfectly
        video_filters = f"fps=30,scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,lut3d={LUT_FILE}"
        
        ffmpeg_cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-ss", str(start_time), "-i", input_path,
            "-t", str(CLIP_DURATION),
            "-vf", video_filters,
            "-an", # Remove terrible drone audio
            "-c:v", "libx264", "-preset", "fast", "-crf", "22",
            output_clip
        ]
        subprocess.run(ffmpeg_cmd, check=True)

    print("\n🧵 Generating Complex Filtergraph for Smooth Transitions...")
    
    # FIX 2: Dynamically build the xfade filtergraph based on how many clips we have
    final_output = f"{prompt.replace(' ', '_')}_cinematic.mp4"
    total_video_length = CLIP_DURATION + (num_clips - 1) * (CLIP_DURATION - FADE_DURATION)
    
    concat_cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    
    # 1. Add all video inputs
    for i in range(num_clips):
        concat_cmd.extend(["-i", os.path.join(temp_dir, f"clip_{i}.mp4")])
        
    # 2. Add music input if it exists
    has_music = os.path.exists(MUSIC_FILE)
    if has_music:
        concat_cmd.extend(["-i", MUSIC_FILE])

    # 3. Build the transition string
    if num_clips > 1:
        filter_chains = []
        offset = CLIP_DURATION - FADE_DURATION
        # First crossfade between clip 0 and 1
        filter_chains.append(f"[0:v][1:v]xfade=transition=fade:duration={FADE_DURATION}:offset={offset}[v1]")
        
        # Chain the rest
        for i in range(2, num_clips):
            offset = i * (CLIP_DURATION - FADE_DURATION)
            filter_chains.append(f"[v{i-1}][{i}:v]xfade=transition=fade:duration={FADE_DURATION}:offset={offset}[v{i}]")
            
        video_map = f"[v{num_clips-1}]"
        filter_complex = "; ".join(filter_chains)
    else:
        video_map = "[0:v]"
        filter_complex = "null" # Dummy filter if only 1 clip

    # Audio fading logic (Fade in at 0, Fade out at end)
    if has_music:
        audio_idx = num_clips
        fade_out_start = total_video_length - FADE_DURATION
        audio_filter = f"[{audio_idx}:a]afade=t=in:ss=0:d={FADE_DURATION},afade=t=out:st={fade_out_start}:d={FADE_DURATION}[aout]"
        if filter_complex != "null":
            filter_complex += f"; {audio_filter}"
        else:
            filter_complex = audio_filter
        audio_map = "[aout]"
    else:
        audio_map = None

    # Assemble the final command
    if filter_complex != "null":
        concat_cmd.extend(["-filter_complex", filter_complex])
        concat_cmd.extend(["-map", video_map])
    else:
        concat_cmd.extend(["-map", "0:v"])
        
    if audio_map:
        concat_cmd.extend(["-map", audio_map])

    # Cap the final video length to the exact visual timeline and encode
    concat_cmd.extend([
        "-t", str(total_video_length),
        "-c:v", "libx264", "-preset", "medium", "-crf", "20",
        final_output
    ])
    
    subprocess.run(concat_cmd, check=True)
    
    shutil.rmtree(temp_dir)
    print(f"\n✅ MASTERPIECE COMPLETE! Saved as: {final_output}")

if __name__ == "__main__":
    if not os.path.exists(LUT_FILE):
        print(f"⚠️ ERROR: Could not find '{LUT_FILE}'.")
    else:
        if not os.path.exists(MUSIC_FILE):
            print("💡 TIP: Add a 'music.mp3' to this folder to automatically add a cinematic soundtrack!\n")
            
        user_prompt = input("🎥 What kind of cinematic montage do you want? (e.g., 'trees'): ")
        create_pro_montage(user_prompt, num_clips=4) # Asking for 4 clips now to show off the transitions!