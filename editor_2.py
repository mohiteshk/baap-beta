import os
import json
import subprocess
import shutil
import cv2
from skimage.metrics import structural_similarity as ssim

# --- SILENCE HUGGING FACE WARNINGS ---
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

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
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

# --- LOAD CONFIGURATION ---
CONFIG_FILE = "config.json"
if not os.path.exists(CONFIG_FILE):
    raise FileNotFoundError(f"Missing {CONFIG_FILE}. Please create it first.")

with open(CONFIG_FILE, 'r') as f:
    config = json.load(f)

# --- HELPER: FRAME EXTRACTOR FOR SSIM ---
def get_frame_at_time(video_path, time_sec):
    cap = cv2.VideoCapture(video_path)
    # Fast forward to exact millisecond
    cap.set(cv2.CAP_PROP_POS_MSEC, int(time_sec * 1000))
    ret, frame = cap.read()
    cap.release()
    if ret:
        # Resize to 320x180 for blazing fast SSIM math, and convert to Grayscale
        frame = cv2.resize(frame, (320, 180))
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return None

def check_jump_cut(clip1_meta, clip2_meta, config):
    """Returns True if the transition between these two clips is too visually similar."""
    # We compare the END of Clip 1 with the START of Clip 2
    end_time_1 = clip1_meta['timestamp'] + config['clip_duration']
    start_time_2 = clip2_meta['timestamp']
    
    path1 = os.path.join(config['video_folder'], clip1_meta['filename'])
    path2 = os.path.join(config['video_folder'], clip2_meta['filename'])
    
    frame1 = get_frame_at_time(path1, end_time_1)
    frame2 = get_frame_at_time(path2, start_time_2)
    
    if frame1 is None or frame2 is None:
        return False # Fallback if read fails
        
    score, _ = ssim(frame1, frame2, full=True)
    return score > config['max_similarity_score']

# --- INITIALIZATION ---
print("Loading AI Models...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "openai/clip-vit-base-patch32"

# Safe loading: Force local cache. If it fails, download once.
try:
    processor = CLIPProcessor.from_pretrained(model_id, local_files_only=True)
    model = CLIPModel.from_pretrained(model_id, torch_dtype=torch.float16, local_files_only=True).to(device)
    print("✅ Model loaded instantly from local disk cache.")
except Exception:
    print("⚠️ Local cache not found. Downloading model... (This will only happen once)")
    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id, torch_dtype=torch.float16).to(device)

chroma_client = chromadb.PersistentClient(path=config['db_folder'])
collection = chroma_client.get_collection(name="drone_footage")

def create_master_montage(prompt):
    print(f"\n🎬 Searching Knowledge Base for: '{prompt}'...")
    inputs = processor(text=[prompt], return_tensors="pt", padding=True).to(device)
    
    with torch.no_grad():
        text_outputs = model.text_model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
        pooled_output = text_outputs.pooler_output
        text_features = model.text_projection(pooled_output)
        text_features = text_features.to(torch.float32)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        embedding = text_features.cpu().numpy().flatten().tolist()
    
    # Fetch top 30 to give our heavy filters room to breathe
    results = collection.query(query_embeddings=[embedding], n_results=30)
    
    selected_clips = []
    print("\n🧠 Applying Multi-Stage Intelligence Filters...")
    
    for i in range(len(results['ids'][0])):
        meta = results['metadatas'][0][i]
        
        # 1. Temporal Filter: Reject if too close to an already selected clip from the same video
        is_too_close = False
        for sc in selected_clips:
            if sc['filename'] == meta['filename'] and abs(sc['timestamp'] - meta['timestamp']) < config['min_gap_seconds']:
                is_too_close = True
                break
        if is_too_close:
            continue
            
        # 2. Structural Filter (SSIM): Reject jump cuts with the immediately preceding clip
        if len(selected_clips) > 0:
            last_clip = selected_clips[-1]
            if check_jump_cut(last_clip, meta, config):
                print(f"   - Rejected {meta['filename']} at {int(meta['timestamp'])}s (Failed SSIM: Jump Cut detected)")
                continue
                
        # If it passes both filters, it makes the cut!
        selected_clips.append(meta)
        print(f"   + Selected: {meta['filename']} at {int(meta['timestamp'])}s")
            
        if len(selected_clips) == config['num_clips_to_generate']:
            break

    num_clips = len(selected_clips)
    if num_clips < 2:
        print("⚠️ Could not find enough diverse clips. Try a simpler prompt or lower the max_similarity_score in config.")
        return

    temp_dir = "./temp_render"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    
    print("\n✂️ Extracting clips, forcing 1080p/30fps, and applying LUT...")
    for i in range(num_clips):
        meta = selected_clips[i]
        start_time = max(0, meta['timestamp'] - 1.5)
        input_path = os.path.join(config['video_folder'], meta['filename'])
        output_clip = os.path.join(temp_dir, f"clip_{i}.mp4")
        
        video_filters = f"fps=30,scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2"
        if os.path.exists(config['lut_file']):
            video_filters += f",lut3d={config['lut_file']}"
        
        ffmpeg_cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-ss", str(start_time), "-i", input_path,
            "-t", str(config['clip_duration']),
            "-vf", video_filters,
            "-an", 
            "-c:v", "libx264", "-preset", "fast", "-crf", "22",
            output_clip
        ]
        subprocess.run(ffmpeg_cmd, check=True)

    print("\n🧵 Generating Filtergraph for Smooth Transitions...")
    final_output = f"{prompt.replace(' ', '_')}_cinematic.mp4"
    fade = config['fade_duration']
    clip_dur = config['clip_duration']
    total_video_length = clip_dur + (num_clips - 1) * (clip_dur - fade)
    
    concat_cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    
    for i in range(num_clips):
        concat_cmd.extend(["-i", os.path.join(temp_dir, f"clip_{i}.mp4")])
        
    has_music = os.path.exists(config['music_file'])
    if has_music:
        concat_cmd.extend(["-i", config['music_file']])

    filter_chains = []
    offset = clip_dur - fade
    filter_chains.append(f"[0:v][1:v]xfade=transition=fade:duration={fade}:offset={offset}[v1]")
    
    for i in range(2, num_clips):
        offset = i * (clip_dur - fade)
        filter_chains.append(f"[v{i-1}][{i}:v]xfade=transition=fade:duration={fade}:offset={offset}[v{i}]")
        
    video_map = f"[v{num_clips-1}]"
    filter_complex = "; ".join(filter_chains)

    if has_music:
        audio_idx = num_clips
        fade_out_start = total_video_length - fade
        audio_filter = f"[{audio_idx}:a]afade=t=in:ss=0:d={fade},afade=t=out:st={fade_out_start}:d={fade}[aout]"
        filter_complex += f"; {audio_filter}"
        audio_map = "[aout]"
    else:
        audio_map = None

    concat_cmd.extend(["-filter_complex", filter_complex, "-map", video_map])
    if audio_map:
        concat_cmd.extend(["-map", audio_map])

    concat_cmd.extend([
        "-t", str(total_video_length),
        "-c:v", "libx264", "-preset", "medium", "-crf", "20",
        final_output
    ])
    
    subprocess.run(concat_cmd, check=True)
    shutil.rmtree(temp_dir)
    print(f"\n✅ MASTERPIECE COMPLETE! Saved as: {final_output}")

if __name__ == "__main__":
    user_prompt = input("🎥 What kind of cinematic montage do you want? (e.g., 'trees'): ")
    if user_prompt.strip():
        create_master_montage(user_prompt)