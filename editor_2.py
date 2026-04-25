import os
import json
import subprocess
import shutil
import cv2
import random
from skimage.metrics import structural_similarity as ssim

# --- SILENCE HUGGING FACE WARNINGS ---
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

# --- AMD ROCm 6700 XT STABILITY FIXES ---
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
os.environ["MIOPEN_DISABLE_CACHE"] = "1"
os.environ["PYTORCH_HIP_ALLOC_CONF"] = "garbage_collection_threshold:0.8,max_split_size_mb:512"

# hipBLASLt is an optimized linear algebra backend for enterprise cards. 
# 6700xt architecture doesn't support it, so it falls back to standard hipblas
warnings.filterwarnings("ignore", message=".*Attempting to use hipBLASLt.*")

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

# Fallback for old configs
video_folders = config.get('video_folders', [config.get('video_folder', './my_drone_videos')])
music_folders = config.get('music_folders', [])

# --- HELPER FUNCTIONS ---
def find_video_path(filename):
    """Scours all provided video folders to find where the video actually lives."""
    for folder in video_folders:
        path = os.path.join(folder, filename)
        if os.path.exists(path):
            return path
    return None

def get_random_music():
    """Finds all audio files across all music folders and picks one randomly."""
    valid_exts = ('.mp3', '.wav', '.m4a')
    all_music = []
    for folder in music_folders:
        if os.path.exists(folder):
            for f in os.listdir(folder):
                if f.lower().endswith(valid_exts):
                    all_music.append(os.path.join(folder, f))
    
    if all_music:
        chosen = random.choice(all_music)
        print(f"🎵 Randomly selected track: {os.path.basename(chosen)}")
        return chosen
    return None

def get_frame_at_time(video_path, time_sec):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, int(time_sec * 1000))
    ret, frame = cap.read()
    cap.release()
    if ret:
        frame = cv2.resize(frame, (320, 180))
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return None

def check_jump_cut(clip1_meta, clip2_meta, clip_dur):
    end_time_1 = clip1_meta['timestamp'] + clip_dur
    start_time_2 = clip2_meta['timestamp']
    
    path1 = find_video_path(clip1_meta['filename'])
    path2 = find_video_path(clip2_meta['filename'])
    
    if not path1 or not path2: return False
    
    frame1 = get_frame_at_time(path1, end_time_1)
    frame2 = get_frame_at_time(path2, start_time_2)
    
    if frame1 is None or frame2 is None: return False
        
    score, _ = ssim(frame1, frame2, full=True)
    return score > config['max_similarity_score']

# --- INITIALIZATION ---
print("Loading AI Models...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "openai/clip-vit-base-patch32"

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
    # 1. Handle Music & Beat Sync Logic
    chosen_music = get_random_music()
    dynamic_clip_dur = config['clip_duration']
    dynamic_fade_dur = config['fade_duration']

    if config.get('sync_to_beats') and chosen_music:
        try:
            import librosa
            import numpy as np
            print("🎧 Analyzing music tempo for beat synchronization...")
            # Load just the first 30 seconds to calculate BPM instantly
            y, sr = librosa.load(chosen_music, duration=30)
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            bpm = tempo[0] if isinstance(tempo, np.ndarray) else tempo
            
            if bpm > 0:
                beat_length = 60.0 / bpm
                # Find how many beats fit closest to the user's desired clip duration
                beats_per_clip = max(1, round(dynamic_clip_dur / beat_length))
                
                # Math: Force clip to exactly match the phrasing, and fade to exactly 1 beat
                dynamic_clip_dur = beats_per_clip * beat_length
                dynamic_fade_dur = beat_length
                
                print(f"   -> Detected {bpm:.1f} BPM. Snapping clips to {dynamic_clip_dur:.2f}s (exactly {beats_per_clip} beats).")
        except Exception as e:
            print(f"⚠️ Could not analyze beats, using default duration. (Error: {e})")

    # 2. Semantic Search
    print(f"\n🎬 Searching Knowledge Base for: '{prompt}'...")
    inputs = processor(text=[prompt], return_tensors="pt", padding=True).to(device)
    
    with torch.no_grad():
        text_outputs = model.text_model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
        pooled_output = text_outputs.pooler_output
        text_features = model.text_projection(pooled_output)
        text_features = text_features.to(torch.float32)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        embedding = text_features.cpu().numpy().flatten().tolist()
    
    results = collection.query(query_embeddings=[embedding], n_results=30)
    
    selected_clips = []
    print("\n🧠 Applying Multi-Stage Intelligence Filters...")
    
    for i in range(len(results['ids'][0])):
        meta = results['metadatas'][0][i]
        
        # Verify file actually exists in one of the folders
        if not find_video_path(meta['filename']):
            continue

        is_too_close = False
        for sc in selected_clips:
            if sc['filename'] == meta['filename'] and abs(sc['timestamp'] - meta['timestamp']) < config['min_gap_seconds']:
                is_too_close = True
                break
        if is_too_close: continue
            
        if len(selected_clips) > 0:
            last_clip = selected_clips[-1]
            if check_jump_cut(last_clip, meta, dynamic_clip_dur):
                print(f"   - Rejected {meta['filename']} at {int(meta['timestamp'])}s (Failed SSIM: Jump Cut detected)")
                continue
                
        selected_clips.append(meta)
        print(f"   + Selected: {meta['filename']} at {int(meta['timestamp'])}s")
            
        if len(selected_clips) == config['num_clips_to_generate']:
            break

    num_clips = len(selected_clips)
    if num_clips < 2:
        print("⚠️ Could not find enough diverse clips. Try a simpler prompt.")
        return

    temp_dir = "./temp_render"
    if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    
    print("\n✂️ Extracting clips, forcing 1080p/30fps, and applying LUT...")
    for i in range(num_clips):
        meta = selected_clips[i]
        start_time = max(0, meta['timestamp'] - (dynamic_fade_dur / 2)) # Pad start for smooth fade
        input_path = find_video_path(meta['filename'])
        output_clip = os.path.join(temp_dir, f"clip_{i}.mp4")
        
        video_filters = f"fps=30,scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2"
        if os.path.exists(config['lut_file']):
            video_filters += f",lut3d={config['lut_file']}"
        
        ffmpeg_cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-ss", str(start_time), "-i", input_path,
            "-t", str(dynamic_clip_dur),
            "-vf", video_filters,
            "-an", 
            "-c:v", "libx264", "-preset", "fast", "-crf", "22",
            output_clip
        ]
        subprocess.run(ffmpeg_cmd, check=True)

    print("\n🧵 Generating Filtergraph for Smooth Transitions...")
    final_output = f"{prompt.replace(' ', '_')}_cinematic.mp4"
    
    total_video_length = dynamic_clip_dur + (num_clips - 1) * (dynamic_clip_dur - dynamic_fade_dur)
    
    concat_cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    for i in range(num_clips):
        concat_cmd.extend(["-i", os.path.join(temp_dir, f"clip_{i}.mp4")])
        
    if chosen_music:
        concat_cmd.extend(["-i", chosen_music])

    filter_chains = []
    offset = dynamic_clip_dur - dynamic_fade_dur
    filter_chains.append(f"[0:v][1:v]xfade=transition=fade:duration={dynamic_fade_dur}:offset={offset}[v1]")
    
    for i in range(2, num_clips):
        offset = i * (dynamic_clip_dur - dynamic_fade_dur)
        filter_chains.append(f"[v{i-1}][{i}:v]xfade=transition=fade:duration={dynamic_fade_dur}:offset={offset}[v{i}]")
        
    video_map = f"[v{num_clips-1}]"
    filter_complex = "; ".join(filter_chains)

    if chosen_music:
        audio_idx = num_clips
        fade_out_start = total_video_length - dynamic_fade_dur
        audio_filter = f"[{audio_idx}:a]afade=t=in:ss=0:d={dynamic_fade_dur},afade=t=out:st={fade_out_start}:d={dynamic_fade_dur}[aout]"
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