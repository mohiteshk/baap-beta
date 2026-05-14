# MUST BE IMPORTED FIRST
from core.env_setup import configure_pytorch

import os
import shutil
import subprocess
import random
from core.config import config
from core.model import VisionTextModel
from core.database import get_chroma_collection
from core.video_utils import (
    find_video_path, get_random_music, check_jump_cut
)
#from core.video_utils import is_physically_smooth

device = configure_pytorch()
ai_model = VisionTextModel(device)
collection = get_chroma_collection()

def create_master_montage(prompt):
    chosen_music = get_random_music()
    
    # Base Configs
    min_dur = config.get('clip_duration_min', 2.5)
    max_dur = config.get('clip_duration_max', 6.0)
    target_clips = config.get('num_clips_to_generate', 4)
    match_threshold = config.get('match_score_threshold', 0.35)
    
    beat_length = None
    global_fade = config.get('fade_duration', 1.0)

    if config.get('sync_to_beats') and chosen_music:
        try:
            import librosa
            import numpy as np
            print("🎧 Analyzing music tempo for beat synchronization...")
            y, sr = librosa.load(chosen_music, duration=30)
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            bpm = tempo[0] if isinstance(tempo, np.ndarray) else tempo
            
            if bpm > 0:
                beat_length = 60.0 / bpm
                global_fade = beat_length # Lock fades to exactly 1 beat
                print(f"   -> Detected {bpm:.1f} BPM. Base beat length is {beat_length:.2f}s.")
        except Exception as e:
            print(f"⚠️ Could not analyze beats, using default durations. (Error: {e})")

    print(f"\n🎬 Searching Knowledge Base for: '{prompt}'...")
    embedding = ai_model.get_text_embedding(prompt)
    
    # Request a dynamic pool size to ensure we hit our target after filtering
    search_pool = config.get('search_pool_size', 100)
    results = collection.query(query_embeddings=[embedding], n_results=search_pool)
    
    selected_clips = []
    print(f"\n🧠 Applying Multi-Stage Filters (Target: {target_clips} clips)...")
    
    for i in range(len(results['ids'][0])):
        meta = results['metadatas'][0][i]
        distance = results['distances'][0][i]
        
        # 1. Relevance Check
        if distance > match_threshold:
            print(f"   - Rejected {meta['filename']} (Match Score {distance:.2f} > Threshold {match_threshold})")
            continue
            
        input_path = find_video_path(meta['filename'])
        if not input_path: continue

        # Determine target duration for this specific clip
        raw_dur = random.uniform(min_dur, max_dur)
        if beat_length:
            beats_per_clip = max(1, round(raw_dur / beat_length))
            clip_dur = beats_per_clip * beat_length
        else:
            clip_dur = raw_dur

        # 2. Gap Check
        if any(sc['filename'] == meta['filename'] and abs(sc['timestamp'] - meta['timestamp']) < config['min_gap_seconds'] for sc in selected_clips):
            continue

        # 3. Jump Cut Check
        if selected_clips and check_jump_cut(selected_clips[-1], meta, clip_dur):
            print(f"   - Rejected {meta['filename']} at {int(meta['timestamp'])}s (Failed SSIM: Jump Cut)")
            continue
                
        # Store the clip AND its unique calculated duration
        meta['assigned_duration'] = clip_dur
        selected_clips.append(meta)
        
        print(f"   + Selected: {meta['filename']} at {int(meta['timestamp'])}s (Len: {clip_dur:.1f}s, Score: {distance:.2f})")
        
        if len(selected_clips) == target_clips:
            break

    num_clips = len(selected_clips)
    if num_clips < 2:
        print(f"⚠️ Only found {num_clips} valid clips. Try raising your match_score_threshold or lowering turbulence_threshold.")
        return

    temp_dir = config.get("temp_render_dir", "./temp_render")
    if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    
    print("\n✂️ Extracting clips and applying LUT...")
    for i, meta in enumerate(selected_clips):
        start_time = max(0, meta['timestamp'] - (global_fade / 2))
        input_path = find_video_path(meta['filename'])
        output_clip = os.path.join(temp_dir, f"clip_{i}.mp4")
        
        video_filters = f"fps=30,scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2"
        if os.path.exists(config.get('lut_file', '')):
            video_filters += f",lut3d={config['lut_file']}"
        
        ffmpeg_cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-ss", str(start_time), "-i", input_path, "-t", str(meta['assigned_duration']),
            "-vf", video_filters, "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "22", output_clip
        ]
        subprocess.run(ffmpeg_cmd, check=True)

    print("\n🧵 Generating Dynamic Filtergraph for Variable Length Transitions...")
    final_output = f"{prompt.replace(' ', '_')}_cinematic.mp4"
    
    concat_cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    for i in range(num_clips): 
        concat_cmd.extend(["-i", os.path.join(temp_dir, f"clip_{i}.mp4")])
    if chosen_music: 
        concat_cmd.extend(["-i", chosen_music])

    filter_chains = []
    
    # Calculate cumulative offsets for xfade
    current_timeline_length = selected_clips[0]['assigned_duration']
    
    # First transition
    offset = current_timeline_length - global_fade
    filter_chains.append(f"[0:v][1:v]xfade=transition=fade:duration={global_fade}:offset={offset}[v1]")
    current_timeline_length += (selected_clips[1]['assigned_duration'] - global_fade)
    
    # Subsequent transitions
    for i in range(2, num_clips):
        offset = current_timeline_length - global_fade
        filter_chains.append(f"[v{i-1}][{i}:v]xfade=transition=fade:duration={global_fade}:offset={offset}[v{i}]")
        current_timeline_length += (selected_clips[i]['assigned_duration'] - global_fade)
        
    video_map = f"[v{num_clips-1}]"
    filter_complex = "; ".join(filter_chains)

    audio_map = None
    if chosen_music:
        audio_idx = num_clips
        fade_out_start = current_timeline_length - global_fade
        filter_complex += f"; [{audio_idx}:a]afade=t=in:ss=0:d={global_fade},afade=t=out:st={fade_out_start}:d={global_fade}[aout]"
        audio_map = "[aout]"

    concat_cmd.extend(["-filter_complex", filter_complex, "-map", video_map])
    if audio_map: concat_cmd.extend(["-map", audio_map])
    concat_cmd.extend(["-t", str(current_timeline_length), "-c:v", "libx264", "-preset", "medium", "-crf", "20", final_output])
    
    subprocess.run(concat_cmd, check=True)
    shutil.rmtree(temp_dir)
    print(f"\n✅ MASTERPIECE COMPLETE! Total Length: {current_timeline_length:.1f}s. Saved as: {final_output}")

if __name__ == "__main__":
    user_prompt = input("🎥 What kind of cinematic montage do you want? (e.g., 'trees'): ")
    if user_prompt.strip():
        create_master_montage(user_prompt)