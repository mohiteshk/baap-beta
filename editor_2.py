# MUST BE IMPORTED FIRST
from core.env_setup import configure_pytorch

# Now import the rest
import os
import shutil
import subprocess
from core.config import config
from core.model import VisionTextModel
from core.database import get_chroma_collection
from core.video_utils import (
    find_video_path, get_random_music, is_smooth_clip, check_jump_cut
)

device = configure_pytorch()
model = VisionTextModel(device)
collection = get_chroma_collection()

def create_master_montage(prompt):
    chosen_music = get_random_music()
    dynamic_clip_dur = config['clip_duration']
    dynamic_fade_dur = config['fade_duration']

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
                beats_per_clip = max(1, round(dynamic_clip_dur / beat_length))
                dynamic_clip_dur = beats_per_clip * beat_length
                dynamic_fade_dur = beat_length
                print(f"   -> Detected {bpm:.1f} BPM. Snapping clips to {dynamic_clip_dur:.2f}s (exactly {beats_per_clip} beats).")
        except Exception as e:
            print(f"⚠️ Could not analyze beats, using default duration. (Error: {e})")

    print(f"\n🎬 Searching Knowledge Base for: '{prompt}'...")
    embedding = model.get_text_embedding(prompt)
    results = collection.query(query_embeddings=[embedding], n_results=30)
    
    selected_clips = []
    print("\n🧠 Applying Multi-Stage Intelligence Filters...")
    
    for i in range(len(results['ids'][0])):
        meta = results['metadatas'][0][i]
        input_path = find_video_path(meta['filename'])
        if not input_path: continue

        # 1. Gap Check
        if any(sc['filename'] == meta['filename'] and abs(sc['timestamp'] - meta['timestamp']) < config['min_gap_seconds'] for sc in selected_clips):
            continue
            
        # 2. Smooth Motion
        is_smooth, variance = is_smooth_clip(input_path, meta['timestamp'], dynamic_clip_dur, config['max_motion_variance'])
        if not is_smooth:
            print(f"   - Rejected {meta['filename']} at {int(meta['timestamp'])}s (Jerky motion, variance: {variance:.1f})")
            continue

        # 3. Jump Cut Check
        if selected_clips and check_jump_cut(selected_clips[-1], meta, dynamic_clip_dur):
            print(f"   - Rejected {meta['filename']} at {int(meta['timestamp'])}s (Failed SSIM: Jump Cut detected)")
            continue
                
        selected_clips.append(meta)
        print(f"   + Selected: {meta['filename']} at {int(meta['timestamp'])}s")
        if len(selected_clips) == config['num_clips_to_generate']: break

    num_clips = len(selected_clips)
    if num_clips < 2:
        print("⚠️ Could not find enough diverse clips. Try a simpler prompt.")
        return

    temp_dir = config.get("temp_render_dir", "./temp_render")
    if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    
    print("\n✂️ Extracting clips, forcing 1080p/30fps, and applying LUT...")
    for i, meta in enumerate(selected_clips):
        start_time = max(0, meta['timestamp'] - (dynamic_fade_dur / 2))
        input_path = find_video_path(meta['filename'])
        output_clip = os.path.join(temp_dir, f"clip_{i}.mp4")
        
        video_filters = f"fps=30,scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2"
        if os.path.exists(config['lut_file']):
            video_filters += f",lut3d={config['lut_file']}"
        
        ffmpeg_cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-ss", str(start_time), "-i", input_path, "-t", str(dynamic_clip_dur),
            "-vf", video_filters, "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "22", output_clip
        ]
        subprocess.run(ffmpeg_cmd, check=True)

    print("\n🧵 Generating Filtergraph for Smooth Transitions...")
    final_output = f"{prompt.replace(' ', '_')}_cinematic.mp4"
    total_video_length = dynamic_clip_dur + (num_clips - 1) * (dynamic_clip_dur - dynamic_fade_dur)
    
    concat_cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    for i in range(num_clips): concat_cmd.extend(["-i", os.path.join(temp_dir, f"clip_{i}.mp4")])
    if chosen_music: concat_cmd.extend(["-i", chosen_music])

    filter_chains = []
    offset = dynamic_clip_dur - dynamic_fade_dur
    filter_chains.append(f"[0:v][1:v]xfade=transition=fade:duration={dynamic_fade_dur}:offset={offset}[v1]")
    for i in range(2, num_clips):
        offset = i * (dynamic_clip_dur - dynamic_fade_dur)
        filter_chains.append(f"[v{i-1}][{i}:v]xfade=transition=fade:duration={dynamic_fade_dur}:offset={offset}[v{i}]")
        
    video_map = f"[v{num_clips-1}]"
    filter_complex = "; ".join(filter_chains)

    audio_map = None
    if chosen_music:
        audio_idx = num_clips
        fade_out_start = total_video_length - dynamic_fade_dur
        filter_complex += f"; [{audio_idx}:a]afade=t=in:ss=0:d={dynamic_fade_dur},afade=t=out:st={fade_out_start}:d={dynamic_fade_dur}[aout]"
        audio_map = "[aout]"

    concat_cmd.extend(["-filter_complex", filter_complex, "-map", video_map])
    if audio_map: concat_cmd.extend(["-map", audio_map])
    concat_cmd.extend(["-t", str(total_video_length), "-c:v", "libx264", "-preset", "medium", "-crf", "20", final_output])
    
    subprocess.run(concat_cmd, check=True)
    shutil.rmtree(temp_dir)
    print(f"\n✅ MASTERPIECE COMPLETE! Saved as: {final_output}")

if __name__ == "__main__":
    user_prompt = input("🎥 What kind of cinematic montage do you want? (e.g., 'trees'): ")
    if user_prompt.strip():
        create_master_montage(user_prompt)