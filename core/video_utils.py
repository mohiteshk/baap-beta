import os
import cv2
import random
import numpy as np
from skimage.metrics import structural_similarity as ssim
from core.config import config

def find_video_path(filename):
    for folder in config['video_folders']:
        path = os.path.join(folder, filename)
        if os.path.exists(path):
            return path
    return None

def get_random_music():
    valid_exts = ('.mp3', '.wav', '.m4a')
    all_music = []
    for folder in config['music_folders']:
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

def is_smooth_clip(video_path, start_time, duration, max_variance=15.0):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, int(start_time * 1000))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames_to_check = int(duration * fps)
    step = max(1, frames_to_check // 10)
    
    prev_frame = None
    motion_deltas = []
    
    for i in range(frames_to_check):
        ret, frame = cap.read()
        if not ret: break
        if i % step == 0:
            gray = cv2.cvtColor(cv2.resize(frame, (320, 180)), cv2.COLOR_BGR2GRAY)
            if prev_frame is not None:
                diff = np.mean(cv2.absdiff(gray, prev_frame))
                motion_deltas.append(diff)
            prev_frame = gray
            
    cap.release()
    if len(motion_deltas) < 2:
        return True, 0.0
        
    motion_variance = np.var(motion_deltas)
    return motion_variance < max_variance, motion_variance