# MUST BE IMPORTED FIRST
from core.env_setup import configure_pytorch

# Now import the rest
import os
import cv2
import multiprocessing as mp
from tqdm import tqdm
from PIL import Image

from core.config import config
from core.model import VisionTextModel
from core.database import get_chroma_collection
import traceback
from core.logger import log  # <-- Import our new logger

def video_worker(worker_id, video_queue, frame_queue):
    cv2.setNumThreads(0) 
    log.info(f"CPU Worker {worker_id} spun up successfully.")
    
    while True:
        try:
            task = video_queue.get()
            if task is None:
                log.debug(f"Worker {worker_id} received shutdown signal.")
                frame_queue.put(None)
                break
                
            video_path, existing_ids, fps_to_extract = task
            filename = os.path.basename(video_path)
            log.debug(f"Worker {worker_id} starting decoding: {filename}")
            
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            if fps == 0:
                log.warning(f"Worker {worker_id}: Cannot read FPS for {filename}, skipping.")
                cap.release()
                frame_queue.put(("DONE", filename, None))
                continue

            frame_interval = max(1, int(fps / fps_to_extract) if fps_to_extract > 0 else int(fps))
            current_frame = 0
            
            while cap.isOpened():
                if current_frame % frame_interval == 0:
                    ret, frame = cap.read()
                    if not ret: break
                    
                    timestamp_sec = current_frame / fps
                    frame_id = f"{filename}_{timestamp_sec:.2f}"
                    
                    if frame_id not in existing_ids:
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        meta = {"filename": filename, "timestamp": timestamp_sec, "ingested_fps": fps_to_extract}
                        
                        # If RAM fills up, this queue.put block can freeze.
                        frame_queue.put(("FRAME", rgb_frame, meta))
                else:
                    if not cap.grab(): break
                current_frame += 1
                
            cap.release()
            log.debug(f"Worker {worker_id} finished decoding: {filename}")
            frame_queue.put(("DONE", filename, None))
            
        except Exception as e:
            # THIS IS THE CRITICAL ADDITION
            log.error(f"🚨 FATAL ERROR in Worker {worker_id} processing {filename if 'filename' in locals() else 'Unknown'}: {str(e)}")
            log.error(traceback.format_exc())
            # Ensure the queue doesn't deadlock by sending a completion signal even on failure
            frame_queue.put(("DONE", filename if 'filename' in locals() else "ERROR", None))
            
def process_batch(frames, metadatas, model, collection):
    if not frames: return
    embeddings = model.get_image_embeddings(frames)
    ids = [f"{m['filename']}_{m['timestamp']:.2f}" for m in metadatas]
    collection.add(embeddings=embeddings, metadatas=metadatas, ids=ids)

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    device = configure_pytorch()
    
    # Initialize Core Resources
    model = VisionTextModel(device)
    collection = get_chroma_collection()

    fps_to_extract = config.get('fps_to_extract', 1)
    batch_size = config.get('batch_size', 32)
    num_workers = config.get('num_workers', 4) 
    
    tasks_to_run = []
    print("\n📂 Scanning folders and validating video properties...")
    for folder in config['video_folders']:
        if not os.path.exists(folder): continue
        for vf in os.listdir(folder):
            if vf.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
                video_path = os.path.join(folder, vf)
                filename = os.path.basename(video_path)
                
                cap = cv2.VideoCapture(video_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                cap.release()
                if fps == 0: continue
                
                rounded_fps = int(round(fps))
                if fps_to_extract > rounded_fps or rounded_fps % fps_to_extract != 0:
                    valid_factors = [str(i) for i in range(1, rounded_fps + 1) if rounded_fps % i == 0]
                    print(f"⚠️ Skipping {filename}: Requested {fps_to_extract} FPS is not a valid factor of ~{rounded_fps} FPS. (Valid: {', '.join(valid_factors)})")
                    continue

                existing_records = collection.get(where={"filename": filename}, include=["metadatas"])
                existing_ids = set(existing_records['ids'])
                
                if existing_ids:
                    ingested_fps = max([m.get("ingested_fps", 0) for m in existing_records['metadatas']])
                    if fps_to_extract <= ingested_fps:
                        print(f"⏭️  Already Synced: {filename} ({ingested_fps} fps)")
                        continue
                    else:
                        print(f"🔄 Queued for Upgrade: {filename} ({ingested_fps} fps -> {fps_to_extract} fps)")
                else:
                    print(f"📥 Queued for Ingestion: {filename}")
                
                tasks_to_run.append((video_path, existing_ids, fps_to_extract))

    if not tasks_to_run:
        print("\n✅ Knowledge Base is fully up to date. Nothing to do!")
        exit()
    
    # Fetch max_size from config to prevent RAM overflow
    queue_limit = config.get('queue_max_size', 100)
    
    video_queue = mp.Queue()
    frame_queue = mp.Queue(maxsize=queue_limit)
    for task in tasks_to_run: video_queue.put(task)
    for _ in range(num_workers): video_queue.put(None)
        
    workers = [mp.Process(target=video_worker, args=(i, video_queue, frame_queue)) for i in range(num_workers)]
    for w in workers: w.start()

    print(f"\n🚀 Launching {num_workers} CPU Cores to feed the GPU...")
    frame_buffer, meta_buffer = [], []
    active_workers = num_workers
    
    with tqdm(total=len(tasks_to_run), desc="Videos Processed") as pbar:
        while active_workers > 0:
            item = frame_queue.get()
            if item is None:
                active_workers -= 1
                continue
                
            msg_type, data1, data2 = item
            if msg_type == "FRAME":
                frame_buffer.append(Image.fromarray(data1))
                meta_buffer.append(data2)
                
                if len(frame_buffer) >= batch_size:
                    process_batch(frame_buffer, meta_buffer, model, collection)
                    frame_buffer.clear()
                    meta_buffer.clear()
                    
            elif msg_type == "DONE":
                pbar.update(1)

    if frame_buffer:
        process_batch(frame_buffer, meta_buffer, model, collection)

    for p in workers: p.join()
    print("\n✅ High-Speed Knowledge Base Sync Complete!")