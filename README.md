This is a repo to make short clips from my drone videos.
Drone videos are cool but after a while it becomes difficult to keep track of and make actual content from.

This is entirely vibe-coded repo but it works good enough actually. I will keep making changes depending on my needs, but this is very specific to my GPU(6700xt) and os(kubuntu).

Here's an actual readme:

🛸 AI Drone Editor (Local Server)

An automated, privacy-first video editing pipeline powered by CLIP (Vision-Language Models) and Vector Databases. This tool allows you to ingest massive amounts of drone footage and generate beat-synced, color-graded montages simply by typing a text prompt.

Designed specifically for AMD GPUs (RX 6000 series) using ROCm on Linux.

🚀 Key Features

    Semantic Video Search: Search your raw footage using natural language (e.g., "gliding over forest canopy" or "sunset reflections on water").

    AMD GPU Acceleration: Fully optimized for Radeon 6700 XT/RDNA2 using ROCm, featuring half-precision (FP16) batch processing.

    Intelligent Editing:

        Temporal Diversity: Automatically avoids selecting clips from the same time window.

        Anti-Jump-Cut (SSIM): Uses Structural Similarity Index math to ensure transitions don't look glitchy or repetitive.

        Adaptive Search: Automatically relaxes filters if a strict search doesn't yield enough clips.

    Pro Production Pipeline:

        Beat-Sync: Analyzes music BPM to snap cuts and transitions to the rhythm.

        Automated LUTs: Applies DJI Mini 2 SE color correction (.cube files) during render.

        Cinematic Transitions: Automatic crossfades and audio ducking via FFmpeg.

🛠️ Tech Stack

    AI: OpenAI CLIP (via Hugging Face Transformers)

    Database: ChromaDB (Vector Store)

    Vision: OpenCV & Scikit-Image (SSIM)

    Audio: Librosa (BPM Analysis)

    Engine: FFmpeg (Video encoding & filtering)

    Hardware: AMD ROCm 6.1+

📦 Installation & Setup

1. Prerequisites (Kubuntu/Ubuntu)

```bash
sudo apt update && sudo apt install ffmpeg python3-venv python3-pip
```

2. Environment Setup

```bash
python3 -m venv venv
source venv/bin/activate
# Install ROCm-enabled PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.1
# Install requirements
pip install chromadb transformers opencv-python pillow tqdm scikit-image librosa
```


3. Configuration (config.json)

Create a config.json in the root directory:
```json

{
    "video_folders": ["/path/to/drone/videos"],
    "db_folder": "./chroma_db",
    "lut_file": "dji.cube",
    "music_folders": ["./music_library"],
    "sync_to_beats": true,
    "fps_to_extract": 1,
    "batch_size": 32,
    "clip_duration": 4.0,
    "fade_duration": 1.0,
    "min_gap_seconds": 10.0,
    "max_similarity_score": 0.65,
    "num_clips_to_generate": 5
}
```

🏗️ Workflow
Step 1: Ingest (Build the Knowledge Base)

Scan your video folders and create the AI embeddings. This script uses a cache—it will skip any video already in the database.

```bash
python ingest.py
```

Step 2: Search (Optional)

Test the AI's understanding of your footage without rendering a full video.

```bash
python search.py
```

Step 3: Generate (The Editor)

Type your prompt, and the system will pick a random song from your library, apply the LUT, and render a cinematic .mp4.
You can also use editor_2.py. It sync to the beats. 

```bash
python editor.py
```
🧪 Hardware Notes (AMD ROCm)

The code includes specific overrides for RDNA2 (GFX 10.3.0). It disables unstable MIOpen kernels and Flash Attention backends that traditionally cause segmentation faults on consumer AMD cards, ensuring a stable "nuclear" fallback for high-precision math.

📝 License

MIT - Build, fly, and edit freely.
