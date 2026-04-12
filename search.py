import os
# --- AMD ROCm 6700 XT STABILITY FIXES ---
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
os.environ["MIOPEN_DISABLE_CACHE"] = "1"
os.environ["PYTORCH_HIP_ALLOC_CONF"] = "garbage_collection_threshold:0.8,max_split_size_mb:512"

import torch

# Disable unstable optimized kernels for RDNA2
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
torch.backends.cudnn.enabled = False
# ----------------------------------------

import chromadb
from transformers import CLIPProcessor, CLIPModel

# --- CONFIGURATION ---
DB_FOLDER = "./chroma_db"

print("Loading CLIP Model (Text Engine) onto GPU...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "openai/clip-vit-base-patch32"

processor = CLIPProcessor.from_pretrained(model_id)
# Load in Half-Precision (Safe Mode)
model = CLIPModel.from_pretrained(model_id, torch_dtype=torch.float16).to(device)

print("Connecting to Knowledge Base...")
chroma_client = chromadb.PersistentClient(path=DB_FOLDER)
collection = chroma_client.get_collection(name="drone_footage")

def search_video(prompt, top_k=5):
    # 1. Process the text prompt
    inputs = processor(text=[prompt], return_tensors="pt", padding=True).to(device)
    
    #with torch.no_grad():
    #    # 2. Get the text features (meaning) from the AI
    #    text_features = model.get_text_features(
    #        input_ids=inputs.input_ids, 
    #        attention_mask=inputs.attention_mask
    #    )
    #    
    #    # 3. Safety catch for HuggingFace wrappers
    #    if not isinstance(text_features, torch.Tensor):
    #        if hasattr(text_features, 'text_embeds'):
    #            text_features = text_features.text_embeds
    #        else:
    #            text_features = text_features[0]
    #            
    #    # 4. Normalize for math comparison
    #    text_features = text_features.to(torch.float32)
    #    text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
    #    embedding = text_features.cpu().numpy().tolist()[0]
    
    #with torch.no_grad():
    #    # 2. Get the text features (meaning) from the AI
    #    text_features = model.get_text_features(
    #        input_ids=inputs.input_ids, 
    #        attention_mask=inputs.attention_mask
    #    )
    #    
    #    # 3. Safety catch for HuggingFace wrappers
    #    if not isinstance(text_features, torch.Tensor):
    #        if hasattr(text_features, 'text_embeds'):
    #            text_features = text_features.text_embeds
    #        else:
    #            text_features = text_features[0]
    #            
    #    # 4. Normalize for math comparison and FLATTEN to a 1D list
    #    text_features = text_features.to(torch.float32)
    #    text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
    #    # THE FIX: Use .flatten() to destroy any extra brackets before making it a list
    #    embedding = text_features.cpu().numpy().flatten().tolist()

    with torch.no_grad():
        # 1. Ask the internal text model directly for its outputs
        text_outputs = model.text_model(
            input_ids=inputs.input_ids, 
            attention_mask=inputs.attention_mask
        )
        
        # 2. Grab the 'pooler_output' (the single 512-dim summary of the whole prompt, ignoring individual words)
        pooled_output = text_outputs.pooler_output
        
        # 3. Project it into the multimodal space so it matches your images
        text_features = model.text_projection(pooled_output)
                
        # 4. Normalize and convert to a pure 1D list of exactly 512 floats
        text_features = text_features.to(torch.float32)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        embedding = text_features.cpu().numpy().flatten().tolist()

    # 5. Search the database!
    results = collection.query(
        query_embeddings=[embedding],
        n_results=top_k
    )
    
    print(f"\n🎥 --- Top {top_k} matches for: '{prompt}' ---")
    
    # ChromaDB returns lists inside lists, so we unpack [0]
    ids = results['ids'][0]
    metadatas = results['metadatas'][0]
    distances = results['distances'][0]
    
    for i in range(len(ids)):
        meta = metadatas[i]
        filename = meta['filename']
        timestamp = meta['timestamp']
        # Distance: closer to 0 is a better match
        distance = distances[i] 
        
        # Format the time beautifully (e.g., 1:04 instead of 64.0)
        mins = int(timestamp // 60)
        secs = int(timestamp % 60)
        
        print(f"{i+1}. {filename} @ {mins}:{secs:02d} (Match Score: {distance:.4f})")

# --- INTERACTIVE LOOP ---
if __name__ == "__main__":
    print("\n✅ System Ready! Type what you want to find in your drone videos.")
    while True:
        try:
            user_prompt = input("\n🔍 Search Prompt (or type 'q' to quit): ")
            if user_prompt.lower() in ['q', 'quit', 'exit']:
                break
            if user_prompt.strip() == "":
                continue
                
            search_video(user_prompt)
        except KeyboardInterrupt:
            break