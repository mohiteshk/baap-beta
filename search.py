# MUST BE IMPORTED FIRST
from core.env_setup import configure_pytorch

# Now import the rest
from core.config import config
from core.model import VisionTextModel
from core.database import get_chroma_collection

device = configure_pytorch()
model = VisionTextModel(device)
collection = get_chroma_collection()

def search_video(prompt, top_k):
    embedding = model.get_text_embedding(prompt)
    
    results = collection.query(
        query_embeddings=[embedding],
        n_results=top_k
    )
    
    print(f"\n🎥 --- Top {top_k} matches for: '{prompt}' ---")
    
    ids = results['ids'][0]
    metadatas = results['metadatas'][0]
    distances = results['distances'][0]
    
    for i in range(len(ids)):
        meta = metadatas[i]
        mins = int(meta['timestamp'] // 60)
        secs = int(meta['timestamp'] % 60)
        print(f"{i+1}. {meta['filename']} @ {mins}:{secs:02d} (Match Score: {distances[i]:.4f})")

if __name__ == "__main__":
    top_k = config.get("search_top_k", 5)
    print("\n✅ System Ready! Type what you want to find in your drone videos.")
    while True:
        try:
            user_prompt = input("\n🔍 Search Prompt (or type 'q' to quit): ")
            if user_prompt.lower() in ['q', 'quit', 'exit']: break
            if user_prompt.strip() == "": continue
            
            search_video(user_prompt, top_k)
        except KeyboardInterrupt:
            break