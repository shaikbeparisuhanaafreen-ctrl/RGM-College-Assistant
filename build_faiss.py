import os
import json
import faiss
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from pathlib import Path


# CONFIG

DATA_DIR = "rag_data_optimized_new"
INDEX_FILE = "faiss_index.bin"
METADATA_FILE = "faiss_metadata.json"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K_DEFAULT = 5



# LOAD EMBEDDING MODEL

print("🔄 Loading embedding model...")
model = SentenceTransformer(EMBED_MODEL)
print("✅ Model loaded")



# LOAD ALL CHUNKS

def load_chunks(data_dir):
    all_chunks = []

    for file in Path(data_dir).glob("*.json"):
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)
            all_chunks.extend(data)

    print(f"📚 Total chunks loaded: {len(all_chunks)}")
    return all_chunks



# BUILD FAISS INDEX

def build_index(chunks):
    texts = []
    metadata = []

    print("🧠 Preparing texts for embedding...")

    for chunk in chunks:
        text = chunk.get("text_for_embedding")

        if not text:
            continue

        texts.append(text)

        # ⭐ store FULL chunk (includes images)
        metadata.append(chunk)

    print(f"🔢 Generating embeddings for {len(texts)} chunks...")

    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    dim = embeddings.shape[1]

    print("⚡ Building FAISS index...")
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    print(f"✅ FAISS index built with {index.ntotal} vectors")

    return index, metadata



# SAVE INDEX

def save_index(index, metadata):
    faiss.write_index(index, INDEX_FILE)

    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("💾 Index and metadata saved")



# MAIN

if __name__ == "__main__":
    chunks = load_chunks(DATA_DIR)
    index, metadata = build_index(chunks)
    save_index(index, metadata)

    print("\n🎉 RAG EMBEDDINGS READY")
    print(f"📦 Index file: {INDEX_FILE}")
    print(f"📦 Metadata file: {METADATA_FILE}")