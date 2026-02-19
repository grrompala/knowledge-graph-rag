import pandas as pd
import torch
from tqdm import tqdm

# ---- Your embedder class ----
from sentence_transformers import SentenceTransformer

class HuggingFaceEmbedder:
    def __init__(self, model_name="cambridgeltl/SapBERT-from-PubMedBERT-fulltext", device="cpu"):
        self.model = SentenceTransformer(model_name, device=device)
    def embed_query(self, text: str) -> list[float]:
        # returns the embedding as list of floats to match OpenAIEmbeddings
        vector = self.model.encode([text], convert_to_numpy=True, normalize_embeddings=True)[0]
        return vector.tolist()
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return vectors.tolist()
# ------- Load queries from TSV ----

qa_df = pd.read_csv("stark_qa_final_enriched_starkmini.tsv", sep="\t")
qa_df["query"] = qa_df["query"].astype(str)
texts = qa_df['query'].tolist()
indices = qa_df['id'].tolist()  # Assuming 'id' is the unique identifier for each query

# ---- Initialize embedder ----
embedder = HuggingFaceEmbedder(device="cpu")  # or "cpu"

emb_dict = {}
batch_size = 100

print(f"Generating embeddings for {len(texts)} queries...")

for i in tqdm(range(0, len(texts), batch_size)):
    batch_texts = texts[i:i+batch_size]
    batch_indices = indices[i:i+batch_size]
    # Get list[list[float]]
    batch_vectors = embedder.embed_documents(batch_texts)
    # Convert to torch tensor
    batch_tensor = torch.tensor(batch_vectors, dtype=torch.float32)
    for idx, emb in zip(batch_indices, batch_tensor):
        emb_dict[idx] = emb.view(1, -1)

# ---- Save in STaRK-compatible format ----
torch.save(emb_dict, "query_emb_dict.pt")

print(f"Saved {len(emb_dict)} embeddings!")