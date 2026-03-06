# stopped at 38400 nodes

from sentence_transformers import SentenceTransformer
import neo4j
import os
from dotenv import load_dotenv
import json
import ast

# ===========================
# 1️⃣ Load Environment
# ===========================
load_dotenv(override=True)

URI = os.environ["NEO4J_BOLT_URI"]
USER = os.environ["NEO4J_USERNAME"]
PASSWORD = os.environ["NEO4J_PASSWORD"]
DATABASE = os.environ.get("NEO4J_DATABASE", "neo4j")

print(f"Connecting to {URI}")
print(f"Using database: {DATABASE}")

driver = neo4j.GraphDatabase.driver(URI, auth=(USER, PASSWORD))


# ===========================
# 2️⃣ Embedder
# ===========================
class HuggingFaceEmbedder:
    def __init__(self,
                 model_name="cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
                 device="cpu"):
        self.model = SentenceTransformer(model_name, device=device)
    def embed_documents(self, texts):
        return self.model.encode(
            texts,
            batch_size=64,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True
        )


embedder = HuggingFaceEmbedder(device="cpu")  # change to "cuda" if available


# ===========================
# 3️⃣ Document Builder
# ===========================
def build_stark_doc(node):
    name = node.get("name", "")
    node_type = node.get("type", "")
    source = node.get("source", "")
    details = node.get("details", {})
    # Parse details safely
    if isinstance(details, str):
        try:
            details = json.loads(details)
        except json.JSONDecodeError:
            try:
                details = ast.literal_eval(details)
            except Exception:
                details = {}
    doc = f"- name: {name}\n"
    doc += f"- type: {node_type}\n"
    doc += f"- source: {source}\n"
    if isinstance(details, dict):
        doc += "- details:\n"
        for k, v in details.items():
            if v is None:
                continue
            val = str(v).strip()
            if val in ["", "nan", "None"] or k.startswith("_") or "_id" in k:
                continue
            # truncate large fields
            val = val[:4000]
            doc += f"  - {k}: {val}\n"
    # HARD truncate for model token limits
    return doc[:4000]


# ===========================
# 4️⃣ Stream Nodes From Neo4j
# ===========================
def stream_nodes(batch_size=500):
    query = """
    MATCH (n:_Entity_)
    WHERE n.textEmbedding IS NULL
    RETURN n.nodeId AS nodeId,
           n.name AS name,
           n.type AS type,
           n.source AS source,
           n.details AS details
    """
    with driver.session(database=DATABASE) as session:
        result = session.run(query)
        batch = []
        for record in result:
            batch.append(record.data())
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch


# ===========================
# 5️⃣ Embed + Write Back
# ===========================
write_query = """
UNWIND $recs AS rec
MATCH (n:_Entity_ {nodeId: rec.nodeId})
CALL db.create.setNodeVectorProperty(n, "textEmbedding", rec.textEmbedding)
"""

total_processed = 0

for node_batch in stream_nodes(batch_size=512):
    node_ids = []
    texts = []
    for node in node_batch:
        if node["nodeId"] is None:
            continue
        node_ids.append(node["nodeId"])
        texts.append(build_stark_doc(node))
    if not texts:
        continue
    vectors = embedder.embed_documents(texts)
    emb_records = [
        {"nodeId": nid, "textEmbedding": vec.tolist()}
        for nid, vec in zip(node_ids, vectors)
    ]
    with driver.session(database=DATABASE) as session:
        session.run(write_query, {"recs": emb_records})
    total_processed += len(emb_records)
    print(f"Processed {total_processed} nodes")
print("Embedding complete.")


# ===========================
# 6️⃣ Create Vector Index
# ===========================
print("Creating vector index...")

with driver.session(database=DATABASE) as session:

    # Get dimension from first embedding
    dim = len(vectors[0])

    session.run("""
        CREATE VECTOR INDEX text_embeddings IF NOT EXISTS
        FOR (n:_Entity_) ON (n.textEmbedding)
        OPTIONS {
          indexConfig: {
            `vector.dimensions`: $dim,
            `vector.similarity_function`: 'cosine'
          }
        }
    """, {"dim": dim})

    session.run('CALL db.awaitIndex("text_embeddings", 300)')

print("Vector index ready.")
driver.close()