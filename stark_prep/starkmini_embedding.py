from sentence_transformers import SentenceTransformer
import numpy as np
import neo4j
import os
import torch
from dotenv import load_dotenv
load_dotenv(override=True)
from neo4j import GraphDatabase
import json
import re
import torch
import pandas as pd


# ---------------------------
# 1️⃣ Connect to Neo4j
# ---------------------------
uri = os.environ["NEO4J_BOLT_URI"]
print(uri)
user = os.environ["NEO4J_USERNAME"]
print(user)
password = os.environ["NEO4J_PASSWORD"]
print(password)



neokb_driver = neo4j.GraphDatabase.driver(uri, auth=(user, password))
session = neokb_driver.session(database="starkmini")


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

def get_sample_nodes(driver, limit=None):
    if limit:
        query = f"""
        MATCH (n)
        RETURN n
        LIMIT {limit}
        """
    else:
        query = """
        MATCH (n)
        RETURN n
        """
    result = session.run(query)
    return [record.data() for record in result]

import json
import ast

def build_starkmini_doc(node):
    """
    Replicates STaRK get_doc_info(add_rel=False, compact=False)
    """
    # Extract node data
    node_data = node.get('n', node)
    
    # Extract properties if Neo4j node object
    if hasattr(node_data, '_properties'):
        props = node_data._properties
    elif hasattr(node_data, 'items'):
        props = dict(node_data.items())
    else:
        props = node_data
    
    name = props.get("name", "")
    node_type = props.get("type", "")
    source = props.get("source", "")
    details = props.get("details", {})
    
    # Parse details
    if isinstance(details, str):
        try:
            # First try JSON
            details = json.loads(details)
        except json.JSONDecodeError:
            try:
                # Fallback to Python literal eval (handles single quotes safely)
                details = ast.literal_eval(details)
            except Exception as e:
                print(f"Warning: Could not parse details for node {name}: {e}")
                details = {}
    
    doc = f"- name: {name}\n"
    doc += f"- type: {node_type}\n"
    doc += f"- source: {source}\n"
    
    gene_protein_text_explain = {
        'name': 'gene name',
        'type_of_gene': 'gene types',
        'alias': 'other gene names',
        'other_names': 'extended other gene names',
        'genomic_pos': 'genomic position',
        'generif': 'PubMed text',
        'interpro': 'protein family and classification information',
        'summary': 'protein summary text'
    }
    
    feature_text = "- details:\n"
    feature_cnt = 0
    
    if isinstance(details, dict) and details:
        for key, value in details.items():
            if value is None or str(value).strip() in ['', 'nan', 'None'] or key.startswith('_') or '_id' in key:
                continue
            
            # Special handling for gene/protein nodes
            if node_type == 'gene/protein' and key in gene_protein_text_explain:
                if key == 'interpro':
                    if isinstance(value, dict):
                        value = [value]
                    if isinstance(value, list):
                        value = [v.get('desc', '') if isinstance(v, dict) else v for v in value]
                
                if key == 'generif':
                    if isinstance(value, list):
                        value = '; '.join([v.get('text', '') if isinstance(v, dict) else str(v) for v in value])
                    value = ' '.join(str(value).split(' ')[:50000])
                
                if key == 'genomic_pos':
                    if isinstance(value, list) and len(value) > 0:
                        value = value[0]
                
                feature_text += f"  - {key} ({gene_protein_text_explain[key]}): {value}\n"
            else:
                feature_text += f"  - {key}: {value}\n"
            
            feature_cnt += 1
    
    if feature_cnt > 0:
        doc += feature_text
    
    return doc


# Initialize embedder and dictionary
embedder = HuggingFaceEmbedder(device="cpu")  # or "cuda"
emb_dict = {}

nodes = get_sample_nodes(session, limit=None)

for n in nodes:
    node_data = n.get('n', n)
    node_id = node_data.get("nodeId")
    if node_id is None:
        continue
    
    text = build_starkmini_doc(n)
    
    vector = embedder.embed_documents([text])[0]
    
    emb_dict[node_id] = torch.tensor(
        vector,
        dtype=torch.float32
    ).view(1, -1)

torch.save(emb_dict, "candidate_starkmini_emb_dict.pt")

    ==================================
Load Embeddings
==================================

print("\nLoading embeddings...")

#EMBEDDING_PATH =  emb_dict
#emb = torch.load(EMBEDDING_PATH)
emb = emb_dict

emb_records = [
    {"nodeId": k, "textEmbedding": v.squeeze().tolist()}
    for k, v in emb.items()
]

embedding_dim = len(emb_records[0]["textEmbedding"])

def chunks(xs, n):
    for i in range(0, len(xs), n):
        yield xs[i:i+n]


emb_query = """
UNWIND $recs AS rec
MATCH (n:_Entity_ {nodeId: rec.nodeId})
CALL db.create.setNodeVectorProperty(n, "textEmbedding", rec.textEmbedding)
"""
BATCH_SIZE_EMB = 500
for batch in chunks(emb_records, BATCH_SIZE_EMB):
    session.run(emb_query, {"recs": batch})

print("\nCreating vector index...")

session.run("""
    CREATE VECTOR INDEX text_embeddings IF NOT EXISTS
    FOR (n:_Entity_) ON (n.textEmbedding)
    OPTIONS {
      indexConfig: {
        `vector.dimensions`: $dim,
        `vector.similarity_function`: 'cosine'
      }
    }
""", {"dim": embedding_dim})

session.run('CALL db.awaitIndex("text_embeddings", 300)')