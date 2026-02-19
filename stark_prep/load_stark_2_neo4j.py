"""
Load exported Prime graph into a specific Neo4j database.

Requirements:
- Neo4j 5.11+ (for vector indexes)
- Database must already exist
- torch export file (prime_graph_export.pt)
"""

# =========================
# Imports
# =========================

import os
import re
import torch
import pandas as pd
from neo4j import GraphDatabase
from dotenv import load_dotenv


# =========================
# Configuration
# =========================

TARGET_DB = "starkmini"

GRAPH_EXPORT_PATH = "prime_graph_subset.pt"
#EMBEDDING_PATH = "emb/prime/text-embedding-ada-002/doc/candidate_emb_dict.pt"

BATCH_SIZE_NODES = 50
BATCH_SIZE_RELS = 500
#BATCH_SIZE_EMB = 1000


# =========================
# Load Environment
# =========================

load_dotenv(".env", override=True)

NEO4J_BOLT_URI = os.getenv("NEO4J_BOLT_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

assert NEO4J_BOLT_URI, "Missing NEO4J_BOLT_URI"
assert NEO4J_USERNAME, "Missing NEO4J_USERNAME"
assert NEO4J_PASSWORD, "Missing NEO4J_PASSWORD"


# =========================
# Helper Functions
# =========================

def format_node_label(s: str) -> str:
    ss = s.replace("/", "_or_").lower().split("_")
    return "".join(t.title() for t in ss)


def format_rel_type(s: str) -> str:
    return re.sub("[^0-9A-Z]+", "_", s.upper())


def chunks(xs, n):
    for i in range(0, len(xs), n):
        yield xs[i:i+n]


def make_constraint_query(label, prop):
    name = f"unique_{label.lower()}_{prop.lower()}"
    return (
        f"CREATE CONSTRAINT {name} IF NOT EXISTS "
        f"FOR (n:{label}) REQUIRE n.{prop} IS UNIQUE"
    )


def make_node_merge_query(label, cols):
    prop_names = [c for c in cols if c != "nodeId"]

    set_clause = ""
    if prop_names:
        set_clause = "\nSET " + ", ".join(
            [f"n.{p} = rec.{p}" for p in prop_names]
        )

    return f"""
    UNWIND $recs AS rec
    MERGE (n:{label} {{nodeId: rec.nodeId}})
    {set_clause}
    """


def make_rel_merge_query(rel_type, cols):
    prop_names = [c for c in cols if c not in ["src", "tgt"]]

    set_clause = ""
    if prop_names:
        set_clause = "\nSET " + ", ".join(
            [f"r.{p} = rec.{p}" for p in prop_names]
        )

    return f"""
    UNWIND $recs AS rec
    MATCH (s:_Entity_ {{nodeId: rec.src}})
    MATCH (t:_Entity_ {{nodeId: rec.tgt}})
    MERGE (s)-[r:{rel_type}]->(t)
    {set_clause}
    """


# =========================
# Load Exported Graph
# =========================

print("Loading graph export...")

data = torch.load(GRAPH_EXPORT_PATH)

nodes = data["nodes"]
edge_index = data["edge_index"]
edge_types = data["edge_types"]
node_type_dict = data["node_type_dict"]
edge_type_dict = data["edge_type_dict"]

print("Building node DataFrame...")

node_list = []
for i, node in enumerate(nodes):
    node_copy = node.copy()
    node_copy["nodeId"] = i
    node_list.append(node_copy)

node_df = pd.DataFrame(node_list)

if "details" in node_df.columns:
    node_df["details"] = node_df["details"].fillna("").astype(str)

print(f"Total nodes: {len(node_df)}")


# =========================
# Neo4j Load
# =========================

with GraphDatabase.driver(
    NEO4J_BOLT_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
) as driver:

    with driver.session(database=TARGET_DB) as session:

        print(f"\nUsing database: {TARGET_DB}")

        # ==================================
        # Load Nodes
        # ==================================

        for _, node_type in node_type_dict.items():

            df = node_df[node_df["type"] == node_type].drop(columns=["type"])
            label = format_node_label(node_type)

            print(f"\nLoading {label} nodes ({len(df)})")

            session.run(make_constraint_query(label, "nodeId"))

            query = make_node_merge_query(label, df.columns)

            for batch in chunks(df.to_dict("records"), BATCH_SIZE_NODES):
                session.run(query, {"recs": batch})

        # Add universal entity label
        print("\nAdding _Entity_ label to all nodes")
        session.run("MATCH (n) SET n:_Entity_")

        session.run("""
            CREATE CONSTRAINT unique__entity__nodeid IF NOT EXISTS
            FOR (n:_Entity_) REQUIRE n.nodeId IS UNIQUE
        """)

        # ==================================
        # Load Relationships
        # ==================================

        print("\nBuilding relationship DataFrame...")

        rel_df = pd.DataFrame(
            torch.cat(
                [edge_index, edge_types.reshape(1, edge_types.size(0))],
                dim=0
            ).t(),
            columns=["src", "tgt", "typeId"],
        )

        for ind, edge_type in edge_type_dict.items():

            df = rel_df[rel_df["typeId"] == ind].drop(columns=["typeId"])
            rel_label = format_rel_type(edge_type)

            print(f"\nLoading {rel_label} relationships ({len(df)})")

            query = make_rel_merge_query(rel_label, df.columns)

            for batch in chunks(df.to_dict("records"), BATCH_SIZE_RELS):
                session.run(query, {"recs": batch})

        # ==================================
        # Load Embeddings
        # ==================================

        # print("\nLoading embeddings...")

        # emb = torch.load(EMBEDDING_PATH)

        # emb_records = [
        #     {"nodeId": k, "textEmbedding": v.squeeze().tolist()}
        #     for k, v in emb.items()
        # ]

        # embedding_dim = len(emb_records[0]["textEmbedding"])

        # emb_query = """
        # UNWIND $recs AS rec
        # MATCH (n:_Entity_ {nodeId: rec.nodeId})
        # CALL db.create.setNodeVectorProperty(n, "textEmbedding", rec.textEmbedding)
        # """

        # for batch in chunks(emb_records, BATCH_SIZE_EMB):
        #     session.run(emb_query, {"recs": batch})

        # print("\nCreating vector index...")

        # session.run("""
        #     CREATE VECTOR INDEX text_embeddings IF NOT EXISTS
        #     FOR (n:_Entity_) ON (n.textEmbedding)
        #     OPTIONS {
        #       indexConfig: {
        #         `vector.dimensions`: $dim,
        #         `vector.similarity_function`: 'cosine'
        #       }
        #     }
        # """, {"dim": embedding_dim})

        # session.run('CALL db.awaitIndex("text_embeddings", 300)')


## Get all nodeIds
with GraphDatabase.driver(
    NEO4J_BOLT_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
) as driver:
    with driver.session(database=TARGET_DB) as session:
        print("\nCollecting distinct nodeIds to file...")
        result = session.run("MATCH (n) RETURN DISTINCT n.name AS names")
        node_ids = [record["name"] for record in result]
        node_output_file = "names.txt"
        with open(node_output_file, "w") as f:
            for name in names:
                f.write(f"{name}\n")
        
        print(f"Saved {len(names)} names to {node_output_file}")
        print("nodeIds collected✅")
        
print("\nDONE 🚀")
