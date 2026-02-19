import torch
import random
from collections import deque

# --- Parameters ---
INPUT_GRAPH_FILE = "prime_graph_export.pt"       # full graph
OUTPUT_GRAPH_FILE = "prime_graph_subset.pt"      # subset output
NUM_NODES_SUBSET = 1000                          # number of nodes in subset
SEED = 42                                        # reproducibility

random.seed(SEED)
torch.manual_seed(SEED)

# --- Load full graph ---
graph = torch.load(INPUT_GRAPH_FILE)
nodes = graph["nodes"]
edge_index = graph["edge_index"]       # 2 x E tensor
edge_types = graph["edge_types"]       # E tensor
node_type_dict = graph["node_type_dict"]
edge_type_dict = graph["edge_type_dict"]

num_total_nodes = len(nodes)
print(f"Full graph: {num_total_nodes} nodes, {edge_index.size(1)} edges")

# --- Subset nodes using BFS/random walk ---
start_node = random.randint(0, num_total_nodes - 1)
subset_node_ids = set([start_node])
queue = deque([start_node])

while len(subset_node_ids) < NUM_NODES_SUBSET and queue:
    current = queue.popleft()
    neighbors = (edge_index[1][edge_index[0] == current].tolist() +
                 edge_index[0][edge_index[1] == current].tolist())
    random.shuffle(neighbors)
    for n in neighbors:
        if n not in subset_node_ids:
            subset_node_ids.add(n)
            queue.append(n)
        if len(subset_node_ids) >= NUM_NODES_SUBSET:
            break

subset_node_ids = sorted(subset_node_ids)
print(f"Subset selected: {len(subset_node_ids)} nodes")

# --- Create mapping old_id -> subset index (for edge reindexing) ---
old2subset_idx = {old_id: new_idx for new_idx, old_id in enumerate(subset_node_ids)}

# --- Reindex nodes (keep original nodeId) ---
subset_nodes_reindexed = [nodes[old_id].copy() for old_id in subset_node_ids]

# --- Reindex edges: only edges between subset nodes ---
subset_edge_index = []
subset_edge_types = []

for i in range(edge_index.size(1)):
    s, t = edge_index[:, i].tolist()
    if s in old2subset_idx and t in old2subset_idx:
        subset_edge_index.append([old2subset_idx[s], old2subset_idx[t]])
        subset_edge_types.append(edge_types[i].item())

subset_edge_index = torch.tensor(subset_edge_index).t()  # 2 x E_subset
subset_edge_types = torch.tensor(subset_edge_types)

print(f"Subset graph: {len(subset_nodes_reindexed)} nodes, {subset_edge_index.size(1)} edges")

# --- Save subset graph ---
subset_graph = {
    "nodes": subset_nodes_reindexed,
    "edge_index": subset_edge_index,
    "edge_types": subset_edge_types,
    "node_type_dict": node_type_dict,
    "edge_type_dict": edge_type_dict
}

torch.save(subset_graph, OUTPUT_GRAPH_FILE)
print(f"Subset graph saved to {OUTPUT_GRAPH_FILE} (clean, no neoNodeId)")
