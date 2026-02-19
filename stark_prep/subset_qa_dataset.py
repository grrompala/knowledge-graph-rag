import pandas as pd
import ast  # for safely parsing lists if stored as strings

# Load the QA dataset parquet file
qa_df = pd.read_parquet("qa_dataset.parquet")

# Load or define the nodeIds in your starkmini subset
# For example, from your subset export:
subset_node_ids = set(range(1000))  # replace with actual IDs from your subset

# Ensure node_names are Python lists
# (only needed if stored as strings in parquet)
if isinstance(qa_df.node_names.iloc[0], str):
    qa_df['node_names'] = qa_df['node_names'].apply(ast.literal_eval)

# Filter rows: keep only rows where at least one answer_id is in subset_node_ids
def has_subset_answer(ans_ids, valid_ids):
    return any(aid in valid_ids for aid in ans_ids)

qa_filtered = qa_df[qa_df['node_names'].apply(lambda x: has_subset_answer(x, subset_node_ids))]

print(f"Original QA rows: {len(qa_df)}, Filtered QA rows: {len(qa_filtered)}")

# Optional: reset index
qa_filtered = qa_filtered.reset_index(drop=True)

# Save filtered dataset for downstream training
qa_filtered.to_parquet("qa_subset.parquet", index=False)

qa_df = pd.read_csv("stark_qa_enriched_starkmini.tsv",sep="\t")
columns_to_keep = ['id', 'query', 'mini_answer_ids']
qa_subset = qa_df[columns_to_keep] 
qa_subset.to_csv("stark_qa_final_enriched_starkmini.tsv", index=False,sep="\t")
