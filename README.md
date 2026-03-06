# Knowledge Graph RAG

Implementation of G-Retriever Graph RAG for optimized knowledgebase queries.

## Getting Started

Create an environment and install dependencies:

```bash
conda create -n kb-grag python=3.11
conda activate kb-grag
pip install -r requirements.txt
```

## Acquiring the STaRK Dataset

- **`get_stark_data.py`** — Difficult to install locally; spun up an EC2 instance to generate the data files. See instructions in the [stark-qa package](https://github.com/snap-stanford/STaRK) for obtaining data.
- **`stark_subsetter.py`** — Script to subset the STaRK database for test cases.
- **`load_stark_2_neo4j.py`** — Script to convert the STaRK Prime database into a Neo4j graph database.
- **`stark_embedding.py`** — Script to generate embeddings for Neo4j nodes.
- **`subset_qa_dataset.py`** — Script to subset the STaRK QA dataset for testing.
- **`starkmini_query_embedding.py`** — Script to embed STaRK queries.

## G-Retriever Training

- **`generate_training_indice_splits.py`** — Script to define training, validation, and test query splits.
- **`train.py`** — Script to extract subgraphs and train vector and graph (GNN + LLM) RAG models.
    - **`STaRKQADatasetGDS.py` — 
        - Vectorize query with open-source hugging face embedder
        - Run cosine similarity search on indexed node embeddings to get 25*k-nodes
        - Retrieve each 1-hop node and the relationship type
        - Graph pruning (PCST)
        - Build PyG PyTroch Geometric Graph Object
    
    
## Evaluation

<!-- TODO: Add descriptions for evaluation scripts once filenames are provided -->

## Resources

- [STaRK Dataset (Prime)](https://stark.stanford.edu/dataset_prime.html)
- [STaRK GitHub Repository](https://github.com/snap-stanford/STaRK)

## AWS Support

Training an LLM (even `tiny_llama`) requires GPU resources beyond what a T4 can provide. Below are instructions for setting up a suitable AWS instance.

### AWS Instance Setup

1. **Instance type:** `g5.xlarge`
2. **Network:** Use the default VPC
3. **AMI:** Deep Learning Base AMI with Single CUDA (Amazon Linux 2023)
4. **Storage:** 100 GB EBS

> **Important:** Ensure your vCPU quota allows GPU instances (check *Running On-Demand G and VT instances* in Service Quotas). Spot instances are significantly cheaper if you use S3 storage for checkpointed training results.

5. **Security:** Add an SSH inbound security rule and restrict it to your IP address.

### Environment Configuration

Ensure Conda is installed in a location with sufficient disk space, then configure the environment and package directories:

```bash
conda config --add envs_dirs /home/ec2-user/anaconda3/envs
conda config --add pkgs_dirs /home/ec2-user/anaconda3/pkgs
```

### Cost Estimate

Training on ~200 samples with an unfrozen model is very low cost (under $1).
