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

## Evaluation

<!-- TODO: Add descriptions for evaluation scripts once filenames are provided -->

## Resources

- [STaRK Dataset (Prime)](https://stark.stanford.edu/dataset_prime.html)
- [STaRK GitHub Repository](https://github.com/snap-stanford/STaRK)


