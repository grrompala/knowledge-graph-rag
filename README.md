# knowledge-graph-rag
Implementation of G-Retriever Graph RAG for optimized knowledgebase queries

Create an environemtn and Install dependencies
conda create -n kb-grag python=3.11
conda activate kb-grag
`pip -m -r requirements.txt`

## Acquiring the STaRK dataset

get_stark_data
- Difficult to install locally. 
- Spun up an ec2 instance to generate the data files
- see instuctions to get data from stark-qa package
stark_subsetter
- script to subset the stark database for test cases
load_stark_2_neo4j
-script to convert stark prime database to neo4j graph database
stark_embedding
-script to embed neo4j nodes  
subset qa dataset
-script to subset from the stark qa dataset for testing
starkmini_query_embedding
-script to embed stark queries

## G-Retriever training
generate training indice splits
-script to define training, validation, and test queries
train.py
-script to get subgraphs and train vector and graph (GNN+LLM) RAG models

# Evaluation


## Resources
https://stark.stanford.edu/dataset_prime.html
https://github.com/snap-stanford/STaRK
