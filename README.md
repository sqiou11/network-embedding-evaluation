# network-embedding-evaluation

## Folder Structure:
- embeddings/\<embedding_method\>/ - holds all relevant exports for a given embedding method (HEER, DistMult, ComplEx, ConvE, etc.)

- datasets/FB15K237/ holds all testing and training datasets as well as other FB15K237-specific metadata needed by the testing framework

- models/ - contains class definitions for each embedding method, these classes will just load existing embeddings from embeddings/ and node/edge IDs from datasets/ and execute prediction functions for testing

- results/ - contains *_metrics.txt and *_scores.txt files

## Testing
To run the edge reconstruction experiment (modeled from HEER paper), just supply a model name as the first argument to the top-level script (i.e. `./eval.sh HEER` or `./eval.sh DistMult`). The results will be output to the results folder.
