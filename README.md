# network-embedding-evaluation

Tentative Folder Structure:
- embeddings/<embedding method>/ - holds all relevant exports for a given embedding method (HEER, DistMult, ComplEx, ConvE, etc.)

- datasets/FB15K237/ 

- models/ - contains class definitions for each embedding method, these classes will just load existing embeddings from embeddings/ and node/edge IDs from datasets/ and execute prediction functions for testing

- evaluate.py - top-level evaluation script that performs all tests on all models and reports metrics
