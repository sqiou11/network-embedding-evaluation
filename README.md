# network-embedding-evaluation

Tentative Folder Structure:
- embeddings/pytorch/ - holds all PyTorch exports (HEER, DistMult, ComplEx, ConvE)
- embeddings/tensorflow/ - holds all TensorFlow exports (ProjE)

- datasets/FB15K237/

- models/ - contains class definitions for each embedding method, these classes will just load existing embeddings and node/edge IDs and execute prediction functions for testing

- evaluate.py - top-level evaluation script that performs all tests on all models and reports metrics
