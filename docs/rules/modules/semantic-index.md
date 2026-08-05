# Semantic Index

## Scope

This document covers semantic retrieval infrastructure: semantic-index
generation contracts, hosted embeddings, reranking, committed semantic-index
artifacts, and database-backed context search entries.

## Boundaries

- The semantic-index owner owns source truncation, generation prompt/output
  contracts, corpus/content hashes, and in-memory vector ranking helpers.
- The embedding owner owns hosted embedding calls for semantic-index texts and
  search queries. A separate embedding contract owner owns model contract,
  dimensions, normalization, and score conversion.
- The semantic-search owner owns reranking and candidate/result count policy.
- Feature-local search/read behavior owns its committed semantic-index
  artifacts.
- Mutable context search entries and search behavior have one storage-backed
  owner. Do not confuse that owner with web context resolution.
- The generation command regenerates committed semantic-index artifacts. The
  check command validates that generated artifacts are present, canonical, and
  fresh without regenerating them.
