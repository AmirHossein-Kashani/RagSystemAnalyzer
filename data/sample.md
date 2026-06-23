# RAG Sample Document

## What is Retrieval-Augmented Generation?

Retrieval-Augmented Generation (RAG) is a technique that combines information retrieval
with language model generation. Instead of relying purely on parametric knowledge stored
in the model's weights, RAG retrieves relevant passages from an external corpus and uses
them to ground the model's output.

## Why use RAG?

The main motivations for RAG include:

- Keeping knowledge fresh without retraining the model.
- Reducing hallucinations by grounding responses in source documents.
- Supporting private or domain-specific corpora that the base model has never seen.
- Citing sources so users can verify claims against the original text.

## Components of a RAG system

A typical RAG pipeline consists of: a document loader, a chunker, an embedding model,
a vector store, and a retriever. Optionally, a generator LLM consumes the retrieved
chunks to produce a final answer. In a retrieval-only setup, the system simply returns
the top-k most relevant passages and lets the user inspect them directly.

## Embeddings on CPU

Small sentence-transformer models such as all-MiniLM-L6-v2 produce 384-dimensional
embeddings and run efficiently on CPU. They are a sensible default when a GPU is not
available and the corpus size is moderate.

## Chunking strategy

Documents are split into overlapping windows so that semantically related sentences
stay together and boundary information is preserved. Typical chunk sizes range from
512 to 1024 characters, with overlaps of around 10-20% of the chunk size.
