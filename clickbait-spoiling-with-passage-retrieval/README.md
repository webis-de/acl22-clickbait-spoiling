# Passage Retrieval Approaches for Clickbait Spoiling

This directory contains all the code to extract spoilers for clickbait tweets with passage retrieval approaches.
The idea is to split a document into passages (we tried sentences and paragraphs, but sentences worked better in pilot experiments) and then retrieve the top passages for a clickbait post to spoil the clickbait post. 

- Clickbait spoiling with MonoBERT and MonoT5 
  - Rerank all passages/sentences in the linked document with [clickbait-spoiling-rerank.ipynb](clickbait-spoiling-rerank.ipynb)
- Clickbait spoiling with variants of BM25 and QLD (with various types of query expansion)
  - Creating the Anserini indexes with [build-anserini-indexes.ipynb](build-anserini-indexes.ipynb)
  - Retrieving the spoilers with [clickbait-spoiling-full-rank.ipynb](clickbait-spoiling-full-rank.ipynb)

