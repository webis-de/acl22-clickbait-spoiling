# Question Answering Approaches for Clickbait Spoiling

This directory contains all the code to extract spoilers for clickbait tweets with question answering approaches.
The idea is to pretend that the clickbait post is a question whose answer is the spoiler. 

- Clickbait spoiling with AllenAI-Document-QA
  - Run spoiling with the pretrained Model with [run_allenai_on_CBS20.py](run_allenai_on_CBS20.py) in the official repository of [Document-QA](https://github.com/allenai/document-qa)
- Clickbait spoiling with huggingface models
  - Train and predict with supported question answering models on SQuAD with [run_qa_original.py](run_qa_original.py)
  - Train and predict with supported question answering models on Clickbait-Spoiling-21-Corpus with [run_qa.py](run_qa.py)

