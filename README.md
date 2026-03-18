# Semantic Search System (Inspired by Endee)

## Project Overview

This project implements a semantic search system inspired by vector databases like Endee
Unlike traditional keyword-based search, this system uses embeddings to understand the meaning of user queries and retrieves the most relevant results based on semantic similarity.

## How It Works

1. User enters a query  
2. The query is converted into a vector representation (embedding)  
3. All stored documents are also represented as vectors  
4. Similarity between query and documents is computed  
5. Top relevant results are returned to the user  

## Features

- Semantic search using embeddings  
- Returns top 3 relevant results  
- Interactive query system  
- Explanation of how results are generated  

## About Endee

Endee is a high-performance open-source vector database built for AI search and retrieval workloads.

It is typically deployed as a server and optimized using C++ for high-speed vector operations.

Due to system-level build requirements (Linux environment and compilation), this project simulates Endee’s core functionality locally using Python.

This demonstrates a clear understanding of how vector databases like Endee perform semantic retrieval.

## Technologies Used

- Python  
- Sentence Transformers  
- NumPy  

## Run Instructions

```bash
pip install -r requirements.txt
python data.py