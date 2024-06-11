
# Supreme-Court-Judgements-RAG-Analysis-Locally

This project intends to implement a config-driven CLI-based Retrieval-Augmented Generation (RAG) pipeline locally, to utilize the judgements of the Supreme Court of India viz facts, issues and precedents to answer legal queries. The pipeline integrates pre-trained language models (LLMs) and embeddings to provide contextually relevant responses.


Components:

*Retrieval* : 
    
Efficiently locate relevant information from a large corpus of documents. This step involves data-loading and embedding generation. We utilize the `all-mpnet-base-v2` embedding to generate and store multi-dimensional embeddings on a dataset with `title`, `category` (`facts`, `issues`, `precedents`) and `sentence_chunk` (actual text from the judgements).

We use similarity search with `faiss` on the stored embeddings to retrive the top scores and their corresponding indices to get the context from the stored metadata. 

***

*Augmentation*:

Augment the query with relevant context information before it is passed to the generation model. This can be illustrated from the code as below:

```
"""
Use the following context items to answer the 
user query:{context}\nRelevant passages: <extract relevant passages from the context here> User query: {query} Answer:
"""
```
***
*Generation*

Generate coherent and contextually relevant responses based on the augmented information.

We select models based on the system configs which can be accessed through `constants.json` file. The choice of model is `google/gemma-2b-it` or `google/gemma-7b-it` (https://huggingface.co/google/gemma-7b-it).











## Set-up