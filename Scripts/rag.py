import constants
import preprocess

import numpy as np
import faiss


def get_similiarity_search(query, constants):
    """Searches for nearest neighbors in pre-processed embeddings for a given query."""

    embeddings, metadata = preprocess.load_embeddings_and_metadata(constants.SAVED_DIR)
    embedding_model = preprocess.get_embedding_model(constants.MODEL_NAME, constants.DEVICE)

    query_embedding = np.array(embedding_model.encode(query))
    
    if query_embedding.ndim == 1:
        query_embedding = np.expand_dims(query_embedding, axis=0)

    # Validate Dimensions
    assert len(query_embedding[0]) == embeddings.shape[1]

    # faiss search
    embed_dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(embed_dimension)
    index.add(embeddings)

    distances, indices = index.search(query_embedding, constants.NUM_NEAREST_SEARCH)

    for index in indices:
        print(f"Nearest Metadata:{metadata.iloc[index]}")

    

