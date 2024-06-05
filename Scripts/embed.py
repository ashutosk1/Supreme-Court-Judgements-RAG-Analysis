import constants
from sentence_transformers import SentenceTransformer


embedding_model = SentenceTransformer(model_name_or_path="all-mpnet-base-v2", 
                                      device=constants.DEVICE)

def embed_sentences(sentence):
    """
    Generate Embeddings for a sentence
    """
    return embedding_model.encode(sentence)


