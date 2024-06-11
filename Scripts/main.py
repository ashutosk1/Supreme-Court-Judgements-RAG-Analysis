import preprocess
import constants 
import llm
from similarity_search import get_similarity_search


if __name__ =="__main__":


    
    query ="What are the precedents related to the issue of freedom of speech?"
    result = preprocess.load_embeddings_and_metadata(constants.SAVED_DIR)
    if result is None:
        print("[INFO] No stored embeddings found! Trying to create, store and then load.")
        preprocess.load_preprocess_and_embed()
        indices = get_similarity_search(query, 
                              embed_dir  = constants.SAVED_DIR, 
                              model_name = constants.MODEL_NAME, 
                              device_name= constants.DEVICE, 
                              num_nearest_search = constants.NUM_NEAREST_SEARCH)
        
    else:
        indices = get_similarity_search(query, 
                              embed_dir  = constants.SAVED_DIR, 
                              model_name = constants.MODEL_NAME, 
                              device_name= constants.DEVICE, 
                              num_nearest_search = constants.NUM_NEAREST_SEARCH)


