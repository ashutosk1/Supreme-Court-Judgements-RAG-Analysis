import preprocess
import constants 
import rag


if __name__ =="__main__":

    query ="Fundamental Right"
    result = preprocess.load_embeddings_and_metadata(constants.SAVED_DIR)
    if result is None:
        print("[INFO] No stored embeddings found! Trying to create, store and load again.")
        preprocess.load_preprocess_and_embed()
        rag.get_similiarity_search(query, constants)
    else:
        rag.get_similiarity_search(query, constants)





    


