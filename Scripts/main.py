from preprocess import load_embeddings_and_metadata, load_preprocess_and_embed
from llm import get_llm_selection, get_llm_model, ask_llm
import torch
import json
import argparse

def load_json_config(file_path):
    """
    Loads JSON configuration from the given file_path.
    """
    try:
        with open(file_path, 'r') as f:
            config = json.load(f)
    except Exception as e:
        raise Exception(f" [ERROR]: {e}")
    return config


def main(query, constants):
    # Get constants from the json file
    data_dir = constants["DATA_DIR"]
    saved_dir = constants["SAVED_DIR"]
    constants["DEVICE"] = "cuda" if torch.cuda.is_available() else "cpu"
    device = constants["DEVICE"]

    # Accessing PARAMS dictionary
    num_examples = constants["PARAMS"]["num_examples"]
    num_sentences_per_chunk = constants["PARAMS"]["num_sentences_per_chunk"]
    min_token_len = constants["PARAMS"]["min_token_len"]
    batch_size = constants["PARAMS"]["batch_size"]
    num_nearest_search = constants["PARAMS"]["num_nearest_search"]

    # Accessing EMBED_MODEL
    embed_model = constants["EMBED_MODEL"]

    # Accessing LLM dictionary
    llm_model_name = constants["LLM"]["model_name"]
    use_quantization = constants["LLM"]["use_quantization"]
    max_new_tokens = constants["LLM"]["max_new_tokens"]
    temperature = constants["LLM"]["temperature"]

    # Dump any modified config
    with open("./constants.json", "w") as file:
      json.dump(constants, file, indent=4)

    query ="What are the precedents related to the issue of freedom of speech?"
    result = load_embeddings_and_metadata(saved_dir)
    if result is None:
        print("[INFO] No stored embeddings found! Trying to create, store and then load.")
        load_preprocess_and_embed(data_dir,
                                    saved_dir,
                                    num_examples,
                                    embed_model,
                                    batch_size,
                                    min_token_len,
                                    num_sentences_per_chunk,
                                    device)
        
    # Get the LLM model id and update the settings for further retreival of the model.
    get_llm_selection(constants)

    # Based on the settings and get the tokenizer and the pre-trained model
    tokenizer, llm_model = get_llm_model(llm_model_name, 
                                         use_quantization, 
                                         device, 
                                         set_flash_attention = True)
    
    # Ask the LLM Model for the query
    ask_llm(query,
            tokenizer,
            llm_model,
            temperature,
            max_new_tokens,
            device,
            format_answer_text=True, 
            return_answer_only=True)


if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("query", type=str, help="Enter your query for the RAG Pipeline.")
    args = parser.parse_args()
    query = args.query
    constants = load_json_config("./constants.json")
    main(query, constants)

