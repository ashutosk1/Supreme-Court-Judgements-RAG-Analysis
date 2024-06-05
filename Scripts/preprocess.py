import glob
import os
import numpy as np
import pandas as pd
from spacy.lang.en import English
nlp = English()
nlp.add_pipe("sentencizer")
from sentence_transformers import SentenceTransformer
import constants

def get_data(data_dir):
    """
    Get all the sheets from the `data_dir`, merge them and read as a single DataFrame.
    """
    all_data_files = glob.glob(data_dir+"/*.xlsx")
    print(f"[INFO] {len(all_data_files)} data sheets detected.")
    merged_df = []

    for data_file in all_data_files:
        df = pd.read_excel(data_file)
        merged_df.append(df)
    
    merged_df = pd.concat(merged_df, ignore_index=True)
    print(f"[INFO] Total {(merged_df.shape[0])} rows merged.")
    return merged_df


def text_to_sentences(text):
    """
    Return a list of sentences split from the text.
    """
    return [sent.text for sent in nlp(text).sents]


def make_sentence_chunks(list_of_sentences, num_senetences_per_chunk):
    """
    Splits the input_list into sublists of chunk_size.
    """
    list_of_chunks =[]
    for i in range(0, len(list_of_sentences), num_senetences_per_chunk):
        list_of_chunks.append(list_of_sentences[i:i+num_senetences_per_chunk])
    return list_of_chunks


def expand_chunks_to_rows(df, text_columns):
    expanded_rows = []
    for idx, row in df.iterrows():
        for col in text_columns:
            chunks = row[f"{col}_Chunks"]
            joined_chunks_as_sentences = [[" ".join(chunk)] for chunk in chunks]
            for joined_chunk_as_sentence in joined_chunks_as_sentences:
                sentence_chunk =  " ".join(joined_chunk_as_sentence)
                expanded_rows.append({
                    "id": row["ID"],
                    "timeline": f"{row['Year']}_{row['Month']}",
                    "title": f"{row['Title']}",
                    "category": col,
                    "sentence_chunk": sentence_chunk, 
                    "chunk_token_count" : len(sentence_chunk)/4
                })
    return pd.DataFrame(expanded_rows)


def get_embedding_model(model_name, device):
    return SentenceTransformer(model_name_or_path=model_name, device=device)


def get_embeddings(list_of_chunks, model_name, batch_size, device):
    """
    Generate Embeddings for a list of sentences in a batch and return then as Tensors.
    """
    embedding_model = get_embedding_model(model_name, device)
    return embedding_model.encode(list_of_chunks, batch_size=batch_size, convert_to_tensor=True)


def save_embeddings_with_metadata(list_of_embeddings, df, save_dir):
    """
    Saves the Embeddings and the Metadata for memory and easy retreival. 
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    embed_path = os.path.join(save_dir, "embeddings.npz")
    metadata_path = os.path.join(save_dir, "metadata.csv")

    # Extract embeddings and Metadata from the DataFrame
    assert (list_of_embeddings.shape[0]) == df.shape[0]  # Number of embeddings are same as number of sentence_chunks

    # Save Embeddings
    embeddings = np.array(list_of_embeddings)
    np.savez(embed_path, embeddings=embeddings)

    # Save Metadata
    metadata = df.drop(columns="sentence_chunk")
    metadata.to_csv(metadata_path, index=False)
    print(f"[INFO] Embeddings and Metadata saved at {embed_path} and {metadata_path} respectively.")


def load_embeddings_and_metadata(save_dir):
    """
    Load the embeddings and Metadata from the saved directory.
    """
    if not os.path.exists(save_dir):
        print(f"[WARNING] Cannot load the embeddings as {save_dir} does not exist.")
        return None
    
    embed_path = os.path.join(save_dir, "embeddings.npz")
    metadata_path = os.path.join(save_dir, "metadata.csv")

    if not (os.path.exists(embed_path) and os.path.exists(metadata_path)):
        print(f"[WARNING] Either one or both of the Embeddings and Metadata files do not exist. Try saving them first.")
        return None
    
    # Load Embeddings
    loaded = np.load(embed_path)
    embeddings = loaded["embeddings"]

    # Load Metadata
    metadata = pd.read_csv(metadata_path)

    print(f"[INFO] Embeddings and Metadata files loaded successfully.")
    return embeddings, metadata


def load_preprocess_and_embed():
    """
    Loads data, preprocesses text, generates embeddings, and saves them.

    """
    df = get_data(constants.DATA_DIR)[:constants.NUM_EXAMPLES]
    # Split text into sentences and then into chunks
    text_columns = ["Issue", "Facts", "Precedent"]
    for col in text_columns:
        df[f"{col}_Sentences"] = df[col].astype(str).apply(text_to_sentences)
        df[f"{col}_Chunks"] = df[f"{col}_Sentences"].apply(lambda x: \
                                                           make_sentence_chunks(x, 
                                                                                        constants.NUM_SENTENCES_PER_CHUNK))
        
    # Expand chunks into separate rows with `category`, `id` and `title` as identifier.
    combined_df = expand_chunks_to_rows(df, text_columns)
    combined_df = combined_df[combined_df["chunk_token_count"] > constants.MIN_TOKEN_LEN]       # Filter chuks with little to no information


    # Embed the sentences using pre-trained Embedding Model
    # Steps: Generate a single list of all the `sentence_chunk` and tokenize them in batches
    
    all_sentence_chunks = combined_df["sentence_chunk"].tolist()
    print(f"[INFO] Total {len(all_sentence_chunks)} sentence chunks to be tokenized.")
    all_sentence_chunks_embeddings = get_embeddings(all_sentence_chunks, constants.MODEL_NAME, 
                                                               constants.BATCH_SIZE, 
                                                               constants.DEVICE)
    print(f"[INFO] Number of chunks: {all_sentence_chunks_embeddings.shape[0]}, Shape of Embeddings:{all_sentence_chunks_embeddings.shape[1]}")
    
    # Store the embeddings
    save_embeddings_with_metadata(all_sentence_chunks_embeddings, combined_df, constants.SAVED_DIR)





