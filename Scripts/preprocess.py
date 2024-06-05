import pandas as pd
import glob

import spacy
from spacy.lang.en import English
nlp = English()
nlp.add_pipe("sentencizer")


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
                expanded_rows.append({
                    "id": row["ID"],
                    "timeline": f"{row['Year']}_{row['Month']}",
                    "title": f"{row['Title']}",
                    "category": col,
                    "chunk": " ".join(joined_chunk_as_sentence)
                })
    return pd.DataFrame(expanded_rows)

        
