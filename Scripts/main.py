import preprocess
import constants 
import embed

if __name__ =="__main__":
    df = preprocess.get_data(constants.DATA_DIR)[:5]
    # Split text into sentences and then into chunks
    text_columns = ["Issue", "Facts", "Precedent"]
    for col in text_columns:
        df[f"{col}_Sentences"] = df[col].astype(str).apply(preprocess.text_to_sentences)
        df[f"{col}_Chunks"] = df[f"{col}_Sentences"].apply(lambda x: \
                                                           preprocess.make_sentence_chunks(x, 
                                                                                        constants.NUM_SENTENCES_PER_CHUNK))
        
    # Expand chunks into separate rows with `category`, `id` and `title` as identifier.
    combined_df = preprocess.expand_chunks_to_rows(df, text_columns)
    combined_df = combined_df.dropna()

    # Embed the sentences using pre-trained Embedding Model
    combined_df["embeddings"] = combined_df["chunk"].apply(embed.embed_sentences)
    
    for i in range(10):
        print(len(combined_df.iloc[i]["embeddings"]))



    


