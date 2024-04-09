import sys
import pandas as pd
from ai.utils import split_into_sentences, create_chunks
from ai.model import get_model


if __name__ == "__main__":
    # Indicates whether command line arguments were provided to specify source and target paths
    optional = False
    
    # Check if the script was called with less than 3 arguments (including the script name)
    if len(sys.argv) < 3:
        # Default CSV file path is used if no arguments are provided
        df = pd.read_csv("./data/medium.csv")
    else:
        # If paths are provided, set 'optional' to True and use the provided paths
        optional = True
        source_path = sys.argv[1]  # The first argument is the source CSV file path
        target_path = sys.argv[2]  # The second argument is the target CSV file save path
        df = pd.read_csv(source_path)  # Read the CSV file from the source path

    # Add an 'ID' column as a unique identifier for each row/entry in the DataFrame
    df['ID'] = range(0, len(df))
    df.set_index('ID')  # Set the 'ID' column as the DataFrame index

    # Convert the DataFrame into a list of dictionaries, where each dictionary represents a row
    records = df.to_dict('records')

    # Split the text in each record into sentences and clean up the text
    # Assumes 'split_into_sentences' is a predefined function
    records = split_into_sentences(records)

    # Organize sentences into chunks of a specified size (here, size 10)
    # Assumes 'create_chunks' is a predefined function
    records = create_chunks(records, 10)

    # Load the embedding model
    # Assumes 'get_model' is a predefined function that loads the model
    embedding_model = get_model()
    
    # Compute embeddings for each chunk
    for item in records:
        item["embedding"] = embedding_model.encode(item["sentence_chunk"])

    # Convert the list of dictionaries into a DataFrame
    chunks_and_embeddings_df = pd.DataFrame(records)

    # Save the DataFrame to a CSV file; the path depends on whether command line arguments were provided
    if optional:
        embeddings_df_save_path = target_path
    else:
        embeddings_df_save_path = "./data/chunks_and_embeddings_df.csv"

    # Write the DataFrame to a CSV file without the index
    chunks_and_embeddings_df.to_csv(embeddings_df_save_path, index=False)