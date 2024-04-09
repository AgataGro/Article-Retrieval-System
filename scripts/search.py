import pandas as pd
import numpy as np
import sys
import hnswlib
import time
from ai.utils import print_wrapped  
from ai.model import get_model  


if __name__ == "__main__":
    # Load the data from CSV file
    if len(sys.argv) < 2:
        df = pd.read_csv("./data/chunks_and_embeddings_df.csv")
    else:
        source_path = sys.argv[1]  # The first argument is the source CSV file path
        df = pd.read_csv(source_path)  # Read the CSV file from the source path

    # Convert the 'embedding' column back to numpy arrays from strings
    df["embedding"] = df["embedding"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))
    records = df.to_dict('records')
    embeddings = np.array(df["embedding"].tolist())

    # Create the HNSW index for efficient similarity search
    dim = embeddings.shape[1]  # Dimensionality of the vectors
    p = hnswlib.Index(space='cosine', dim=dim)
    p.init_index(max_elements=embeddings.shape[0], ef_construction=200, M=16)
    p.add_items(embeddings)
    p.set_ef(50)  # Higher ef leads to more accurate but slower search

    # Load the embedding model
    embedding_model = get_model()

    print("Type 'exit' to terminate")
    while True:
        user_input = input("Please enter query: ")
        if user_input == "exit":
            break
        start_time = time.time()  # Start time for query processing
        query_embedding = embedding_model.encode(user_input)
        ids, distances = p.knn_query(query_embedding, k=5)

        # Calculate and print the time taken to retrieve results
        end_time = time.time()
        print(f"Query: '{user_input}'\n")
        print(f"Results retrieved in {end_time - start_time} seconds:\n")
        print("Results:")

        # Loop through zipped distances and IDs
        for score, idx in zip(distances[0], ids[0]):
            print(f"Distance: {score}")
            print("Title:")
            print_wrapped(records[idx]["Title"])
            print("Text:")
            print_wrapped(records[idx]["sentence_chunk"])
            print(f"ID: {records[idx]['ID']}\n")

