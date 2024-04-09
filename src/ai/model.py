import torch
from sentence_transformers import SentenceTransformer

def get_model():
    """
    Initializes and returns a sentence embedding model from the Sentence Transformers
    library, configured to run on GPU if available, otherwise on CPU. The model
    specified is "BAAI/bge-base-en-v1.5", which is a pretrained model provided by
    the Beijing Academy of Artificial Intelligence (BAAI), tailored for generating
    embeddings for English language sentences.

    The function checks the availability of CUDA (GPU support) on the system and
    sets the device accordingly to ensure optimal performance.

    Returns:
    - SentenceTransformer: An instance of the SentenceTransformer model, loaded with
      the specified pretrained model and set to run on the appropriate device.

    Example usage:
        # Simply call the function to get the model
        model = get_model()
        # The model can then be used to encode sentences into embeddings
        embeddings = model.encode(["This is a sentence."])
    """
    # Determine if CUDA (GPU support) is available and set the device accordingly
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize the Sentence Transformer model with the specified pretrained model
    # and configure it to use the determined device (GPU/CPU)
    embedding_model = SentenceTransformer(model_name_or_path="BAAI/bge-base-en-v1.5", device=device)

    # Return the initialized model
    return embedding_model