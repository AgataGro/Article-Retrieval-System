from spacy.lang.en import English
import re
import pandas as pd
import textwrap

def split_into_sentences(entries:list[dict[str, any]]):
    """
    Splits the text within each entry of a list of dictionaries into sentences,
    using spaCy's English model and sentencizer pipeline. Each dictionary must have a 
    key "Text" with the text to be split. The function updates each entry in the list 
    with two new keys: "sentences", a list of the extracted sentence strings; and 
    "sentence_count_spacy", the count of identified sentences.

    Parameters:
    - entries (List[Dict[str, Any]]): A list of dictionaries, each containing a "Text" key
      with the text to be processed.

    Returns:
    - List[Dict[str, Any]]: The input list of dictionaries, updated with "sentences" and
      "sentence_count_spacy" for each entry.

    Example usage:
        entries = [
            {"Text": "Hello world. This is a test.\nNew line here."},
            {"Text": "Another entry here. It also has sentences."}
        ]
        processed_entries = split_into_sentences(entries)
        for entry in processed_entries:
            print(entry["sentences"])
            print("Sentence count:", entry["sentence_count_spacy"])
    """

    # Initialize spaCy's English language model
    nlp = English()

    # Add a sentencizer pipeline to the model for sentence segmentation
    nlp.add_pipe("sentencizer")

    # Process each entry in the provided list
    for item in entries:
        # Clean the text by removing carriage returns and new lines
        clean_text = item["Text"].replace('\r', '').replace('\n', ' ')

        # Use spaCy's model to segment the cleaned text into sentences
        doc = nlp(clean_text)

        # Extract sentences as strings from the spaCy document
        item["sentences"] = [str(sentence) for sentence in doc.sents]

    # Return the updated list of entries
    return entries


def split_list(input_list: list, slice_size: int) -> list[list[str]]:
    """
    Splits a list into sublists, each of a specified maximum size. This function 
    iterates through the input list, creating a new sublist for every 'slice_size' 
    elements, until all elements of the input list have been placed into sublists. 
    If the total number of elements in the input list is not perfectly divisible by 
    'slice_size', the final sublist will contain the remaining elements and may be 
    smaller than 'slice_size'.

    Parameters:
    - input_list (list): The list to be split. It can contain elements of any type, 
      but the function signature implicitly suggests strings. In practice, it works 
      with any type.
    - slice_size (int): The maximum size of each sublist. Must be a positive integer.

    Returns:
    - list[list[str]]: A list of sublists, where each sublist contains up to 'slice_size' 
      elements from the input list. The type hint suggests sublists of strings, but 
      the actual type will match the elements of 'input_list'.

    Example usage:
        input_list = ['a', 'b', 'c', 'd', 'e']
        slice_size = 2
        # This will return [['a', 'b'], ['c', 'd'], ['e']]
        print(split_list(input_list, slice_size))
    """
    return [input_list[i:i + slice_size] for i in range(0, len(input_list), slice_size)]


def create_chunks(entries:list[dict[str, any]], k:int):
    """
    Processes a list of entries to organize sentences into chunks of specified size,
    then compiles these chunks along with their metadata into a list of dictionaries.
    Each entry is expected to have sentences, an ID, and a Title. The function
    splits sentences into chunks, then for each chunk, it creates a dictionary
    capturing the chunk, title and ID.

    Parameters:
    - entries (List[Dict[str, Any]]): A list of dictionaries, where each dictionary
      represents an entry with at least "ID", "Title", and "sentences" keys.
    - k (int): The maximum number of sentences per chunk.

    Returns:
    - List[Dict[str, Any]]: A list of dictionaries, where each dictionary represents
      a chunk with its metadata, including the original ID, Title, the chunk of
      sentences combined into a single string.

    The function also adjusts sentence spacing and formatting, ensuring that sentences
    are properly spaced and that there is a space following any period that is
    directly followed by a capital letter, addressing common punctuation issues.

    Example usage:
        entries = [
            {
                "ID": "1",
                "Title": "Sample Entry",
                "sentences": ["Sentence one.", "Sentence two.", "Sentence three."]
            }
        ]
        chunks = create_chunks(entries, 2)
        
      This will process 'entries' to create sentence chunks of up to 2 sentences,
      along with metadata for each chunk.
    """

    # Split sentences into chunks of size 'k' for each entry
    for item in entries:
        item["sentence_chunks"] = split_list(input_list=item["sentences"], slice_size=k)
        item["num_chunks"] = len(item["sentence_chunks"])

    record_chunks = []
    for item in entries:
        for sentence_chunk in item["sentence_chunks"]:
            chunk_dict = {}
            chunk_dict["ID"] = item["ID"]
            chunk_dict["Title"] = item["Title"]

            # Join sentences in the chunk into a paragraph-like structure
            joined_sentence_chunk = "".join(sentence_chunk).replace("  ", " ").strip()
            # Ensure proper spacing after periods before capital letters
            joined_sentence_chunk = re.sub(r'\.([A-Z])', r'. \1', joined_sentence_chunk)

            # Populate the chunk dictionary with the chunk and its statistics
            chunk_dict["sentence_chunk"] = joined_sentence_chunk     
            record_chunks.append(chunk_dict)

    # Convert the list of chunk dictionaries into a DataFrame and then to a list of records
    df = pd.DataFrame(record_chunks)
    records = df.to_dict(orient="records")
    return records

def print_wrapped(text, wrap_length=80):
    """
    Prints the given text to the console, wrapping it at a specified number of characters.
    This ensures that each line of the text does not exceed the wrap_length limit, making
    the output more readable, especially for long strings.

    The function uses the `textwrap.fill` method from Python's standard library to
    automatically wrap the text. This method breaks the text into lines, each having
    a maximum width of `wrap_length`. It then prints the wrapped text to the console.

    Parameters:
    - text (str): The text string to be printed. This can be of any length.
    - wrap_length (int, optional): The maximum number of characters that each line
      of the printed text should contain. Defaults to 80 characters.

    Returns:
    - None

    Example usage:
        sample_text = "This is a long text string that will be wrapped and printed \
                       to the console so that each line does not exceed a certain \
                       number of characters."
        print_wrapped(sample_text, wrap_length=50)
    """
    # Wrap the input text at the specified length
    wrapped_text = textwrap.fill(text, wrap_length)

    # Print the wrapped text to the console
    print(wrapped_text)