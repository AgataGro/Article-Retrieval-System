# Article-Retrieval-System

Nokia - Machine Learning Summer Trainee - recruitment task

# Environment setup

1. Create new virtual environment:

`conda create --name rag python=3.10`

2. Activate environment:

`conda activate rag`

3. Update _pip_ version:

`python -m pip install --upgrade pip`

4. Install required packages

`python -m pip install -e .[dev] --extra-index-url https://download.pytorch.org/whl/cu121`

# Usage

## 1. Preprocess the data

1. Default usage:

   `python .\scripts\preprocess.py`

2. Optional usage:

   `python .\scripts\preprocess.py <source_file_path> <target_file_path>`

The script `preprocess.py` loads a .csv file and with a structure:

| Title         | Text         |
| ------------- | ------------ |
| Example title | Example Text |
| ...           | ...          |

Cleans up the text and divides it into chunks of 10 sentences. The result is written into a .csv file with a structure:

| ID  | Title | sentence_chunk | embedding |
| --- | ----- | -------------- | --------- |
| ... | ...   | ...            | ...       |

## 2. Search queries

1. Default usage:

   `python .\scripts\search.py`

2. Optional usage:

   `python .\scripts\search.py <source_file_path>`

After loading the dataset the program will enter a loop where the user can repeatedly write queries to be searched. The loop will end upon writing the `exit` keyword.

The model will return the top 5 results. The lower the distance the more similar the retrieved chunk is to the query.

# Project structure

The `data` directory contains the "1300 Towards Data Science Medium Articles" dataset, by default the result of the data preprocessing will also be saved here.

The `scripts` directory contains the scripts for dataset preprocessing as well as query searching.

The `src\ai` directory contains 2 files

      model.py contains the model used for generating vector embeddings

      utils.py contains various helper functions
