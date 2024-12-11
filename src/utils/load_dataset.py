import pandas as pd
from datasets import IterableDataset
from datasets import load_dataset as load


# Load the dataset
def load_dataset():
    data = load("mongodb/devcenter-articles", split="train", streaming=True)
    assert isinstance(data, IterableDataset)
    data_head = list(data.take(20))
    docs = pd.DataFrame(data_head).to_dict("records")

    # Check the number of documents in the dataset
    print(f"Number of documents: {len(docs)}")
    # Preview a document
    print(f"preview: {docs[0]}")
