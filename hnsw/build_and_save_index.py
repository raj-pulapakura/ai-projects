from implementation import HNSW, Node
import pandas as pd
import numpy as np
from pathlib import Path
import json

if __name__ == "__main__":

    # Load dataset
    print("Loading dataset...")
    dataset_path = (
        Path(__file__).resolve().parent / "data" / "flipkart-products-embeddings.csv"
    )
    dataset = pd.read_csv(dataset_path)

    print("Building index...")
    index = HNSW()

    for i, row in dataset.iterrows():
        if i % 1000 == 0:
            print(f"Inserting node {i + 1}/{len(dataset)}")
        embedding = np.array(json.loads(row["description_embedding"]))
        node = Node(
            vector=embedding,
            metadata={
                "id": row["id"],
                "product_name": row["product_name"],
                "brand": row["brand"],
                "description": row["description"],
            },
        )
        index.insert(node)

    print("Saving index to file...")
    index_path = (
        Path(__file__).resolve().parent / "index" / "flipkart_products_index.pkl"
    )
    index.save(index_path)
    print("Index saved at:", index_path)
