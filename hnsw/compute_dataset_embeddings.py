import numpy as np
import pandas as pd
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
import csv

load_dotenv()

if __name__ == "__main__":
    cur_path = Path(__file__).resolve()
    csv_path = cur_path.parent / "data" / "flipkart-products.csv"
    dataset = pd.read_csv(csv_path)
    records = dataset.to_dict(orient="records")
    client = OpenAI()

    print("Computing embeddings for dataset...")
    batch_size = 500
    num_batches = (len(records) + batch_size - 1) // batch_size
    results = []
    for i in range(0, len(records), batch_size):

        print(f"Preparing batch {i // batch_size + 1}/{num_batches}...")
        batch = records[i : i + batch_size]
        descriptions = []
        for record in batch:
            record["description"] = str(record.get("description", ""))
            record["product_name"] = str(record.get("product_name", ""))
            record["brand"] = str(record.get("brand", ""))
            record["id"] = str(record.get("id", ""))
            descriptions.append(record["description"])

        print(f"Processing batch {i // batch_size + 1}/{num_batches}...")
        response = client.embeddings.create(
            input=descriptions, model="text-embedding-3-small"
        )

        for i, record in enumerate(batch):
            record["description_embedding"] = response.data[i].embedding

        results.extend(batch)

    print("Saving embeddings to CSV...")
    output_path = cur_path.parent / "data" / "flipkart-products-embeddings.csv"
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "product_name",
                "brand",
                "description",
                "description_embedding",
            ],
        )
        writer.writeheader()
        writer.writerows(results)
