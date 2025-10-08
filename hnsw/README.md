# HNSW (Hierarchical Navigable Small World) Implementation

A Python implementation of the Hierarchical Navigable Small World (HNSW) algorithm for efficient approximate nearest neighbor search, with a practical application as a product search chatbot using Flipkart product data.

## Overview

This project implements the HNSW algorithm from scratch, which is a graph-based approximate nearest neighbor search algorithm that provides logarithmic time complexity for both insertion and search operations. The implementation includes a complete product search chatbot that demonstrates the algorithm's practical application.

## Features

- **Complete HNSW Implementation**: Custom implementation of the HNSW algorithm with all core components
- **Product Search Chatbot**: Interactive chatbot powered by OpenAI GPT-4 for product recommendations
- **Vector Embeddings**: Integration with OpenAI's text-embedding-3-small model for semantic search
- **Efficient Indexing**: Optimized for large-scale product datasets
- **Persistent Storage**: Save and load HNSW indices for reuse

## Project Structure

```
hnsw/
├── implementation.py          # Core HNSW algorithm implementation
├── chatbot.py                # Interactive chatbot with product search
├── main.py                   # Entry point for the chatbot
├── build_and_save_index.py   # Script to build and save HNSW index
├── compute_dataset_embeddings.py  # Generate embeddings for product data
├── requirements.txt          # Python dependencies
├── data/                     # Dataset directory
│   └── flipkart-products.csv # Product dataset
└── index/                    # Saved HNSW indices (created during execution)
```

## Algorithm Components

### Core Classes

- **`Node`**: Represents a data point with vector and metadata
- **`DistanceIdPair`**: Helper class for distance-based comparisons
- **`HNSW`**: Main implementation with the following key methods:
  - `insert()`: Add new nodes to the graph
  - `search()`: Find k nearest neighbors
  - `search_layer()`: Search within a specific layer
  - `select_neighbours()`: Optimize neighbor selection

### Key Parameters

- **M**: Maximum number of neighbors per node (default: 24)
- **M_0**: Maximum neighbors for bottom layer (default: 48)
- **efConstruction**: Beam search size during construction (default: 200)
- **efSearch**: Beam search size during query (default: 200)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd hnsw
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file with your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

## Usage

### 1. Generate Embeddings

First, compute embeddings for your product dataset:

```bash
python compute_dataset_embeddings.py
```

This script:
- Loads the Flipkart products dataset
- Generates embeddings using OpenAI's text-embedding-3-small model
- Saves the results to `data/flipkart-products-embeddings.csv`

### 2. Build the HNSW Index

Create and save the HNSW index:

```bash
python build_and_save_index.py
```

This script:
- Loads the embeddings dataset
- Builds the HNSW index with all products
- Saves the index to `index/flipkart_products_index.pkl`

### 3. Run the Chatbot

Start the interactive product search chatbot:

```bash
python main.py
```

The chatbot provides:
- Natural language product search
- Semantic similarity matching
- Product recommendations based on user queries

## Example Usage

```
You: I'm looking for wireless headphones under 5000 rupees
Chatbot: Based on your search, here are some wireless headphones under ₹5000:

1. Sony WH-CH720N Wireless Headphones - Premium noise cancellation
2. JBL Tune 760NC Wireless Headphones - Active noise cancellation
3. Boat Rockerz 450 Wireless Headphones - Bass-heavy sound
...

You: What about gaming keyboards?
Chatbot: Here are some gaming keyboards that might interest you:

1. Razer BlackWidow V3 Gaming Keyboard - Mechanical switches
2. Logitech G413 Gaming Keyboard - Tactile switches
3. Corsair K70 RGB Gaming Keyboard - Cherry MX switches
...
```

## Technical Details

### HNSW Algorithm

The implementation follows the original HNSW paper with these key features:

1. **Hierarchical Structure**: Multi-layer graph with decreasing density
2. **Greedy Search**: Efficient nearest neighbor search
3. **Neighbor Selection**: Optimized connection strategy
4. **Layer Assignment**: Probabilistic level assignment

### Performance Characteristics

- **Insertion Time**: O(log N) average case
- **Search Time**: O(log N) average case
- **Memory Usage**: O(N) for N data points
- **Search Quality**: High recall with proper parameter tuning

### Customization

You can adjust the algorithm parameters in `implementation.py`:

```python
def __init__(self):
    self.M = 24                    # Max neighbors per node
    self.M_0 = self.M * 2          # Max neighbors for bottom layer
    self.efConstruction = 200      # Construction beam size
    self.mL = 1 / np.log(self.M)   # Level normalization factor
```

## Dependencies

- **numpy**: Numerical computations
- **pandas**: Data manipulation
- **openai**: Embedding generation and chat completion
- **python-dotenv**: Environment variable management

## Dataset

The project uses a Flipkart products dataset containing:
- Product names
- Brand information
- Descriptions
- Product IDs

The dataset is processed to generate embeddings for semantic search capabilities.

## Future Enhancements

- Support for different embedding models
- Batch processing optimizations
- Real-time index updates
- Performance benchmarking tools
- Web interface for the chatbot

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## References

- [Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs](https://arxiv.org/abs/1603.09320)
- [OpenAI Embeddings API](https://platform.openai.com/docs/guides/embeddings)
- [OpenAI Chat Completions API](https://platform.openai.com/docs/guides/text-generation)
