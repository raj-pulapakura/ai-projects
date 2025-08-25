# Azure AI Foundry Product RAG Assistant

A Retrieval-Augmented Generation (RAG) system built with Azure AI Foundry that provides intelligent product recommendations and information for outdoor/camping gear and clothing. This system uses semantic search and AI-powered chat to help users find the right products based on their queries.

## 🚀 Features

- **Intelligent Product Search**: Uses semantic search with vector embeddings to find relevant products
- **AI-Powered Chat**: GPT-4 powered chat interface that provides grounded responses based on product data
- **Intent Mapping**: Automatically maps user queries to search intent for better results
- **Azure AI Search Integration**: Leverages Azure AI Search for fast and accurate product retrieval
- **Telemetry & Monitoring**: Built-in observability with Azure Monitor and OpenTelemetry
- **Prompt Engineering**: Uses Prompty templates for consistent and maintainable AI interactions

## 🏗️ Architecture

The system consists of several key components:

1. **Search Index Creation** (`create_search_index.py`) - Sets up Azure AI Search with vector search capabilities
2. **Product Document Retrieval** (`get_product_documents.py`) - Handles semantic search and document retrieval
3. **Chat Interface** (`chat_with_products.py`) - Main chat functionality with RAG integration
4. **Configuration** (`config.py`) - Environment setup and telemetry configuration
5. **Prompt Templates** (`assets/`) - Prompty files for consistent AI interactions

## 📋 Prerequisites

- Azure subscription with access to Azure AI Foundry
- Python 3.8 or higher
- Azure CLI installed and configured
- Product data in CSV format (see `assets/products.csv` for example structure)

## 🛠️ Installation

1. **Clone the repository**:

   ```bash
   git clone <your-repo-url>
   cd azure-ai-foundry-product-rag-assistant
   ```

2. **Create and activate a virtual environment**:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   Create a `.env` file in the project root with the following variables:
   ```env
   AIPROJECT_CONNECTION_STRING=
   AISEARCH_INDEX_NAME=
   EMBEDDINGS_MODEL=
   INTENT_MAPPING_MODEL=
   CHAT_MODEL=
   ```

## 🚀 Quick Start

### 1. Create Search Index

First, create the Azure AI Search index with vector search capabilities:

```bash
python create_search_index.py --index-name "products" --model "text-embedding-3-small"
```

### 2. Chat with Products

Start a conversation about outdoor gear:

```bash
python chat_with_products.py --query "I need a waterproof tent for 4 people"
```

Or run with telemetry enabled:

```bash
python chat_with_products.py --query "What are the best hiking boots?" --enable-telemetry
```

## 📁 Project Structure

```
azure-ai-foundry-product-rag-assistant/
├── assets/
│   ├── grounded_chat.prompty      # Chat system prompt template
│   ├── intent_mapping.prompty     # Intent extraction prompt template
│   └── products.csv               # Sample product data
├── chat_with_products.py          # Main chat interface
├── config.py                      # Configuration and telemetry setup
├── create_search_index.py         # Search index creation
├── get_product_documents.py       # Document retrieval logic
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 🔧 Configuration

### Environment Variables

| Variable                      | Description                                | Required |
| ----------------------------- | ------------------------------------------ | -------- |
| `AIPROJECT_CONNECTION_STRING` | Azure AI Foundry project connection string | Yes      |
| `CHAT_MODEL`                  | Chat completions model deployment name     | Yes      |
| `EMBEDDINGS_MODEL`            | Embeddings model deployment name           | Yes      |
| `INTENT_MAPPING_MODEL`        | Model for intent mapping                   | Yes      |
| `AISEARCH_INDEX_NAME`         | Azure AI Search index name                 | Yes      |

### Azure AI Foundry Setup

1. **Create an AI Project** in Azure AI Foundry
2. **Configure connections** for Azure AI Search
3. **Deploy models** for chat completions and embeddings
4. **Set up Application Insights** for telemetry (optional)

## 🎯 Usage Examples

### Basic Product Query

```bash
python chat_with_products.py --query "What's the best sleeping bag for cold weather?"
```

### Specific Product Information

```bash
python chat_with_products.py --query "Tell me about the waterproof rating of the Alpine Explorer Tent"
```

### Price Inquiries

```bash
python chat_with_products.py --query "How much do the TrailWalker hiking shoes cost?"
```

## 🔍 How It Works

1. **User Query**: User asks a question about outdoor gear
2. **Intent Mapping**: AI extracts the search intent from the conversation
3. **Semantic Search**: Query is embedded and used for vector search in Azure AI Search
4. **Document Retrieval**: Relevant product documents are retrieved based on semantic similarity
5. **Grounded Response**: AI generates a response based on the retrieved documents
6. **Context Maintenance**: Conversation context is maintained for follow-up questions

## 📊 Monitoring & Telemetry

The system includes comprehensive monitoring capabilities:

- **OpenTelemetry Integration**: Automatic tracing and metrics collection
- **Azure Monitor**: Application Insights integration for production monitoring
- **Structured Logging**: Detailed logging for debugging and analysis
- **Performance Metrics**: Search performance and response time tracking

To enable telemetry:

```bash
python chat_with_products.py --enable-telemetry
```

## 🧪 Testing

Test the system with various query types:

```bash
# Test basic functionality
python chat_with_products.py --query "What camping gear do you have?"

# Test specific product queries
python chat_with_products.py --query "Is the Mountain Pro tent waterproof?"

# Test price inquiries
python chat_with_products.py --query "How much does the Explorer backpack cost?"
```
