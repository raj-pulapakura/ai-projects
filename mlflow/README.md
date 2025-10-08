# MLflow GenAI Evaluation and Tracing

A comprehensive project demonstrating MLflow's capabilities for Generative AI evaluation and tracing, featuring advanced monitoring, evaluation metrics, and observability for AI applications.

## Overview

This project showcases MLflow's powerful features for managing and monitoring Generative AI applications, including:

- **GenAI Evaluation**: Automated evaluation of AI models with custom and built-in scorers
- **Distributed Tracing**: Comprehensive tracing of AI workflows and function calls
- **Experiment Tracking**: Centralized tracking of AI experiments and runs
- **Model Monitoring**: Real-time monitoring of AI model performance and behavior

## Features

- **Automated GenAI Evaluation**: Built-in and custom evaluation metrics for AI models
- **Distributed Tracing**: End-to-end tracing of complex AI workflows
- **Multi-Provider Support**: Integration with OpenAI, Anthropic, and other AI providers
- **Custom Scorers**: Create and use custom evaluation metrics
- **Agentic Workflow Tracing**: Monitor complex agent-based AI systems
- **Vector Database Integration**: Trace interactions with vector databases
- **Real-time Monitoring**: Live monitoring of AI application performance

## Project Structure

```
mlflow/
├── src/                              # Source code and notebooks
│   ├── genai_evaluation.ipynb      # GenAI evaluation examples
│   └── tracing.ipynb               # Distributed tracing examples
├── mlruns/                          # MLflow experiment runs
├── mlartifacts/                     # MLflow artifacts and traces
├── requirements.txt                 # Python dependencies
├── .gitignore                      # Git ignore rules
└── .venv/                          # Virtual environment
```

## Key Components

### 1. GenAI Evaluation (`genai_evaluation.ipynb`)

Demonstrates MLflow's GenAI evaluation capabilities:

- **Custom Scorers**: Create evaluation metrics like conciseness, correctness
- **Built-in Scorers**: Use MLflow's pre-built evaluation metrics
- **Batch Evaluation**: Evaluate multiple model versions
- **Comparative Analysis**: Compare different model configurations

#### Example Usage:

```python
from mlflow.genai import scorer
from mlflow.genai.scorers import Correctness, Guidelines

# Custom scorer
@scorer
def is_concise(outputs: str) -> bool:
    """Evaluate if the answer is concise (less than 5 words)"""
    return len(outputs.split()) <= 5

# Built-in scorers
scorers = [
    Correctness(),
    Guidelines(name="english_guidelines", guidelines="The response should be in English."),
    is_concise
]

# Run evaluation
results = mlflow.genai.evaluate(
    data=eval_dataset,
    scorers=scorers,
    predict_fn=qa_predict_fn,
)
```

### 2. Distributed Tracing (`tracing.ipynb`)

Comprehensive tracing of AI workflows:

- **Automatic Tracing**: One-line tracing for AI providers
- **Manual Tracing**: Custom tracing with `@mlflow.trace` decorator
- **Agentic Workflows**: Trace complex multi-step AI processes
- **Tool Integration**: Monitor AI tool usage and function calls

#### Example Usage:

```python
import mlflow
from mlflow.entities import SpanType

# Automatic tracing
mlflow.openai.autolog()

# Manual tracing
@mlflow.trace(span_type=SpanType.AGENT)
def run_weather_agent(question: str):
    # Complex AI workflow
    pass

# Tool tracing
@mlflow.trace(span_type=SpanType.TOOL)
def get_weather(latitude, longitude):
    # External API call
    pass
```

## Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd mlflow
```

2. **Create and activate virtual environment:**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables:**
Create a `.env` file with your API keys:
```env
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
```

## Usage

### 1. Start MLflow Server

```bash
mlflow server --host 0.0.0.0 --port 2000
```

Access the MLflow UI at: http://localhost:2000

### 2. Run GenAI Evaluation

Open `src/genai_evaluation.ipynb` and follow the notebook to:

- Set up evaluation datasets
- Create custom scorers
- Run model evaluations
- Compare different model versions

### 3. Explore Distributed Tracing

Open `src/tracing.ipynb` and follow the notebook to:

- Set up automatic tracing
- Create manual traces
- Monitor agentic workflows
- Trace tool integrations

## Key Features in Detail

### GenAI Evaluation

#### Built-in Scorers
- **Correctness**: Evaluate factual accuracy
- **Guidelines**: Check adherence to specific guidelines
- **Relevance**: Measure response relevance
- **Toxicity**: Detect harmful content

#### Custom Scorers
```python
@scorer
def custom_metric(outputs: str) -> float:
    # Your custom evaluation logic
    return score
```

#### Evaluation Workflow
1. Define evaluation dataset
2. Create prediction function
3. Select scorers (built-in + custom)
4. Run evaluation
5. Analyze results in MLflow UI

### Distributed Tracing

#### Span Types
- **LLM**: Large language model calls
- **TOOL**: External tool/API calls
- **AGENT**: Agentic workflow orchestration
- **CHAIN**: Multi-step processing chains
- **RETRIEVER**: Vector database queries

#### Tracing Features
- **Automatic Instrumentation**: One-line setup for AI providers
- **Manual Tracing**: Fine-grained control with decorators
- **Nested Spans**: Hierarchical trace organization
- **Metadata Capture**: Automatic capture of inputs, outputs, and metadata
- **Performance Metrics**: Latency, token usage, and cost tracking

### Experiment Management

#### Experiment Organization
- **Experiments**: Group related runs
- **Runs**: Individual evaluation or training runs
- **Traces**: Detailed execution traces
- **Artifacts**: Models, datasets, and outputs

#### Monitoring Dashboard
- **Real-time Metrics**: Live performance monitoring
- **Trace Visualization**: Interactive trace exploration
- **Comparative Analysis**: Side-by-side run comparison
- **Alerting**: Automated alerts for performance issues

## Advanced Features

### Multi-Provider Support

```python
# OpenAI integration
mlflow.openai.autolog()

# Anthropic integration
mlflow.anthropic.autolog()

# Custom provider integration
@mlflow.trace
def custom_llm_call(prompt):
    # Your custom LLM integration
    pass
```

### Agentic Workflow Tracing

```python
@mlflow.trace(span_type=SpanType.AGENT)
def complex_agent(question: str):
    # Multi-step AI workflow
    # Tool calls, reasoning, planning
    pass
```

### Vector Database Integration

```python
import chromadb

# Trace vector database operations
@mlflow.trace(span_type=SpanType.RETRIEVER)
def search_documents(query: str):
    # Vector similarity search
    pass
```

## Dependencies

### Core MLflow Components
- **mlflow**: Core MLflow functionality
- **mlflow-tracing**: Distributed tracing capabilities
- **mlflow-genai**: GenAI evaluation features

### AI Provider Integrations
- **openai**: OpenAI API integration
- **anthropic**: Anthropic Claude integration
- **litellm**: Multi-provider LLM interface

### Data and Analytics
- **pandas**: Data manipulation
- **numpy**: Numerical computations
- **chromadb**: Vector database
- **requests**: HTTP client

### Development Tools
- **jupyter**: Interactive notebooks
- **python-dotenv**: Environment management
- **rich**: Enhanced terminal output

## Configuration

### MLflow Server Configuration

```bash
# Basic server
mlflow server --host 0.0.0.0 --port 2000

# With database backend
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts

# With S3 artifact storage
mlflow server --default-artifact-root s3://your-bucket/mlflow
```

### Client Configuration

```python
import mlflow

# Set tracking URI
mlflow.set_tracking_uri("http://localhost:2000")

# Set experiment
mlflow.set_experiment("My GenAI Experiment")

# Enable autologging
mlflow.openai.autolog()
```

## Best Practices

### Evaluation
1. **Define Clear Metrics**: Use both quantitative and qualitative measures
2. **Diverse Test Cases**: Include edge cases and failure scenarios
3. **Baseline Comparisons**: Compare against reference implementations
4. **Regular Evaluation**: Set up automated evaluation pipelines

### Tracing
1. **Meaningful Names**: Use descriptive span and trace names
2. **Appropriate Granularity**: Balance detail with performance
3. **Metadata Capture**: Include relevant context and parameters
4. **Error Handling**: Trace and monitor error conditions

### Experiment Management
1. **Organized Experiments**: Group related work logically
2. **Descriptive Names**: Use clear, searchable run names
3. **Tagging**: Use tags for filtering and organization
4. **Artifact Management**: Store models, datasets, and outputs

## Monitoring and Alerting

### Performance Monitoring
- **Latency Tracking**: Monitor response times
- **Token Usage**: Track API consumption and costs
- **Error Rates**: Monitor failure rates and types
- **Quality Metrics**: Track evaluation scores over time

### Alerting Setup
```python
# Custom alerting logic
def check_performance_metrics(run_id):
    # Check if metrics exceed thresholds
    # Send alerts if needed
    pass
```

## Troubleshooting

### Common Issues

1. **Tracing Not Working**: Ensure MLflow server is running and accessible
2. **Evaluation Failures**: Check API keys and model availability
3. **Performance Issues**: Monitor resource usage and optimize batch sizes
4. **Storage Issues**: Ensure sufficient disk space for artifacts

### Debug Mode
```python
import mlflow

# Enable debug logging
mlflow.set_tracking_uri("http://localhost:2000")
mlflow.set_experiment("Debug Experiment")
```

## Future Enhancements

- **Advanced Analytics**: Enhanced visualization and analysis tools
- **Automated Testing**: Automated evaluation and testing pipelines
- **Cost Optimization**: Cost tracking and optimization recommendations
- **Integration Hub**: Extended provider and tool integrations
- **Real-time Monitoring**: Live dashboards and alerting systems

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## License

This project is open source and available under the MIT License.

## References

- [MLflow Documentation](https://mlflow.org/docs/)
- [MLflow GenAI Evaluation](https://mlflow.org/docs/latest/genai/index.html)
- [MLflow Tracing](https://mlflow.org/docs/latest/tracing/index.html)
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Anthropic API Documentation](https://docs.anthropic.com/)

