# LongHealth Example

This example demonstrates the use of the HTA Evaluation Harness for evaluating Hierarchical Task Analysis (HTA) in the context of long health-related tasks.

## Directory Structure

- `src/`: Source code for the example pipeline
- `scripts/`: Utility scripts
- `outputs/`: Generated outputs and results
- `data/`: Input data and ground truth
- `config/`: Configuration files

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up your environment variables (if needed):
```bash
cp .env.example .env
# Edit .env with your API keys and configurations
```

## Usage

The example includes several components:

1. Response Generation
2. Semantic Filtering
3. Evaluation Pipeline
4. Analysis Tools

Refer to the individual Python files in the `src/` directory for specific usage instructions.

## Notes

- This example combines functionality from both the original LongHealth example and the semantic filtering extension
- Some features may require additional API keys or configurations
- Check the individual script documentation for specific requirements 