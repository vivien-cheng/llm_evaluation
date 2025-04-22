#!/usr/bin/env bash
set -e

# Get script directory and project paths
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
EXAMPLE_DIR=$(realpath "$SCRIPT_DIR/..")
PROJECT_ROOT=$(realpath "$EXAMPLE_DIR/../..")

echo "Testing HTA Evaluation Setup..."
echo "Project Root: $PROJECT_ROOT"
echo "Example Directory: $EXAMPLE_DIR"

# Test data directories
echo -n "Checking data directories... "
if [ -d "$EXAMPLE_DIR/data" ] && [ -d "$EXAMPLE_DIR/data/raw" ] && [ -d "$EXAMPLE_DIR/data/processed" ]; then
    echo "OK"
else
    echo "FAILED"
    echo "Error: Data directories not found. Please ensure the following exist:"
    echo "- $EXAMPLE_DIR/data"
    echo "- $EXAMPLE_DIR/data/raw"
    echo "- $EXAMPLE_DIR/data/processed"
    exit 1
fi

# Test Python imports
echo -n "Testing Python imports... "
PYTHONPATH="$PROJECT_ROOT:$EXAMPLE_DIR" python3 -c "
try:
    from hta_evaluation_harness.evaluator import HTAEvaluator
    from hta_evaluation_harness.utils import load_json, save_json
    print('OK')
except ImportError as e:
    print('FAILED')
    print(f'Error: {e}')
    exit(1)
"

# Test dummy data
echo -n "Testing dummy data access... "
if [ -f "$EXAMPLE_DIR/data/raw/longhealth_dummy.jsonl" ]; then
    echo "OK"
else
    echo "FAILED"
    echo "Warning: Dummy data not found at $EXAMPLE_DIR/data/raw/longhealth_dummy.jsonl"
    echo "You may need to create or download the dummy data file."
fi

# Test configuration
echo -n "Testing config access... "
if [ -f "$EXAMPLE_DIR/config/models.yaml" ]; then
    echo "OK"
else
    echo "FAILED"
    echo "Warning: Config file not found at $EXAMPLE_DIR/config/models.yaml"
    echo "You may need to create or configure the models.yaml file."
fi

echo "Setup test complete." 