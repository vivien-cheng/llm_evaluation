# LLM Evaluation Harness

## Overview

This repository provides a framework and example pipeline for evaluating the outputs of Large Language Models (LLMs). It includes a core evaluation harness focused on assessing response quality based on metrics inspired by Hierarchical Task Analysis (HTA), along with a practical example demonstrating an end-to-end workflow on the LongHealth clinical dataset.

The primary goal is to offer a structured way to generate, filter, and evaluate LLM responses, particularly for tasks involving complex context understanding and response generation.

## Features

* **`Core Evaluation Harness (hta_evaluation_harness/):`**
    * A reusable Python package for evaluating responses.
    * Defines 5 key metrics: **Comprehensiveness, Task Coverage, Usability, Efficiency, and Accuracy**.
    * Includes an **LLM Judge** component (`llm_judge.py`) capable of using external models (like GPT-3.5/4, Claude) via API calls to score responses against configurable, rubric-based prompts (`prompts.py`).
* **`Example Pipeline (examples/longhealth_semantic_filtering/):`**
    * **Data Processing:** Handles the complex, nested structure of the LongHealth `benchmark_v5.json` dataset (or a simpler dummy dataset).
    * **Response Generation:** Uses Hugging Face `transformers` to generate responses from specified LLMs (currently configured for Qwen 2.5 Instruct models) based on processed data. Includes support for efficient loading (e.g., `bfloat16`).
    * **Semantic Filtering:** Applies embedding-based clustering (`sentence-transformers`, `scikit-learn`) to select the most relevant or diverse responses from multiple generated candidates.
    * **Evaluation:** Utilizes the `hta_evaluation_harness` with the **LLM Judge** to evaluate the final filtered responses against the 5 core metrics using detailed rubrics.

## Repository Structure
```
llm_evaluation/
│
├── .env                       # <-- Stores your secret API keys (DO NOT COMMIT TO GIT)
├── .env.example               # Example environment file
├── .gitignore                 # Specifies intentionally untracked files that Git should ignore
├── config/                    # Configuration files
│   └── models.yaml            # Configuration for models, including LLM Judge settings
├── hta_evaluation_harness/    # Core evaluation package
│   ├── init.py
│   ├── evaluator.py           # Main HTAEvaluator class
│   ├── llm_judge.py           # Class for interacting with LLM Judge APIs (fixed)
│   ├── prompts.py             # Prompts used for LLM Judge evaluation (rubric-based)
│   ├── utils.py               # Basic utilities (e.g., load/save JSON)
│   └── metrics/               # Individual metric evaluation logic (using LLM Judge)
│       ├── init.py
│       ├── accuracy.py
│       ├── comprehensiveness.py
│       ├── efficiency.py
│       ├── task_coverage.py
│       └── usability.py
├── examples/                  # Example pipeline implementations
│   └── longhealth_semantic_filtering/ # Specific example for LongHealth dataset
│       ├── README.md              # Example-specific README
│       ├── requirements.txt       # Python dependencies for this example only
│       ├── data/                  # Data directory for the example
│       │   ├── raw/               # Raw input data
│       │   │   ├── longhealth_dummy.jsonl # Small dummy dataset (updated format)
│       │   │   └── benchmark_v5.json    # Full dataset from LongHealth Github repo
│       │   └── processed/         # Output of data preparation step
│       │       └── longhealth_cleaned.jsonl
│       ├── src/                   # Python source code for example pipeline steps
│       │   ├── init.py
│       │   ├── data_utils.py        # Processes raw data (handles both formats, includes --max_items)
│       │   ├── generate_responses.py  # Generates responses using LLMs (MCQ prompt)
│       │   ├── filter_responses.py    # Filters generated responses (semantic, fixed save)
│       │   └── evaluate_pipeline.py   # Runs evaluation using the harness (fixed ID, loads .env)
│       ├── scripts/               # Shell scripts to run the pipeline
│       │   ├── 0_prepare_data.sh    # Runs data_utils.py or copies dummy data (conditional logic)
│       │   ├── 1_generate_responses.sh # Runs generate_responses.py for models (Temp 0.7)
│       │   ├── 2_filter_responses.sh   # Runs filter_responses.py (Max Resp 1)
│       │   ├── 3_evaluate.sh         # Runs evaluate_pipeline.py (passes config)
│       │   └── run_all.sh            # Runs steps 0-3 sequentially (with comments)
│       └── outputs/               # Directory for pipeline outputs
│           ├── model_responses/     # Raw responses from Step 1
│           │   └── *.jsonl
│           ├── filtered_responses.jsonl # Output from Step 2
│           └── evaluation_results.json  # Final metrics from Step 3
├── README.md                  # This file
├── requirements.txt           # Python dependencies for the core harness (includes dotenv, PyYAML, openai)
└── setup.py                   # Makes the core harness installable (pip install -e .)
```

## Getting Started

### Prerequisites

* Python 3.8+
* Git
* Access to a machine with a GPU (especially NVIDIA) is **highly recommended** for running the generation step with local models. Sufficient VRAM is needed (e.g., >16GB for 7B models, potentially more without quantization). Using cloud GPUs (like Google Colab or Lightning AI) is advised.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd llm_evaluation
    ```

2.  **Set up a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install Core Harness Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(This installs `PyYAML`, `python-dotenv`, `numpy`, and the LLM Judge client library specified in the file, e.g., `openai`)*

4.  **Install the Core Harness Package (Editable Mode):**
    ```bash
    pip install -e .
    ```
    *(This makes `hta_evaluation_harness` importable by the example scripts)*

5.  **Install Example Dependencies:**
    ```bash
    pip install -r examples/longhealth_semantic_filtering/requirements.txt
    ```
    *(This installs `torch`, `transformers`, `accelerate`, `bitsandbytes`, `sentence-transformers`, `scikit-learn`, `pandas`, `tqdm`)*

### Configuration

1.  **API Keys (`.env` file):**
    * Copy `.env.example` to a new file named `.env` in the project root (`llm_evaluation/.env`).
    * Open `.env` and add your API key(s) for the LLM Judge service you intend to use (e.g., OpenAI). **Ensure the variable name matches the one in `config/models.yaml`**.
        ```dotenv
        # Example for OpenAI:
        OPENAI_API_KEY="sk-YOUR_ACTUAL_OPENAI_API_KEY_HERE"
        ```
    * **Important:** Add `.env` to your `.gitignore` file to avoid committing secrets.

2.  **LLM Judge Model (`config/models.yaml`):**
    * Edit `config/models.yaml`.
    * Under the `llm_judge` section, configure the `provider` (e.g., "openai"), `model_name` (e.g., "gpt-3.5-turbo" or "gpt-4-turbo-preview"), and ensure `api_key_env` matches the variable name in your `.env` file.

## Running the LongHealth Example

This example demonstrates the full pipeline on the LongHealth dataset.

1.  **Look at Data Source:**
    * The `benchmark_v5.json` file from the [LongHealth repository](https://github.com/kbressem/LongHealth) is already included.
    * The `longhealth_dummy.jsonl` file is already included for quick testing.

2.  **Navigate to Example Directory:**
    ```bash
    cd examples/longhealth_semantic_filtering
    ```

3.  **Make Scripts Executable (One time):**
    ```bash
    chmod +x scripts/*.sh
    ```

4.  **Choose Data Source & Run:**
    * Edit `scripts/run_all.sh`.
    * **To run the quick dummy test:** Uncomment the line `bash "$SCRIPT_DIR/0_prepare_data.sh" --use-dummy` and comment out the line below it.
    * **To run the real data subset (default 10 items):** Ensure the line `bash "$SCRIPT_DIR/0_prepare_data.sh"` is uncommented and the `--use-dummy` line is commented out. You can adjust the subset size by editing `MAX_ITEMS` inside `scripts/0_prepare_data.sh`.
    * Save `scripts/run_all.sh`.
    * Execute the pipeline:
        ```bash
        bash scripts/run_all.sh
        ```
    * **Warning:** Running even the 10-item subset will take time, especially the first time models are downloaded. Step 1 (Generation) involves loading large models, and Step 3 (Evaluation) involves multiple API calls (~50 calls for 10 items) which take time and may incur costs depending on your chosen judge model. Running the *full* dataset is very time-consuming (many hours).

## Understanding the Pipeline Steps

The `run_all.sh` script executes the following steps:

1.  **`0_prepare_data.sh`**: Prepares the input data. Either copies the dummy file or runs `src/data_utils.py` to process `benchmark_v5.json` (potentially subset) into the required JSONL format (`data/processed/longhealth_cleaned.jsonl`).
2.  **`1_generate_responses.sh`**: Iterates through configured models (e.g., Qwen variants). For each model, it runs `src/generate_responses.py` to generate responses for each query in the processed data file, saving them to `outputs/model_responses/`. Uses Temperature 0.7 by default now.
3.  **`2_filter_responses.sh`**: Runs `src/filter_responses.py`. It gathers all generated responses, uses sentence embeddings (`all-MiniLM-L6-v2`) and clustering to select the top N (default 1) distinct responses per query, saving them to `outputs/filtered_responses.jsonl`.
4.  **`3_evaluate.sh`**: Runs `src/evaluate_pipeline.py`. It loads the filtered responses and the original processed data (for context/query). It initializes the `HTAEvaluator` with the LLM Judge configuration from `config/models.yaml` (and loads API keys from `.env`). For each filtered response, it calls the evaluator, which in turn calls the LLM Judge via API for each of the 5 metrics (Comprehensiveness, Task Coverage, Usability, Efficiency, Accuracy) using the rubric-based prompts. The final scores are saved to `outputs/evaluation_results.json`.

## Current Status & Known Issues

* **End-to-End Functional:** The pipeline runs successfully from data preparation through generation, filtering, and LLM Judge-based evaluation on both dummy and real (subset) data.
* **LLM Judge Integration:** Works correctly, making API calls and returning scores based on the defined rubrics.
* **Evaluation Scores:** The scores obtained from the LLM Judge are real but may appear low (many 1.0s) for certain metrics when evaluating the LongHealth example. This is likely due to the **mismatch between the concise MCQ-style answers generated by the LLMs (following the dataset's nature) and the evaluation rubrics (designed for more descriptive, plan-like text)**, potentially exacerbated by context truncation sent to the judge. This highlights a challenge in applying generic rubrics to specific task outputs.
* **Dataset Flexibility:** The core harness is flexible, but adapting the pipeline to *new* datasets requires modifying `data_utils.py` and the generation prompt format in `generate_responses.py`.
