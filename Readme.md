# CyberPlaybookLLM - Automated Issue Tracking from User Feedback

This repository implements the core components described in the paper:
**"Semantic Feedback Processing with LLMs: Automating Issue Detection and Prioritization in DevOps", KES 2025.**

The system automatically links user feedback (e.g., Reddit posts) to existing Jira issues or generates new ones using a fine-tuned LLaMA 3 model. The pipeline includes summarization, sentiment filtering, semantic matching, and CACAO playbook generation. Optionally, it can deployed to Azure to your workspace.


---

## Project Structure

- folder `srcdata`: Contains scripts for gathering data from Reddit and upload issues to a Jira client. We prepared a small sample demo of files (raw, and processed using the scripts below) in folder `data`.
- `data_preparation.py`: Gathers and formats Reddit + Jira feedback into training samples.
- `train_pipeline.py`: Fine-tunes a LLaMA 3 model (via HuggingFace Trainer + PEFT) on processed data.
- `train_pipeline_bertVersion.py`: As above but uses facebook/bert model and Seq2Seq trainer from HuggingFace.
- `inference_pipeline.py`: Performs inference on new feedback to output summarized, matched, or new tickets.
- `azure_deploy.bicep`: Deploys an Azure Function app to serve the model via container.
- `README.md`: This documentation.

---

## Training Pipeline

```bash
python train_pipeline.py \
    --model_path meta-llama/Meta-Llama-3-8B-Instruct \
    --dataset_path ./data/processed_dataset.json \
    --output_dir ./checkpoints/llama3-feedback
```
This will:
- Tokenize the Reddit + Jira text pairs.
- Train a classification model (or summarization model, depending on config).
- Log metrics, supports PEFT (LoRA) and integrates Hugging Face Trainer.

To switch between tasks (e.g., summarization vs classification), update the training configuration inside `train_pipeline.py` or pass a flag (e.g., `--task summarization`).

## Inference Pipeline

```bash
python inference_pipeline.py \
    --model_dir ./checkpoints/llama3-feedback \
    --input_file ./data/test_feedback.json \
    --output_file ./outputs/predictions.json
```

## Dataset Preparation

This project uses paired Reddit feedback and Jira issues to fine-tune a language model for automated issue tracking. To create the training dataset:

1. Provide Input Files
Place your raw JSON data in the data/ folder with the following filenames:

- data/raw_reddit_comments.json: A list of Reddit comments in the format:
```json
[
  {"text": "Multiplayer crashes constantly after the last update."},
  {"text": "Inventory doesn't open on controller."}
]
```
- data/raw_jira_issues.json: A list of Jira issues in the format:
```json
[
  {"summary": "Crash in multiplayer mode", "component": "Multiplayer", "priority": "High"},
  {"summary": "Controller input bug", "component": "UI/UX", "priority": "Medium"}
]
```

2. Generate Training Data
Run the script below to generate the aligned dataset used for training:

```bash
python dataset_pipeline.py
```

This creates `data/processed_pairs.jsonl`, where each line contains a structured `input → output` training pair for the LLM.

```json
{
  "input": "Reddit feedback: multiplayer crashes constantly after the last update.",
  "output": "Mitigation: Crash in multiplayer mode\nComponent: Multiplayer\nPriority: High"
}
```

You can now use data/processed_pairs.jsonl to fine-tune the model via HuggingFace's Trainer.

Note: The repository includes small demo files under `data/`, but users can substitute them with real Jira export data and Reddit dumps for fine-tuning on larger datasets. The original dataset is proprietary and can't be made open-source.

### How to Build the Dataset

The file `build_dataset.py` is an entry point that uses `dataset_pipeline.py` to:

- Load and clean raw Reddit and Jira data.
- Perform semantic alignment (if required).
- Generate training-ready pairs (e.g., for summarization or classification tasks).

### Run dataset processing:

```bash
python build_dataset.py \
  --reddit_path data/raw_reddit_comments.json \
  --jira_path data/raw_jira_issues.json \
  --output_path data/processed_pairs.jsonl
```

This script:
- Reads raw_reddit_comments.json and raw_jira_issues.json.
- Pairs semantically similar entries.
- Outputs the final dataset in JSONL format, with each line containing:
    - reddit_id, reddit_text
    - jira_id, jira_summary
    - label: binary relevance (1 = match)

### Output Format
```json
{
  "reddit_id": "r2",
  "reddit_text": "UI is too slow when opening the inventory.",
  "jira_id": "JIRA-102",
  "jira_summary": "Slow inventory UI",
  "label": 1
}
```


## Azure Deployment (Optional)
An example Azure Bicep script (`azure_deploy.bicep`) is included for deploying the model as a web API in a containerized environment (e.g., using Azure ML or App Services).

To deploy the model as an Azure Function App using Docker:


## Citation

If you use this work, please cite our paper:
```
Ciprian Paduraru, Miruna Zavelca, Alin Stefanescu, "Semantic Feedback Processing with LLMs: Automating Issue Detection and Prioritization in DevOps", KES 2025.
```
