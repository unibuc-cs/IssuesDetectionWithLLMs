# dataset_builder.py
import json
import os
from pathlib import Path

RAW_REDDIT_FILE = "data/raw_reddit_comments.json"
RAW_JIRA_FILE = "data/raw_jira_issues.json"
PROCESSED_FILE = "data/processed_pairs.jsonl"

def normalize_text(text):
    return text.strip().replace("\n", " ").lower()

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def build_dataset(reddit_data, jira_data):
    pairs = []
    for reddit in reddit_data:
        reddit_text = normalize_text(reddit["text"])
        best_match = max(jira_data, key=lambda x: x["component"] in reddit_text)
        pairs.append({
            "input": f"Reddit feedback: {reddit_text}",
            "output": f"Mitigation: {best_match['summary']}\nComponent: {best_match['component']}\nPriority: {best_match['priority']}"
        })
    return pairs

def save_jsonl(data, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

def main():
    os.makedirs("data", exist_ok=True)
    reddit_data = load_data(RAW_REDDIT_FILE)
    jira_data = load_data(RAW_JIRA_FILE)
    dataset = build_dataset(reddit_data, jira_data)
    save_jsonl(dataset, PROCESSED_FILE)
    print(f"Processed {len(dataset)} examples saved to {PROCESSED_FILE}")

if __name__ == "__main__":
    main()
