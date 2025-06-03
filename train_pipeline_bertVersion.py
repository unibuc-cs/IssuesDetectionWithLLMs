import argparse
import os
from datasets import load_dataset, load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
import evaluate

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="facebook/bart-base")
    parser.add_argument("--task", type=str, choices=["summarization", "classification"], default="summarization")
    parser.add_argument("--dataset_path", type=str, default="data/processed_pairs.jsonl")
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    return parser.parse_args()

def preprocess_function(example, tokenizer, task):
    if task == "summarization":
        inputs = example["reddit_comment"]
        targets = example["jira_summary"]
        model_inputs = tokenizer(inputs, max_length=512, truncation=True)
        labels = tokenizer(targets, max_length=128, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    elif task == "classification":
        inputs = example["reddit_comment"]
        labels = 1 if example["label"] == "bug" else 0
        model_inputs = tokenizer(inputs, max_length=512, truncation=True)
        model_inputs["labels"] = labels
        return model_inputs

def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    print("Loading dataset...")
    dataset = load_dataset("json", data_files={"train": args.dataset_path, "validation": args.dataset_path})

    print("Tokenizing...")
    tokenized = dataset.map(lambda x: preprocess_function(x, tokenizer, args.task), batched=True)

    print("Loading model...")
    if args.task == "summarization":
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    elif args.task == "classification":
        from transformers import AutoModelForSequenceClassification
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=2,
        weight_decay=0.01,
        predict_with_generate=(args.task == "summarization"),
        logging_dir=f"{args.output_dir}/logs",
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model) if args.task == "summarization" else None,
    )

    print("Starting training...")
    trainer.train()
    trainer.save_model(args.output_dir)

if __name__ == "__main__":
    main()
