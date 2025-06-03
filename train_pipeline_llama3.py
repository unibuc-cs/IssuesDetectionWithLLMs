import os
import argparse
import torch
from datasets import load_dataset, load_metric
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    DataCollatorWithPadding,
)

def load_and_prepare_data(task, dataset_path):
    raw_dataset = load_dataset('json', data_files=dataset_path, split='train')

    if task == "summarization":
        def preprocess(example):
            return {
                "input_ids": tokenizer(
                    example["input"], truncation=True, padding="max_length", max_length=512
                )["input_ids"],
                "labels": tokenizer(
                    example["target"], truncation=True, padding="max_length", max_length=128
                )["input_ids"],
            }
    elif task == "classification":
        label2id = {label: i for i, label in enumerate(set(raw_dataset["label"]))}
        def preprocess(example):
            return {
                "input_ids": tokenizer(
                    example["input"], truncation=True, padding="max_length", max_length=512
                )["input_ids"],
                "label": label2id[example["label"]],
            }
    else:
        raise ValueError("Unsupported task.")

    dataset = raw_dataset.map(preprocess, remove_columns=raw_dataset.column_names)
    return dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["summarization", "classification"], required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--output_dir", type=str, default="./output")
    args = parser.parse_args()

    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    if args.task == "summarization":
        model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16)
        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    dataset = load_and_prepare_data(args.task, args.dataset_path)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        evaluation_strategy="no",
        num_train_epochs=3,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        fp16=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

if __name__ == "__main__":
    main()
