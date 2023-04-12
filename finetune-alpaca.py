import torch
import transformers
from transformers import BloomTokenizerFast, BloomForCausalLM, TrainingArguments
from datasets import load_dataset
from utils import ModifiedTrainer, tokenise_data, data_collator
from utils import ModelArguments, DataArguments
from datasets import Dataset, DatasetDict

import json

def local_load_dataset(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def add_text_field(data):
    for entry in data:
        entry['text'] = (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{entry['instruction']}\n\n"
            f"### Input:\n{entry['input']}\n\n"
            f"### Response:\n{entry['output']}"
        )
    return data

def main():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model_name = model_args.model_name_or_path
    tokeniser = BloomTokenizerFast.from_pretrained(
        f"{model_name}", add_prefix_space=True
    )
    model = BloomForCausalLM.from_pretrained(f"{model_name}").to(device)

    data_name = data_args.data_name_or_path
    data = add_text_field(local_load_dataset(data_name))

    dataset_train = Dataset.from_dict({k: [d[k] for d in data] for k in data[0]})
    dataset = DatasetDict({"train": dataset_train})
    input_ids = tokenise_data(dataset, tokeniser)

    model.gradient_checkpointing_enable()
    model.is_parallelizable = True
    model.model_parallel = True

    # train
    trainer = ModifiedTrainer(
        model=model,
        train_dataset=input_ids,
        args=training_args,
        data_collator=data_collator,
    )
    trainer.train()


if __name__ == "__main__":
    main()
