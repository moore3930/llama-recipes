# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://huggingface.co/datasets/samsum

import copy
import os
import datasets
import random

from llama_recipes.configs import (
    fsdp_config as FSDP_CONFIG,
    quantization_config as QUANTIZATION_CONFIG,
    train_config as TRAIN_CONFIG,
)

from transformers import (
    AutoConfig,
    AutoProcessor,
    AutoTokenizer,
)

from unittest.mock import patch

lang_map = {"en": "eng_Latn", "de": "deu_Latn", "ar": "ace_Arab", "zh": "zho_Hans"}
random.seed(42)

@patch('builtins.input', return_value="N")
def load_flores(split, lang_pairs, _):
    if split == "train":
        cur_split = "dev"
    elif split == "test":
        cur_split = "devtest"

    dir_name = "/Users/moore/workplace/projects/llama-recipes/customer_data/flores200_dataset"
    output_dataset = []
    for lp in lang_pairs:
        src, tgt = lp.split("-")
        assert src in lang_map.keys(), f"Unknown dataset: {src}"
        assert tgt in lang_map.keys(), f"Unknown dataset: {tgt}"

        if cur_split == "dev":
            src_name = "{}.dev".format(lang_map[src])
            tgt_name = "{}.dev".format(lang_map[tgt])
        elif cur_split == "devtest":
            src_name = "{}.devtest".format(lang_map[src])
            tgt_name = "{}.devtest".format(lang_map[tgt])

        with open(os.path.join(dir_name, cur_split, src_name)) as src_fin, \
                open(os.path.join(dir_name, cur_split, tgt_name)) as tgt_fin:
            for src_sent, tgt_sent in zip(src_fin, tgt_fin):
                src_sent = src_sent.strip()
                tgt_sent = tgt_sent.strip()
                idx_32bit = str(random.randint(-2 ** 31, 2 ** 31 - 1))
                row = {"id": idx_32bit, "src_lang": src, "tgt_lang": tgt, "src": src_sent, "tgt": tgt_sent}
                output_dataset.append(row)

    dataset = datasets.Dataset.from_list(output_dataset)
    return dataset


def get_preprocessed_flores(dataset_config, tokenizer, split, lang_pairs):
    dataset = load_flores(split, lang_pairs)

    lang_name = {"en": "English", "zh": "Chinese", "ar": "Arabic", "de": "German"}

    prompt = (
        f"Translate this from {{src_lang}} to {{tgt_lang}}:\n{{src_lang}}: {{src}}\n{{tgt_lang}}: "
    )

    def apply_prompt_template(sample):
        return {
            "prompt": prompt.format(src_lang=lang_name[sample["src_lang"]],
                                    tgt_lang=lang_name[sample["tgt_lang"]],
                                    src=sample["src"]),
            "summary": sample["tgt"],
        }

    dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))

    def tokenize_add_label(sample):
        prompt = tokenizer.encode(tokenizer.bos_token + sample["prompt"], add_special_tokens=False)
        summary = tokenizer.encode(sample["summary"] + tokenizer.eos_token, add_special_tokens=False)

        sample = {
            "input_ids": prompt + summary,
            "attention_mask" : [1] * (len(prompt) + len(summary)),
            "labels": [-100] * len(prompt) + summary,
            }

        return sample

    dataset = dataset.map(tokenize_add_label, remove_columns=list(dataset.features))

    return dataset

# Unit Test
lang_pairs = ["en-de", "en-zh", "en-ar"]
print(load_flores("train", lang_pairs)[0])

train_config, fsdp_config = TRAIN_CONFIG(), FSDP_CONFIG()

# Load the tokenizer and add special tokens
tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-2-7b-hf"
    if train_config.tokenizer_name is None
    else train_config.tokenizer_name
)
if not tokenizer.pad_token_id:
    tokenizer.pad_token_id = tokenizer.eos_token_id

dataset = get_preprocessed_flores(None, tokenizer, "train", lang_pairs)
print(dataset[0])
output_text = tokenizer.decode(dataset[0]['input_ids'])
print(output_text)