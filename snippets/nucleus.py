#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch.utils.data import DataLoader
from torch import Tensor

from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM, AutoModel

from datasets import Dataset

from typing import List


model_name = "bigscience/bloom-560m"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

input_ids = tokenizer('John is the father of Mark, and Mark is the father of Mike; hence, John is Mike\'s', return_tensors='pt').input_ids

# generate text until the output length (which includes the context length) reaches 50
greedy_output = model.generate(input_ids, max_length=50)

print("Output:\n" + 100 * '-')
print(tokenizer.decode(greedy_output[0], skip_special_tokens=True))

# deactivate top_k sampling and sample only from 92% most likely words
sample_output = model.generate(input_ids, do_sample=True, max_length=50, top_p=0.92, top_k=0)

print("Output:\n" + 100 * '-')
print(tokenizer.decode(sample_output[0], skip_special_tokens=True))
