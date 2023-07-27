#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import datasets
import torch
import re
import numpy as np

from tqdm import tqdm
from datasets import load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, MaxLengthCriteria

gsm8k = load_dataset('gsm8k', 'main')
gsm8k_test = gsm8k['test']

validation_index = np.load('lib_prompt/gsm8k/validation_index.npy')
validation_data = gsm8k['train'].select(validation_index)

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small", device_map='auto').to('cpu')

q_lens = [len(d['question']) for d in gsm8k['train']]

np.percentile(q_lens, [50, 80, 90, 95, 99, 100])

np.argmax(q_lens)

input_text = gsm8k['train'][3331]['question']

gsm8k['train'][3331]['answer']

with open('lib_prompt/gsm8k/prompt_hard.txt', 'r') as f:
    prompt_complex = f.read()

prompt_q = prompt_complex + '\nQuestion: ' + input_text + '\n'
print(prompt_q)

input_ids = tokenizer(prompt_q, return_tensors="pt").input_ids.to('cpu')
# input_ids.to("cuda:0")
input_ids.size()

outputs = model.generate(input_ids, max_length=256)
print(tokenizer.decode(outputs[0]))

with torch.inference_mode():
    outputs = model.generate(input_ids,
                             do_sample=True,
                             max_new_tokens=256,
                             output_scores=True,
                             return_dict_in_generate=True,
                             num_return_sequences=10)

print(tokenizer.decode(outputs['sequences'][0]))
print(tokenizer.decode(outputs['sequences'][1]))
print(tokenizer.decode(outputs['sequences'][2]))

len(outputs['scores'])

outputs['scores'][0].size()

