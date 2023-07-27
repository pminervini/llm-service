#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import openai
import json

import requests
import numpy as np

from tqdm import tqdm


def completion(endpoint, data):
    r = requests.post(url=endpoint, json=data)
    # breakpoint()
    # return openai.Completion.create(**kwargs)
    return r.json()


def extract_ans(ans_model):
    ans_model = ans_model.split('\n')
    ans = []
    for li, al in enumerate(ans_model):
        ans.append(al)
        if 'answer is' in al:
            break
    residual = list(ans_model[li + 1:])
    ans = '\n'.join(ans)
    residual = '\n'.join(residual)
    return ans, residual


def load_from_json(path):
    with open(path, 'r') as f:
        res = json.load(f)
    return res


def main():
    train_data = load_from_json('data/strategyqa/strategyqa_train.json')

    # train_filtered = load_from_json('data/strategyqa_train_filtered.json')
    # train_data_paragraphs = load_from_json('data/strategyqa_train_paragraphs.json')
    # test_data = load_from_json('data/strategyqa_test.json')

    # dev_idx = np.load('data/dev_idx.npy')

    dev_idx = np.array([0, 1, 2])
    dev_data = [train_data[i] for i in dev_idx]

    with open('lib_prompt/strategyqa/prompt_original_step.txt') as f:
        prompt_original_step = f.read()

    model_name = 'pminervini/llama-65b'
    endpoint = "http://127.0.0.1:5000/generate"

    acc = 0
    total = 0

    with open('outputs/dev_codex_original_step.txt', 'w') as f:
        for d in tqdm(dev_data):
            q = d['question']
            a = d['answer']

            if a is True:
                a = 'yes'
            else:
                a = 'no'

            prompt_q = prompt_original_step + '\nQ: ' + q + '\n'
            prompt_q += 'A:'

            data = {
                "model": model_name,
                "prompt": prompt_q,
                "temperature": 0,
                "max_new_tokens": 256
            }

            response = completion(endpoint, data)

            ans_model = response['choices'][0]['text']
            ans_model, _ = extract_ans(ans_model)
            f.write('Q: %s\nA_model:\n%s\nA:\n%s\n\n' % (q, ans_model, a))

            ans_ = ans_model.split('answer is ')
            if len(ans_) > 1 and 'yes' in ans_[1]:
                ans_ = 'yes'
            else:
                ans_ = 'no'

            if ans_ == a:
                acc += 1
            total += 1

    print('Total %d correct %d acc %.4f' % (total, acc, acc / total))


if __name__ == '__main__':
    main()
