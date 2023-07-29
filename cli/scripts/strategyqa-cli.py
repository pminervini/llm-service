#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import json

import psutil
import argparse
import requests

from tqdm import tqdm

import logging

logger = logging.getLogger(__name__)


def completion(endpoint, data):
    r = requests.post(url=f'{endpoint}/generate', json=data)
    return r.json()


def free_space(endpoint):
    r = requests.post(url=f'{endpoint}/clean', json={})
    return r.json()


def extract_ans(ans_model):
    ans_model = ans_model.split('\n')
    ans = []
    li = None
    for li, al in enumerate(ans_model):
        ans.append(al)
        if 'answer is' in al:
            break
    assert li is not None
    residual = list(ans_model[li + 1:])
    ans = '\n'.join(ans)
    residual = '\n'.join(residual)
    return ans, residual


def load_from_json(path):
    with open(path, 'r') as f:
        res = json.load(f)
    return res


def main(argv):
    parser = argparse.ArgumentParser('StrategyQA', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--prompt', action='store', type=str, default='lib_prompt/strategyqa/prompt_original_step.txt')
    parser.add_argument('--endpoint', action='store', type=str, default='http://127.0.0.1:5000')
    parser.add_argument('--model', action='store', type=str, default='pminervini/llama-7b')
    parser.add_argument('--output', action='store', type=str, default='outputs/dev_codex_original_step.txt')

    args = parser.parse_args(argv)
    prompt_path = args.prompt
    endpoint = args.endpoint
    model_name = args.model
    output_path = args.output

    train_data = load_from_json('data/strategyqa/strategyqa_train.json')

    # >>> rs = np.random.RandomState(0)
    # >>> rs.choice(2290, 256, replace=False)
    dev_idx = [1255, 2100, 711, 1328, 53, 966, 2027, 501, 963, 1665, 1732, 1929, 320, 883, 2169, 1544, 224, 314, 69, 1385, 132, 1567, 1440, 811, 789, 713, 1620, 946, 1189, 1788,
               1985, 1776, 1358, 2123, 361, 1332, 148, 1581, 2234, 712, 385, 672, 487, 657, 629, 768, 2195, 820, 817, 2275, 1216, 512, 1399, 532, 996, 582, 1585, 98, 1005, 2073,
               10, 1621, 156, 728, 1374, 1263, 2254, 535, 905, 2167, 2244, 810, 579, 918, 521, 1878, 991, 1976, 1609, 1103, 1294, 1449, 568, 1303, 438, 1099, 1633, 1642, 1452, 39,
               1639, 1510, 195, 1091, 2211, 1322, 70, 459, 651, 251, 974, 1927, 1945, 1622, 286, 1271, 1227, 892, 1785, 1026, 1728, 692, 1509, 1295, 57, 1076, 2001, 480, 220, 1984,
               1147, 124, 182, 2111, 1758, 2037, 581, 1245, 530, 384, 446, 1857, 245, 2183, 1809, 405, 162, 2213, 135, 1521, 1826, 1137, 701, 443, 1081, 1185, 1375, 2187, 2282,
               2114, 1101, 1767, 2035, 107, 596, 402, 1, 553, 1794, 76, 667, 727, 682, 1519, 1911, 1860, 457, 2069, 1603, 1832, 982, 2151, 296, 575, 1670, 2182, 399, 1722, 2149,
               2279, 14, 792, 2109, 1550, 938, 518, 422, 1357, 1199, 1330, 1471, 648, 661, 2102, 654, 638, 1379, 1446, 805, 517, 861, 117, 609, 333, 48, 289, 461, 440, 937, 1023,
               1140, 1990, 562, 2020, 1869, 264, 1765, 1003, 489, 1753, 818, 191, 1327, 1049, 1827, 746, 962, 703, 1146, 610, 1200, 1405, 565, 1193, 1087, 965, 1426, 1093, 1261,
               488, 1074, 547, 206, 1058, 1001, 563, 1273, 252, 27, 1858, 276, 1825, 2056, 878, 1482, 1725]

    dev_data = [train_data[i] for i in dev_idx]

    with open(prompt_path) as f:
        prompt_original_step = f.read()

    hdd = psutil.disk_usage('/')
    if (hdd.free / (2 ** 30)) < 400.0:
        free_space(endpoint)

    acc = 0.0
    total = 0.0

    with open(output_path, 'w') as f:
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
                "temperature": 0.0,
                "max_new_tokens": 256
            }

            response = completion(endpoint, data)

            # breakpoint()

            ans_model = response['choices'][0]['text']

            ans_model, _ = extract_ans(ans_model)

            f.write(f'Q: {q}\nA_model:\n{ans_model}\nA:\n{a}\n\n')
            f.flush()

            # breakpoint()

            ans_ = ans_model.split('answer is ')
            if len(ans_) > 1 and 'yes' in ans_[1]:
                ans_ = 'yes'
            else:
                ans_ = 'no'

            if ans_ == a:
                acc += 1.0
            total += 1.0

        f.write(f'\n\nDONE -- Total {total} correct {acc} accuracy {acc / total:.4f}\n')
        f.flush()

    print(f'\n\nDONE -- Total {total} correct {acc} accuracy {acc / total:.4f}\n')


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    main(sys.argv[1:])
