#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import time
import torch

# from peft import PeftModel

from flask import Flask, make_response, request
from flask.json import jsonify

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from transformers import GenerationConfig

from chainer.util import get_models, decode_kwargs, clean_cache

from typing import Tuple, Any

import logging

logger = logging.getLogger(__name__)


# LORA_WEIGHTS = "tloen/alpaca-lora-7b"

# set up the Flask application
app = Flask(__name__)

cached_model_name = None
cached_tokenizer = None
cached_model = None

# find out which device we are using
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"

logger.info(f"Using device: {device}")


def get_model(model_name: str) -> Tuple[str, Any, Any]:
    global cached_model_name, cached_tokenizer, cached_model

    tokenizer = cached_tokenizer
    model = cached_model

    if model_name is None or model_name != cached_model_name:
        logger.info(f"Loading model: {model_name}")

        model_kwargs = {}
        # peft_kwargs = {}

        if device == "cuda":
            model_kwargs['torch_dtype'] = torch.bfloat16
            # peft_kwargs['torch_dtype'] = torch.bfloat16
        else:
            model_kwargs['low_cpu_mem_usage'] = True

        tokenizer = AutoTokenizer.from_pretrained(model_name, resume_download=True)

        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", resume_download=True, **model_kwargs).to(device)
        # model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map="auto", resume_download=True, **model_kwargs).to(device)

        # model = PeftModel.from_pretrained(model, LORA_WEIGHTS, device_map="auto", resume_download=True, **peft_kwargs)
        # model = torch.compile(model)

        # breakpoint()

        cached_model_name = model_name
        cached_tokenizer = tokenizer
        cached_model = model

    return model_name, tokenizer, model


def evaluate(model_name: str, prompt: str, temperature: float = 0.1, top_p: float = 0.75, top_k: int = 40, num_beams: int = 1,
             max_new_tokens: int = 128, **kwargs):

    model_name, tokenizer, model = get_model(model_name)

    logger.info(f"model_name: {model_name}")
    logger.info(f"prompt: {prompt}")
    logger.info(f"temperature: {temperature}")
    logger.info(f"top_p: {top_p}")
    logger.info(f"top_k: {top_k}")
    logger.info(f"num_beams: {num_beams}")
    logger.info(f"max_new_tokens: {max_new_tokens}")

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)

    generation_config = GenerationConfig(temperature=temperature, top_p=top_p, top_k=top_k, num_beams=num_beams, **kwargs)
    with torch.inference_mode():
        generation_output = model.generate(input_ids=input_ids, generation_config=generation_config, return_dict_in_generate=True,
                                           output_scores=True, max_new_tokens=max_new_tokens)

    s = generation_output.sequences[0]
    res = tokenizer.decode(s)
    return res


# define the completion endpoint
@app.route("/generate", methods=["POST"])
def generate():
    # get the request data
    data = request.get_json(force=True)

    model_name = data["model"]

    # update model
    model_name, tokenizer, model = get_model(model_name)

    # get the prompt and other parameters from the request data
    prompt = data["prompt"]

    temperature = data.get("temperature", 1.0)
    top_p = data.get("top_p", 0.75)
    top_k = data.get("top_k", 40)
    num_beams = data.get("num_beams", 1)
    max_new_tokens = data.get("max_new_tokens", 256)

    kwargs = decode_kwargs(data)

    # generate the completion
    generated_text = evaluate(model_name, prompt, temperature=temperature, top_p=top_p, top_k=top_k,
                              num_beams=num_beams, max_new_tokens=max_new_tokens, **kwargs)

    prompt_tokens = len(tokenizer.encode(prompt))
    completion_tokens = len(tokenizer.encode(generated_text))
    total_tokens = prompt_tokens + completion_tokens

    res = jsonify({
        'object': 'text_completion',
        'id': 'dummy',

        'created': int(time.time()),
        'model': model_name,
        'choices':
            [{'text': generated_text, 'finish_reason': 'length'}],

        'usage': {
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'total_tokens': total_tokens
        }
    })

    return res


@app.route('/models')
def models():
    model_lst = sorted(get_models())
    return make_response(jsonify({
        'data': [{
            'object': 'engine',
            'id': id,
            'ready': True,
            'owner': 'huggingface',
            'permissions': None,
            'created': None
        } for id in model_lst]
    }))


@app.route('/clean')
def clean():
    clean_cache()
    return make_response(jsonify({'done': True}))


if __name__ == "__main__":
    app.run()

