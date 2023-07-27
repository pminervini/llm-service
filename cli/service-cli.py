#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import time
import torch

from peft import PeftModel

from flask import Flask, make_response, request, json
from werkzeug.exceptions import HTTPException

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from transformers import GenerationConfig

from chainer.util import get_models, decode_kwargs, clean_cache

from typing import Tuple, Optional, Any

import threading
import logging

logger = logging.getLogger(__name__)

# set up the Flask application
app = Flask(__name__)
semaphore = threading.Semaphore()

cached_model_name = None
cached_peft_model_name = None

cached_tokenizer = None
cached_model = None

# find out which device we are using
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"

logger.info(f"Using device: {device}")


def get_model(model_name: str, peft_model_name: Optional[str], do_compile: bool = True) -> Tuple[str, str, Any, Any]:
    global cached_model_name, cached_peft_model_name, cached_tokenizer, cached_model

    tokenizer = cached_tokenizer
    model = cached_model

    if model_name is None or (model_name != cached_model_name or peft_model_name != cached_peft_model_name):
        logger.info(f"Loading model: {model_name}")

        model_kwargs = {}
        peft_kwargs = {}

        if device == "cuda":
            model_kwargs['torch_dtype'] = torch.bfloat16
            peft_kwargs['torch_dtype'] = torch.bfloat16
        else:
            model_kwargs['low_cpu_mem_usage'] = True

        tokenizer = AutoTokenizer.from_pretrained(model_name, resume_download=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", resume_download=True, **model_kwargs)
        # model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map="auto", resume_download=True, **model_kwargs).to(device)

        if peft_model_name is not None:
            model = PeftModel.from_pretrained(model, peft_model_name, device_map="auto", resume_download=True, **peft_kwargs)

        if do_compile is True:
            model = torch.compile(model)

        # breakpoint()

        cached_model_name = model_name
        cached_peft_model_name = peft_model_name
        cached_tokenizer = tokenizer
        cached_model = model

    return model_name, peft_model_name, tokenizer, model


def evaluate(model_name: str, peft_model_name: Optional[str], prompt: str, temperature: float = 0.1, top_p: float = 0.75,
             top_k: int = 40, num_beams: int = 1, max_new_tokens: int = 128, **kwargs):

    model_name, peft_model_name, tokenizer, model = get_model(model_name, peft_model_name)

    logger.info(f"model_name: {model_name}")
    logger.info(f"peft_model_name: {peft_model_name}")
    logger.info(f"prompt: {prompt}")
    logger.info(f"temperature: {temperature}")
    logger.info(f"top_p: {top_p}")
    logger.info(f"top_k: {top_k}")
    logger.info(f"num_beams: {num_beams}")
    logger.info(f"max_new_tokens: {max_new_tokens}")

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)

    generation_config = GenerationConfig(temperature=temperature, top_p=top_p, top_k=top_k, num_beams=num_beams, **kwargs)

    with torch.inference_mode():
        generation_output = model.generate(input_ids=input_ids, generation_config=generation_config, return_dict_in_generate=True,
                                           output_scores=True, max_new_tokens=max_new_tokens)

    s = generation_output.sequences[0]
    res = tokenizer.decode(s)
    return res, tokenizer


# define the completion endpoint
@app.route("/generate", methods=["POST"])
def generate():
    semaphore.acquire()

    # get the request data
    data = request.get_json(force=True)

    model_name = data["model"]
    peft_model_name = data.get("peft_model", None)

    # get the prompt and other parameters from the request data
    prompt = data["prompt"]

    temperature = data.get("temperature", 1.0)
    top_p = data.get("top_p", 0.75)
    top_k = data.get("top_k", 40)
    num_beams = data.get("num_beams", 1)
    max_new_tokens = data.get("max_new_tokens", 256)

    kwargs = decode_kwargs(data)

    # generate the completion
    generated_text, tokenizer = evaluate(model_name, peft_model_name, prompt, temperature=temperature, top_p=top_p, top_k=top_k,
                                         num_beams=num_beams, max_new_tokens=max_new_tokens, **kwargs)

    prompt_tokens = len(tokenizer.encode(prompt))
    completion_tokens = len(tokenizer.encode(generated_text))

    total_tokens = prompt_tokens + completion_tokens

    res = json.jsonify({
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

    semaphore.release()

    return res


@app.route('/models')
def models():
    semaphore.acquire()
    model_lst = sorted(get_models())
    res = make_response(json.jsonify({
        'data': [{
            'object': 'engine',
            'id': model_id,
            'ready': True,
            'owner': 'huggingface',
            'permissions': None,
            'created': None
        } for model_id in model_lst]
    }))
    semaphore.release()
    return res


@app.route('/clean')
def clean():
    semaphore.acquire()
    clean_cache()
    res = make_response(json.jsonify({'done': True}))
    semaphore.release()
    return res


@app.errorhandler(HTTPException)
def handle_exception(e):
    """Return JSON instead of HTML for HTTP errors."""
    # start with the correct headers and status code from the error
    response = e.get_response()
    # replace the body with JSON
    response.data = json.dumps({
        "code": e.code,
        "name": e.name,
        "description": e.description,
    })
    response.content_type = "application/json"
    return response


if __name__ == "__main__":
    app.run()

