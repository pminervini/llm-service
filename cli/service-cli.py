#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import time
import torch

from flask import Flask, make_response, request, json
from werkzeug.exceptions import HTTPException

from chainer.util import get_models, decode_kwargs, clean_cache, find_sub_list
from chainer.base import create_model, generate

from typing import Tuple, Optional, Any, List

import threading
import logging

logger = logging.getLogger(__name__)

# set up the Flask application
app = Flask(__name__)
semaphore = threading.Semaphore()

# Only keep one model in memory at any given time
cache = {
    'name': {
        'model': None,
        'peft_model': None,
        'dtype': None
    },

    'tokenizer': None,
    'model': None
}

# find out which device we are using
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"

logger.info(f"Using device: {device}")


def get_model(model_name: str, peft_model_name: Optional[str],
              do_compile: bool = True, dtype: torch.dtype = torch.bfloat16) -> Tuple[Any, Any]:
    global cache

    load_new_model = cache['model'] is None
    if model_name != cache['name']['model'] \
            or peft_model_name != cache['name']['peft_model'] \
            or dtype != cache['name']['dtype']:
        load_new_model = True

    if load_new_model:
        logger.info(f"Loading model: {model_name}")

        # Free some memory before loading a new model
        cache['model'] = None
        cache['tokenizer'] = None

        if device in {'cuda'}:
            torch.cuda.empty_cache()

        tokenizer, model = create_model(model_name, peft_model_name,
                                        device=device, do_compile=do_compile, dtype=dtype)
        cache = {
            'name': {
                'model': model_name,
                'peft_model': peft_model_name,
                'dtype': dtype
            },

            'tokenizer': tokenizer,
            'model': model
        }

    tokenizer = cache['tokenizer']
    model = cache['model']

    return tokenizer, model


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
    top_p = data.get("top_p", 1.0)
    top_k = data.get("top_k", 50)

    num_beams = data.get("num_beams", 1)
    max_new_tokens = data.get("max_new_tokens", 256)

    do_sample = data.get("do_sample", False)
    no_repeat_ngram_size = data.get("no_repeat_ngram_size", 0)
    early_stopping = data.get("early_stopping", False)
    num_return_sequences = data.get("num_return_sequences", 1)

    dtype_str = data.get("dtype", "bfloat16")
    dtype = getattr(torch, dtype_str)

    kwargs = decode_kwargs(data)

    logger.info(f"model_name: {model_name}")
    logger.info(f"peft_model_name: {peft_model_name}")

    tokenizer, model = get_model(model_name, peft_model_name, dtype=dtype)

    # generate the completion
    output_lst, tokenizer = generate(tokenizer, model, prompt,
                                     temperature=temperature,
                                     top_p=top_p,
                                     top_k=top_k,
                                     num_beams=num_beams,
                                     max_new_tokens=max_new_tokens,
                                     do_sample=do_sample,
                                     no_repeat_ngram_size=no_repeat_ngram_size,
                                     early_stopping=early_stopping,
                                     num_return_sequences=num_return_sequences,
                                     dtype=dtype,
                                     **kwargs)

    prompt_ids = tokenizer.encode(prompt)

    generated_lst = []

    for output in output_lst:
        generated_text = output['text']
        generated_ids = tokenizer.encode(generated_text)
        prompt_idx = find_sub_list(prompt_ids, generated_ids)
        assert prompt_idx is not None
        completion_text = tokenizer.decode(generated_ids[prompt_idx[1] + 1:], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        entry = {
            'completion': completion_text,
            'full': generated_text,
        }
        if 'score' in output:
            entry['score'] = output['score']
        generated_lst += [entry]

    def to_choice(entry_):
        res_ = {
            'text': entry_['completion'],
            'full_text': entry_['full'],
            'finish_reason': 'length'
        }
        if 'score' in entry_ and entry_['score'] is not None:
            res_['score'] = entry_['score']
        return res_

    res = json.jsonify({
        'object': 'text_completion',
        'id': 'dummy',

        'created': int(time.time()),
        'model': model_name,
        'choices': [to_choice(entry) for entry in generated_lst]
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
    semaphore.release()
    return response


if __name__ == "__main__":
    app.run()

