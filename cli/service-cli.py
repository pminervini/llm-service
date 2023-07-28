#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import time
import torch

from peft import PeftModel

from flask import Flask, make_response, request, json
from werkzeug.exceptions import HTTPException

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import GenerationConfig

from chainer.util import get_models, decode_kwargs, clean_cache, find_sub_list

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

        model_kwargs = {}
        peft_kwargs = {}

        if device == "cuda":
            model_kwargs['torch_dtype'] = peft_kwargs['torch_dtype'] = dtype
            model_kwargs['device_map'] = peft_kwargs['device_map'] = "auto"
        else:
            model_kwargs['low_cpu_mem_usage'] = True

        # Free some memory before loading a new model
        cache['model'] = None
        cache['tokenizer'] = None

        tokenizer = AutoTokenizer.from_pretrained(model_name, resume_download=True, add_prefix_space=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, resume_download=True, **model_kwargs)

        if peft_model_name is not None:
            model = PeftModel.from_pretrained(model, peft_model_name, resume_download=True, **peft_kwargs)

        if do_compile is True:
            model = torch.compile(model)

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


def evaluate(model_name: str, peft_model_name: Optional[str], prompt: str, temperature: float = 0.1, top_p: float = 0.75,
             top_k: int = 40, num_beams: int = 1, max_new_tokens: int = 128, dtype: torch.dtype = torch.bfloat16, **kwargs) -> Tuple[List[Any], Any]:

    tokenizer, model = get_model(model_name, peft_model_name, dtype=dtype)

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

    res_lst = []
    for i in range(generation_output.sequences.shape[0]):
        seq = generation_output.sequences[i]
        res = tokenizer.decode(seq, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        entry = {'text': res}
        if hasattr(generation_output, "sequences_scores"):
            entry['score'] = generation_output.sequences_scores[i].item()
        res_lst += [entry]

    return res_lst, tokenizer


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

    dtype_str = data.get("dtype", "bfloat16")
    dtype = getattr(torch, dtype_str)

    kwargs = decode_kwargs(data)

    # generate the completion
    output_lst, tokenizer = evaluate(model_name, peft_model_name, prompt, temperature=temperature, top_p=top_p, top_k=top_k,
                                     num_beams=num_beams, max_new_tokens=max_new_tokens, dtype=dtype, **kwargs)

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

