# -*- coding: utf-8 -*-

import torch

from peft import PeftModel

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import GenerationConfig

from typing import Tuple, Optional, Any, List

import logging

logger = logging.getLogger(__name__)


def create_model(model_name: str, peft_model_name: Optional[str], device: str,
                 do_compile: bool = True, dtype: torch.dtype = torch.bfloat16) -> Tuple[Any, Any]:

    logger.info(f"Loading model: {model_name}")

    model_kwargs = {}
    peft_kwargs = {}

    if device == "cuda":
        model_kwargs['torch_dtype'] = peft_kwargs['torch_dtype'] = dtype
        model_kwargs['device_map'] = peft_kwargs['device_map'] = 'balanced_low_0'  # 'auto'
    else:
        model_kwargs['low_cpu_mem_usage'] = True

    tokenizer = AutoTokenizer.from_pretrained(model_name, resume_download=True, padding_side="left")  # add_prefix_space=True,
    model = AutoModelForCausalLM.from_pretrained(model_name, resume_download=True, trust_remote_code=True, **model_kwargs)

    if peft_model_name is not None:
        model = PeftModel.from_pretrained(model, peft_model_name, resume_download=True, **peft_kwargs)

    if do_compile is True:
        model = torch.compile(model)

    model.eval()

    return tokenizer, model


def evaluate(tokenizer, model, prompt: str, temperature: float = 0.1, top_p: float = 0.75,
             top_k: int = 40, num_beams: int = 1, max_new_tokens: int = 128, **kwargs) -> Tuple[List[Any], Any]:

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
