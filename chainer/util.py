# -*- coding: utf-8 -*-

import json

from huggingface_hub import scan_cache_dir

from typing import List, Optional, Tuple

import logging

logger = logging.getLogger(__name__)


def get_models() -> List[str]:
    # collect the models available in the cache
    report = scan_cache_dir()
    res = []
    for repo in report.repos:
        logger.info("repo_id:", json.dumps(repo.repo_id, indent=4))
        logger.info("repo_type:", json.dumps(repo.repo_type, indent=4))
        logger.info("repo_path:", json.dumps(str(repo.repo_path), indent=4))
        logger.info("size_on_disk:", json.dumps(repo.size_on_disk, indent=4))
        logger.info("nb_files:", json.dumps(repo.nb_files, indent=4))
        # print(json.dumps(repo.str(refs),indent=4))
        res += [repo.repo_id]
    return res


def clean_cache():
    cache_info = scan_cache_dir()
    hash_lst = [revision.commit_hash for repo in cache_info.repos for revision in repo.revisions]
    delete_strategy = cache_info.delete_revisions(*hash_lst)
    logger.info(f"Freeing {delete_strategy.expected_freed_size_str}")
    delete_strategy.execute()
    return


def decode_kwargs(data):
    # map the data to the kwargs (openai to huggingface)

    kwargs = {}
    if 'n' in data:
        kwargs['num_return_sequences'] = data['n']
    if 'stop' in data:
        kwargs['early_stopping'] = True
        kwargs['stop_token'] = data['stop']
    if 'suffix' in data:
        kwargs['suffix'] = data['suffix']
    if 'presence_penalty' in data:
        kwargs['presence_penalty'] = data['presence_penalty']
    if 'frequency_penalty' in data:
        kwargs['repetition_penalty'] = data['frequency_penalty']
    if 'repetition_penalty ' in data:
        kwargs['repetition_penalty'] = data['repetition_penalty ']
    if 'best_of ' in data:
        kwargs['num_return_sequences'] = data['best_of ']

    return kwargs


def find_sub_list(sl: List[int], l: List[int]) -> Optional[Tuple[int, int]]:
    sll = len(sl)
    for ind in (i for i, e in enumerate(l) if e == sl[0]):
        if l[ind:ind + sll] == sl:
            return ind, ind + sll - 1
    return None
