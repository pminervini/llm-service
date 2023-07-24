#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

import requests

import logging

logger = logging.getLogger(__name__)


def main(argv):
    service_endpoint = "http://127.0.0.1:5000/generate"

    data = {
        # "model": "pminervini/llama-30b",
        # "peft_model": "chansung/alpaca-lora-30b",
        "model": "bigscience/bloom-560m",
        "prompt": "Hello world",
        "temperature": 0
    }

    r = requests.post(url=service_endpoint, data=data)

    # extracting response text
    # pastebin_url = r.text
    # logger.info("The pastebin URL is:%s" % pastebin_url)

    print(r)


if __name__ == '__main__':
    main(sys.argv[1:])
