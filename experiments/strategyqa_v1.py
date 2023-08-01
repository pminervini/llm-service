#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools
import os
import os.path

import sys
import logging


def cartesian_product(dicts):
    return (dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))


def summary(configuration):
    kvs = sorted([(k, v) for k, v in configuration.items()], key=lambda e: e[0])
    return '_'.join([('%s=%s' % (k, v)) for (k, v) in kvs])


def to_cmd(c, _path=None):

    model_name = c["m"]
    if c["m"] in {'llama-7b', 'llama-13b', 'llama-30b', 'llama-65b'}:
        model_name = f'pminervini/{c["m"]}'
    elif c["m"] in {'Llama-2-7b-hf', 'Llama-2-13b-hf', 'Llama-2-70b-hf', 'Llama-2-7b-chat-hf', 'Llama-2-13b-chat-hf', 'Llama-2-70b-chat-hf'}:
        model_name = f'meta-llama/{c["m"]}'

    command = f'PYTHONPATH=. python3 ./cli/scripts/strategyqa-cli.py ' \
              f'--prompt lib_prompt/strategyqa/prompt_{c["p"]}.txt ' \
              f'--model {model_name} --output outputs/dev_{c["m"]}_{c["p"]}.log'

    return command


def to_logfile(c, path):
    summary_str = summary(c).replace("/", "_")
    outfile = f"{path}/strategyqa_v1.{summary_str}.log"
    return outfile


def main(argv):
    hyp_space = dict(
        m=['llama-7b', 'llama-13b', 'llama-30b', 'llama-65b',
           'Llama-2-7b-hf', 'Llama-2-13b-hf', 'Llama-2-70b-hf',
           'Llama-2-7b-chat-hf', 'Llama-2-13b-chat-hf', 'Llama-2-70b-chat-hf'],
        p=['complex', 'complex_conjunction', 'complex_step', 'original', 'original_step', 'simple', 'simple_step']
    )

    configurations = list(cartesian_product(hyp_space))

    path = 'logs/strategyqa/strategyqa_v1'

    is_beaker = False
    is_slurm = False

    # Check that we are on the UCLCS cluster first
    if os.path.exists('/home/pminervi/'):
        is_beaker = True

    import socket
    if 'inf.ed.ac.uk' in socket.gethostname():
        is_beaker = False
        is_slurm = True

    assert not (is_beaker and is_slurm)

    # If the folder that will contain logs does not exist, create it
    if not os.path.exists(path):
        os.makedirs(path)

    command_lines = set()
    for cfg in configurations:
        logfile = to_logfile(cfg, path)

        completed = False
        if os.path.isfile(logfile):
            with open(logfile, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                completed = ('DONE' in content)

        if not completed:
            cmd = to_cmd(cfg)
            if cmd is not None:
                command_line = f'{cmd} > {logfile} 2>&1'
                command_lines |= {command_line}

    # Sort command lines and remove duplicates
    sorted_command_lines = sorted(command_lines)

    import random
    rng = random.Random(0)
    rng.shuffle(sorted_command_lines)

    nb_jobs = len(sorted_command_lines)
    header = None

    if is_beaker is True:
        header = f"""#!/bin/bash -l

#$ -cwd
#$ -S /bin/bash
#$ -o $HOME/array.out
#$ -e $HOME/array.err
#$ -t 1-{nb_jobs}
#$ -l tmem=12G
#$ -l h_rt=12:00:00
#$ -l gpu=true

conda activate gpu

export LANG="en_US.utf8"
export LANGUAGE="en_US:en"

cd $HOME/workspace/kbc-lm

"""

    if is_slurm is True:
        header = f"""#!/usr/bin/env bash

#SBATCH --output=/home/%u/slogs/kbclm-%A_%a.out
#SBATCH --error=/home/%u/slogs/kbclm-%A_%a.err
#SBATCH --partition=PGR-Standard
#SBATCH --gres=gpu:1
#SBATCH --mem=14GB # memory
#SBATCH --cpus-per-task=4 # number of cpus to use - there are 32 on each node.
#SBATCH -t 6:00:00 # time requested in hours:minutes:seconds
#SBATCH --array 1-{nb_jobs}

echo "Setting up bash environment"
source ~/.bashrc
set -e # fail fast

conda activate gpu

export LANG="en_US.utf8"
export LANGUAGE="en_US:en"

cd $HOME/workspace/kbc-lm

"""

    if header is not None:
        print(header)

    for job_id, command_line in enumerate(sorted_command_lines, 1):
        if is_beaker:
            print(f'test $SGE_TASK_ID -eq {job_id} && sleep 10 && {command_line}')
        elif is_slurm:
            print(f'test $SLURM_ARRAY_TASK_ID -eq {job_id} && sleep 10 && {command_line}')
        else:
            print(f'{command_line}')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
