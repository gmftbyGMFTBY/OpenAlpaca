#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python inference.py --model_path ../ckpt/openalpaca --max_length 512 --generate_len 512 --top_k 50 --top_p 0.9
