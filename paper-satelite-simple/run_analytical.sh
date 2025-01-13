#!/bin/bash

/home/alex/miniconda3/envs/phd-models/bin/python analytical.py \
  --real_time_flows 1 \
  --real_time_lambdas 0.042 \
  --real_time_mus 0.003333 \
  --real_time_resources 3 \
  --data_resources_min 1 \
  --data_resources_max 5 \
  --data_lambda 4.2 \
  --data_mu 0.0625 \
  --beam_capacity 50 \
  --max_error 1e-7 \
  --max_iter 35000