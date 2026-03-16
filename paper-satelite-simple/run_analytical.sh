#!/bin/bash


/home/alex/miniconda3/envs/phd-models/bin/python analytical.py \
  --real_time_flows 2 \
  --real_time_lambdas 4 2 \
  --real_time_mus 1 1 \
  --real_time_resources 1 4 \
  --data_resources_min 2 \
  --data_resources_max 8 \
  --data_lambda 10 \
  --data_mu 1 \
  --beam_capacity 30 \
  --max_error 1e-8 \
  --max_iter 15000 \
  --data_requests_batch_probs 0.125 0.125 0.125 0.125 0.125 0.125 0.125 0.125
