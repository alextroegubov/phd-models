#!/bin/bash

/home/alex/miniconda3/envs/phd-models/bin/python simulation.py \
  --real_time_flows 0 \
  --real_time_lambdas 1 4 \
  --real_time_mus 1 1 \
  --real_time_resources 4 2 \
  --data_resources_min 2 \
  --data_resources_max 8 \
  --data_lambda 10 \
  --data_mu 1 \
  --beam_capacity 30 \
  --warmup 400000 \
  --events 1500000  \
  --seed 2 \
  --elastic_data_flow \
  --data_requests_batch_probs 0.3 0.7

