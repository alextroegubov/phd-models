#!/bin/bash

# /home/alex/miniconda3/envs/phd-models/bin/python simulation.py \
#   --real_time_flows 1 \
#   --real_time_lambdas 0.042 \
#   --real_time_mus 0.003333 \
#   --real_time_resources 3 \
#   --data_resources_min 3 \
#   --data_resources_max 5 \
#   --data_lambda 4.2 \
#   --data_mu 0.0625 \
#   --beam_capacity 50 \
#   --warmup 50000 \
#   --events 1000000 \
#   --seed 1


# /home/alex/miniconda3/envs/phd-models/bin/python approximation.py \
#   --real_time_flows 2 \
#   --real_time_lambdas 4 10 \
#   --real_time_mus 1.5 2.5 \
#   --real_time_resources 2 4 \
#   --data_resources_min 2 \
#   --data_resources_max 4 \
#   --data_lambda 10 \
#   --data_mu 2 \
#   --beam_capacity 10 \

/home/alex/miniconda3/envs/phd-models/bin/python approximation.py \
  --real_time_flows 2 \
  --real_time_lambdas 4 10 \
  --real_time_mus 1.5 4 \
  --real_time_resources 2 4 \
  --data_resources_min 2 \
  --data_resources_max 8 \
  --data_lambda 8 \
  --data_mu 2 \
  --beam_capacity 30 \