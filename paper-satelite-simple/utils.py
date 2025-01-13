from dataclasses import dataclass
from typing import Optional

@dataclass
class ParametersSet:
    real_time_flows: int
    real_time_lambdas: list[float]
    real_time_mus: list[float]
    real_time_resources: list[int]

    data_resources_min: int
    data_resources_max: int
    data_lambda: float
    data_mu: float

    beam_capacity: int

@dataclass
class Metrics:
    # pi_k, k=1,...,n and pi_e
    rt_request_rej_prob: list[float] = None
    data_request_rej_prob: float = 0
    # y_k, k=1,...n and y_e
    mean_rt_requests_in_service: list[float] = None
    mean_data_requests_in_service: float = 0
    # m_k, k=1,...n and m_e
    mean_resources_per_rt_flow: list[float] = None
    mean_resources_per_data_flow: float = 0
    # W and b_e
    mean_data_request_service_time: float = 0
    mean_resources_per_data_request: float = 0

    beam_utilization: float = 0