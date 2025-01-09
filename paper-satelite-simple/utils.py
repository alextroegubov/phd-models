from dataclasses import dataclass

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