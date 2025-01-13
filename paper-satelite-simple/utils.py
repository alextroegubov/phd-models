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

    def __str__(self) -> str:

        rt_gen = zip(self.real_time_lambdas, self.real_time_mus, self.real_time_resources)
        real_time_params = "\n".join(
            f"  Flow {i}:\n\tλ = {lam:.5f},\n\tμ = {mu:.5f},\n\tb = {res}"
            for i, (lam, mu, res) in enumerate(rt_gen)
        )

        return (
            f"\nParametersSet:\n"
            f"  Beam Capacity: {self.beam_capacity}\n"
            f"  Real-Time Flows: {self.real_time_flows}\n"
            f"  {real_time_params}\n"
            f"  Elastic Data Flow\n"
            f"    b_min = {self.data_resources_min}\n"
            f"    b_max = {self.data_resources_max}\n"
            f"    λ = {self.data_lambda:.5f}, μ = {self.data_mu:.5f}"
        )


@dataclass
class Metrics:
    # pi_k, k=1,...,n and pi_e
    rt_request_rej_prob: Optional[list[float]] = None
    data_request_rej_prob: float = 0
    # y_k, k=1,...n and y_e
    mean_rt_requests_in_service: Optional[list[float]] = None
    mean_data_requests_in_service: float = 0
    # m_k, k=1,...n and m_e
    mean_resources_per_rt_flow: Optional[list[float]] = None
    mean_resources_per_data_flow: float = 0
    # W and b_e
    mean_data_request_service_time: float = 0
    mean_resources_per_data_request: float = 0

    beam_utilization: float = 0

    text: str = ""

    def __str__(self):

        def f_lst_5f(lst):
            return ", ".join(f"{x:.5f}" for x in lst) if lst else "[]"

        def f_lst_3f(lst):
            return ", ".join(f"{x:.3f}" for x in lst) if lst else "[]"

        return (
            f"\nMetrics: {self.text}\n"
            f"  Overall:\n"
            f"      Beam Utilization            : {self.beam_utilization:.4f}\n"
            f"  Real-time flows:\n"
            f"      Request rejection prob.     : [{f_lst_5f(self.rt_request_rej_prob)}]\n"
            f"      Mean requests in service    : [{f_lst_3f(self.mean_rt_requests_in_service)}]\n"
            f"      Mean resources per flow     : [{f_lst_3f(self.mean_resources_per_rt_flow)}]\n"
            f"  Elastic data flow:\n"
            f"      Request rejection prob.     : {self.data_request_rej_prob:.4f}\n"
            f"      Mean requests in Service    : {self.mean_data_requests_in_service:.4f}\n"
            f"      Mean resources per flow     : {self.mean_resources_per_data_flow:.4f}\n"
            f"      Mean resources per request  : {self.mean_resources_per_data_request:.4f}\n"
            f"      Mean request service time   : {self.mean_data_request_service_time:.4f}\n"
        )
