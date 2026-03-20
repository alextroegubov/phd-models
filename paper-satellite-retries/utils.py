from dataclasses import dataclass


@dataclass
class ParametersSet:
    # n
    real_time_flows: int
    # lambda_1, lambda_2, ..., lambda_n
    real_time_lambdas: list[float]
    # mu_1, mu_2, ..., mu_n
    real_time_mus: list[float]
    # b_1, b_2, ..., b_n
    real_time_resources: list[int]

    # b_min
    data_resources_min: int
    # b_max
    data_resources_max: int
    # lambda_e
    data_lambda: float
    # mu_e
    data_mu: float

    # sigma
    queue_intensity: float
    # nu
    retry_intensity: float
    # H
    leave_probability: float

    # v
    beam_capacity: int
    random_seed: int = 0

    def __str__(self) -> str:
        def f_lst_4f(lst):
            return ", ".join(f"{x:.4f}" for x in lst) if lst else "[]"

        rt_gen = zip(self.real_time_lambdas, self.real_time_mus, self.real_time_resources)
        real_time_params = "\n".join(
            f"  Flow {i}:\n\t\tλ = {lam:.5f},\n\t\tμ = {mu:.5f},\n\t\tb = {res}"
            for i, (lam, mu, res) in enumerate(rt_gen)
        )

        return (
            f"\nParametersSet:\n"
            f"  Beam Capacity: {self.beam_capacity}\n"
            f"  Real-Time Flows: {self.real_time_flows}\n"
            f"  {real_time_params}\n"
            f"  Elastic Data Flow\n"
            f"    λ = {self.data_lambda:.5f}\n"
            f"    μ = {self.data_mu:.5f}\n"
            f"    sigma = {self.queue_intensity:.5f}\n"
            f"    nu = {self.retry_intensity:.5f}\n"
            f"    H = {self.leave_probability:.5f}\n"
        )


@dataclass
class Metrics:
    # pi_k, k=1,...,n
    rt_request_rej_prob: list[float]
    # y_k, k=1,...n
    mean_rt_requests_in_service: list[float]
    # m_k, k=1,...n
    mean_resources_per_rt_flow: list[float]

    # y_r
    mean_retry_requests: float = 0
    # y_q
    mean_freeze_requests: float = 0
    # Lambda_b
    intensity_blocked_requests: float = 0

    # m_e
    mean_resources_per_data_flow: float = 0
    # Lambda
    intensity_all_requests: float = 0

    # # pi_e
    # data_request_rej_prob: float = 0
    # y_e
    mean_data_requests_in_service: float = 0
    # # W
    # mean_data_request_service_time: float = 0
    # b_e
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
            # f"      Request rejection prob.     : {self.data_request_rej_prob:.4f}\n"
            f"      Mean requests in Service    : {self.mean_data_requests_in_service:.4f}\n"
            # f"      Mean resources per flow     : {self.mean_resources_per_data_flow:.4f}\n"
            f"      Mean resources per request  : {self.mean_resources_per_data_request:.4f}\n"
            # f"      Mean request service time   : {self.mean_data_request_service_time:.4f}\n"
        )
