from dataclasses import dataclass, field


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
    # lambda_e
    data_lambda: float
    # mu_e
    data_mu: float

    # sigma
    queue_intensity: float
    # nu
    retry_intensity: float
    # H
    retry_probability: float

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
            f"    b_min = {self.data_resources_min}\n"
            f"    λ_e = {self.data_lambda:.5f}\n"
            f"    μ_e = {self.data_mu:.5f}\n"
            f"    σ = {self.queue_intensity:.5f}\n"
            f"    ν = {self.retry_intensity:.5f}\n"
            f"    H = {self.retry_probability:.5f}\n"
        )


@dataclass
class Metrics:
    # pi_k, k=1,...,n
    rt_request_rej_prob: list[float] = field(default_factory=list)
    # y_k, k=1,...n
    mean_rt_requests_in_service: list[float] = field(default_factory=list)
    # m_k, k=1,...n
    mean_resources_per_rt_flow: list[float] = field(default_factory=list)

    # y_r
    mean_retry_requests: float = 0
    # y_q
    mean_freeze_requests: float = 0
    # y_d
    mean_data_requests_in_system: float = 0
    # y_e
    mean_data_requests_in_service: float = 0

    # m_e
    mean_resources_per_data_flow: float = 0
    # b_e
    mean_resources_per_data_request: float = 0
    # Lambda_e
    intensity_all_requests: float = 0
    # Lambda_e,b
    intensity_blocked_requests: float = 0

    # pi_e,0
    primary_data_request_reject_prob: float = 0
    # pi_e,a
    data_request_attempt_reject_prob: float = 0
    # pi_e,r
    data_request_not_serviced_prob: float = 0

    # W_sess
    mean_data_request_in_system_time: float = 0
    # A
    retry_amplification_factor: float = 0

    # v
    beam_utilization: float = 0

    # caption
    text: str = ""

    @property
    def pi_k(self) -> list[float]:
        return self.rt_request_rej_prob

    @property
    def y_k(self) -> list[float]:
        return self.mean_rt_requests_in_service

    @property
    def m_k(self) -> list[float]:
        return self.mean_resources_per_rt_flow

    @property
    def y_r(self) -> float:
        return self.mean_retry_requests

    @property
    def y_q(self) -> float:
        return self.mean_freeze_requests

    @property
    def y_d(self) -> float:
        return self.mean_data_requests_in_system

    @property
    def y_e(self) -> float:
        return self.mean_data_requests_in_service

    @property
    def m_e(self) -> float:
        return self.mean_resources_per_data_flow

    @property
    def b_e(self) -> float:
        return self.mean_resources_per_data_request

    @property
    def Lambda_e(self) -> float:
        return self.intensity_all_requests

    @property
    def Lambda_e_b(self) -> float:
        return self.intensity_blocked_requests

    @property
    def pi_e_0(self) -> float:
        return self.primary_data_request_reject_prob

    @property
    def pi_e_a(self) -> float:
        return self.data_request_attempt_reject_prob

    @property
    def pi_e_r(self) -> float:
        return self.data_request_not_serviced_prob

    @property
    def W_sess(self) -> float:
        return self.mean_data_request_in_system_time

    @property
    def A(self) -> float:
        return self.retry_amplification_factor

    @property
    def util(self) -> float:
        return self.beam_utilization


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
            f"      Mean retry requests         : {self.mean_retry_requests:.4f}\n"
            f"      Mean freeze requests        : {self.mean_freeze_requests:.4f}\n"
            f"      Mean requests in system     : {self.mean_data_requests_in_system:.4f}\n"
            f"      Mean requests in service    : {self.mean_data_requests_in_service:.4f}\n"
            f"      Mean resources per flow     : {self.mean_resources_per_data_flow:.4f}\n"
            f"      Mean resources per request  : {self.mean_resources_per_data_request:.4f}\n"
            f"      Primary reject prob.        : {self.primary_data_request_reject_prob:.5f}\n"
            f"      Attempt reject prob.        : {self.data_request_attempt_reject_prob:.5f}\n"
            f"      Not serviced prob.          : {self.data_request_not_serviced_prob:.5f}\n"
            f"      Mean time in system         : {self.mean_data_request_in_system_time:.4f}\n"
            f"      Retry amplification factor   : {self.retry_amplification_factor:.4f}\n"
            f"      Intensity of all requests    : {self.intensity_all_requests:.4f}\n"
            f"      Intensity of blocked requests: {self.intensity_blocked_requests:.4f}\n"
        )
