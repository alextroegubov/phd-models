from dataclasses import dataclass, field
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
import numpy as np


class ParametersSet(BaseModel):

    model_config = ConfigDict(
        populate_by_name=True,
    )
    # n
    real_time_flows: int = Field(ge=0, description="Number of real-time flows")
    # lambda_1, lambda_2, ..., lambda_n
    real_time_lambdas: list[float] = Field(description="Real-time flow arrival rates")
    # mu_1, mu_2, ..., mu_n
    real_time_mus: list[float] = Field(description="Real-time flow service rates")
    # b_1, b_2, ..., b_n
    real_time_resources: list[int] = Field(description="Real-time flow resource units")

    # b_min
    data_resources_min: int = Field(ge=1, description="Minimum data resource units")
    # b_max
    data_resources_max: int = Field(ge=1, description="Maximum data resource units")
    # lambda_e
    data_lambda: float = Field(ge=0, description="Data batch arrival rate")
    # mu_e
    data_mu: float = Field(ge=0, description="Data service rate")

    # sigma
    queue_intensity: float = Field(ge=0, description="Queue intensity")
    # nu
    retry_intensity: float = Field(ge=0, description="Retry intensity")
    # H_e
    retry_primary_prob: float = Field(ge=0, le=1, description="Retry probability for rejected primary requests")
    # H_r
    retry_retry_prob: float = Field(ge=0, le=1, description="Retry probability for rejected retry requests")
    # H_q
    retry_freeze_prob: float = Field(ge=0, le=1, description="Retry probability for freeze requests")

    # v
    beam_capacity: int = Field(ge=1, description="Beam capacity")

    # f_s, s = 1, ..., B
    data_batch_probs: list[float] = Field(description="Data batch probabilities")

    random_seed: int = Field(default=0, description="Random seed")

    @field_validator("data_batch_probs", mode="before")
    def validate_data_batch_probs(cls, v):
        if len(v) == 0:
            raise ValueError("Empty data_batch_probs")
        if not np.isclose(np.sum(v), 1.0):
            raise ValueError("Batch probabilities must sum to 1.0")
        if np.any(np.array(v) < 0):
            raise ValueError("Some batch probabilities are negative")
        return v

    @model_validator(mode="after")
    def validate_rt_params(self):
        if len(self.real_time_lambdas) != self.real_time_flows:
            raise ValueError("Number of RT lambdas != number of RT flows")
        if np.any(np.array(self.real_time_lambdas) < 0):
            raise ValueError("Some RT lambdas are negative")
        if len(self.real_time_mus) != self.real_time_flows:
            raise ValueError("Number of RT mus != number of RT flows")
        if np.any(np.array(self.real_time_mus) < 0):
            raise ValueError("Some RT mus are negative")
        if len(self.real_time_resources) != self.real_time_flows:
            raise ValueError("Number of RT resources != number of RT flows")
        if np.any(np.array(self.real_time_resources) < 0):
            raise ValueError("Some RT resources are negative")
        return self

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
            f"    b_max = {self.data_resources_max}\n"
            f"    λ_e = {self.data_lambda:.5f}\n"
            f"    μ_e = {self.data_mu:.5f}\n"
            f"    σ = {self.queue_intensity:.5f}\n"
            f"    ν = {self.retry_intensity:.5f}\n"
            f"    H_e = {self.retry_primary_prob:.5f}\n"
            f"    H_r = {self.retry_retry_prob:.5f}\n"
            f"    H_q = {self.retry_freeze_prob:.5f}\n"
            f"    f_s = [{f_lst_4f(self.data_batch_probs)}]\n"
        )


class Metrics(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
    )
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

    # Lambda_e_p
    primary_intensity: float = 0
    # Lambda_e_p^(b)
    primary_blocked_intensity: float = 0
    # Lambda_e_r
    retry_intensity: float = 0
    # Lambda_e_r^(b)
    retry_blocked_intensity: float = 0
    # Lambda_e^(b)
    total_blocked_data_intensity: float = 0

    # pi_e,0
    primary_request_reject_prob: float = 0
    # pi_e,a
    attempt_request_reject_prob: float = 0
    # pi_e,r
    not_serviced_request_prob: float = 0

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
    def Lambda_e_p(self) -> float:
        return self.primary_intensity

    @property
    def Lambda_e_p_b(self) -> float:
        return self.primary_blocked_intensity

    @property
    def Lambda_e_r(self) -> float:
        return self.retry_intensity

    @property
    def Lambda_e_r_b(self) -> float:
        return self.retry_blocked_intensity

    @property
    def Lambda_e(self) -> float:
        return self.Lambda_e_p + self.Lambda_e_r

    @property
    def Lambda_e_b(self) -> float:
        return self.Lambda_e_p_b + self.Lambda_e_r_b

    @property
    def pi_e_0(self) -> float:
        return self.primary_request_reject_prob

    @property
    def pi_e_a(self) -> float:
        return self.attempt_request_reject_prob

    @property
    def pi_e_r(self) -> float:
        return self.not_serviced_request_prob

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
            f"      Mean retry requests         : {self.y_r:.4f}\n"
            f"      Mean freeze requests        : {self.y_q:.4f}\n"
            f"      Mean requests in system     : {self.y_d:.4f}\n"
            f"      Mean requests in service    : {self.y_e:.4f}\n"
            f"      Mean resources per flow     : {self.m_e:.4f}\n"
            f"      Mean resources per request  : {self.b_e:.4f}\n"
            f"      Primary reject prob.        : {self.pi_e_0:.5f}\n"
            f"      Attempt reject prob.        : {self.pi_e_a:.5f}\n"
            f"      Not serviced prob.          : {self.pi_e_r:.5f}\n"
            f"      Mean time in system         : {self.W_sess:.4f}\n"
            f"      Retry amplification factor  : {self.A:.4f}\n"
            f"      Total data intensity        : {self.Lambda_e:.4f}\n"
            f"      Total blocked data intensity: {self.Lambda_e_b:.4f}\n"
            f"      Primary intensity           : {self.Lambda_e_p:.4f}\n"
            f"      Primary blocked intensity   : {self.Lambda_e_p_b:.4f}\n"
            f"      Retry intensity             : {self.Lambda_e_r:.4f}\n"
            f"      Retry blocked intensity     : {self.Lambda_e_r_b:.4f}\n"
        )
