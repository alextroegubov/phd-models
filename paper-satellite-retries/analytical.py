"""Analytical model for the satellite communication system with real-time and data flows."""

from __future__ import annotations

import logging
import itertools
import time
from math import comb
from dataclasses import dataclass
import numpy as np
from utils import ParametersSet, Metrics
from numba import njit
from typing import ClassVar

logger = logging.getLogger(__name__)
logging.basicConfig(filename="analytical.log", filemode="w", level=logging.INFO, encoding="utf-8")


@dataclass(frozen=True, slots=True)
class State:
    """State of the markov process"""

    # (i_1, ..., i_n) - number of RT requests for k-th RT flow
    i_vec: tuple
    # number of frozen and being served ET requests
    d: int
    # number of requests to retransmit
    r: int

    r_max: ClassVar[int] = 50

    def ik_(self, k: int, delta=1) -> State:
        """Get state (i_1, ..., i_k + delta, ..., i_n, d, r)"""
        i_vec = list(self.i_vec)
        i_vec[k] += delta
        return State(tuple(i_vec), self.d, self.r)

    def d_(self, delta=1) -> State:
        """Get state (i_1, ..., i_n, d + delta, r)"""
        return State(self.i_vec, self.d + delta, self.r)

    def r_(self, delta=1) -> State:
        """Get state (i_1, ..., i_n, d, r + delta)"""
        return State(self.i_vec, self.d, self.r + delta)

    def dr_(self, d_d=1, d_r=1) -> State:
        """Get state (i_1, ..., i_n, d + d_d, r + d_r)"""
        return State(self.i_vec, self.d + d_d, self.r + d_r)

    def __hash__(self) -> int:
        return hash((self.i_vec, self.d, self.r))


class Solver:
    """Analytical model for the satellite communication system with real-time and data flows."""

    def __init__(self, params: ParametersSet, max_eps: float, max_iter: int):
        """Initialize the solver

        `params`: ParametersSet
        `max_eps`: maximum error
        `max_iter`: maximum number of iterations
        """
        self.params: ParametersSet = params
        self.max_eps: float = max_eps
        self.max_iter: int = max_iter

    def init_state_list(self, r_max: int = 20):
        State.r_max = r_max
        self.state_list = self.get_possible_states()
        self.state_to_idx = {state: idx for idx, state in enumerate(self.state_list)}
        self.p = np.full(len(self.state_list), 1e-2, dtype=np.float64)

        logger.info("Init solver: r_max=%d, number of states: %d", r_max, len(self.state_list))

    def precompute_state_indices(self):
        """Precompute state indices for faster access during the iterations"""

        n_flows = self.params.real_time_flows
        # return index of state in the state_list or -1 if state is not in the state space
        index_of = lambda state: self.state_to_idx.get(state, -1)

        self.idx_d_plus_1 = np.array([index_of(state.d_(1)) for state in self.state_list], dtype=np.int32)

        self.idx_r_plus_1 = np.array([index_of(state.r_(1)) for state in self.state_list], dtype=np.int32)

        self.idx_d_plus_1_r_minus_1 = np.array(
            [index_of(state.dr_(d_d=1, d_r=-1)) for state in self.state_list], dtype=np.int32
        )

        self.idx_d_minus_1_r_plus_1 = np.array(
            [index_of(state.dr_(d_d=-1, d_r=1)) for state in self.state_list], dtype=np.int32
        )

        self.idx_rt_minus = np.array(
            [[index_of(state.ik_(k, -1)) for k in range(n_flows)] for state in self.state_list], dtype=np.int32
        )

        self.idx_rt_plus = np.array(
            [[index_of(state.ik_(k, +1)) for k in range(n_flows)] for state in self.state_list], dtype=np.int32
        )

    def precompute_denominator(self):
        """Precompute demoninator for SEE for each state"""

        n_flows = self.params.real_time_flows
        lamb = np.array(self.params.real_time_lambdas)
        mu = np.array(self.params.real_time_mus)
        b = np.array(self.params.real_time_resources)

        lambda_e = self.params.data_lambda
        mu_e = self.params.data_mu
        b_min = self.params.data_resources_min
        b_max = self.params.data_resources_max

        v = self.params.beam_capacity
        sigma = self.params.queue_intensity
        nu = self.params.retry_intensity
        H = self.params.retry_probability

        batch_probs = self.params.data_batch_probs
        batch_sizes = np.arange(1, len(batch_probs) + 1, dtype=np.float64)
        at_least_one_retry_prob = float(np.sum([f_s * (1 - (1 - H) ** s) for f_s, s in zip(batch_probs, batch_sizes)]))

        self.denominator = np.zeros(len(self.state_list), dtype=np.float64)

        self.l_arr = np.array(
            [np.dot(state.i_vec, b) for state in self.state_list],
            dtype=np.float64,
        )
        self.q_arr = np.array(
            [max(0, state.d - (v - l) // b_min) for state, l in zip(self.state_list, self.l_arr)],
            dtype=np.float64,
        )
        self.q_prime_arr = np.array(
            [max(0, state.d + 1 - (v - l) // b_min) for state, l in zip(self.state_list, self.l_arr)],
            dtype=np.float64,
        )
        for idx, state in enumerate(self.state_list):
            i_vec, d, r = state.i_vec, state.d, state.r
            l = self.l_arr[idx]
            q = self.q_arr[idx]
            # accept new RT request
            real_time_arrival_d = sum(lamb[k] * (l + b[k] <= v) for k in range(n_flows))
            # serve RT request
            real_time_serv_d = sum(i_vec[k] * mu[k] * (i_vec[k] > 0) for k in range(n_flows))

            # accept at least one ET request
            data_arr_accept_d = lambda_e * (l + d * b_min + b_min <= v)
            # accept no ET requests and at least one is retried
            data_arr_reject_d = lambda_e * at_least_one_retry_prob * (l + d * b_min + b_min > v)

            # serve ET request
            data_serv_d = (mu_e / b_min) * min(v - l, (d - q) * b_max)
            # go to retries or leaves the system from freeze queue
            freeze_d = q * sigma * (q > 0)

            # accept retry request
            retry_accept_d = r * nu * (l + d * b_min + b_min <= v)
            # reject retry request and it leaves the system
            retry_reject_d = r * nu * (1 - H) * (l + d * b_min + b_min > v)

            self.denominator[idx] = (
                real_time_arrival_d
                + real_time_serv_d
                + data_arr_accept_d
                + data_arr_reject_d
                + data_serv_d
                + freeze_d
                + retry_accept_d
                + retry_reject_d
            )

    def precompute_numerator_coefs(self):
        """Precompute numerator coefs for SEE for each state."""

        n_flows = self.params.real_time_flows
        lamb = np.array(self.params.real_time_lambdas)
        mu = np.array(self.params.real_time_mus)
        b = np.array(self.params.real_time_resources)

        lambda_e = self.params.data_lambda
        mu_e = self.params.data_mu
        b_min = self.params.data_resources_min
        b_max = self.params.data_resources_max
        batch_probs = self.params.data_batch_probs
        batch_sizes = list(range(1, len(batch_probs) + 1))

        v = self.params.beam_capacity
        sigma = self.params.queue_intensity
        nu = self.params.retry_intensity
        H = self.params.retry_probability

        self.data_serv_n_coef = np.zeros(len(self.state_list), dtype=np.float64)
        self.freeze_n_coef = np.zeros(len(self.state_list), dtype=np.float64)
        self.freeze_out_n_coef = np.zeros(len(self.state_list), dtype=np.float64)
        self.retry_accept_n_coef = np.zeros(len(self.state_list), dtype=np.float64)
        self.retry_reject_n_coef = np.zeros(len(self.state_list), dtype=np.float64)
        self.real_time_arr_n_coefs = np.zeros((len(self.state_list), n_flows), dtype=np.float64)
        self.real_time_serv_n_coefs = np.zeros((len(self.state_list), n_flows), dtype=np.float64)

        # For each target state, we have a sum of multiple [source_state_prob * coef] terms.
        # The number of terms is variable and depends on the target state, so we need to store the offsets.
        # batch_offsets[idx + 1] - batch_offsets[idx] is the number of terms for the target state at index idx.
        # batch_src_indices and batch_coefs are the source states and coefficients for the terms.
        # For the state self.state_list[idx], the slice is [batch_offsets[idx], ..., batch_offsets[idx + 1] - 1]

        batch_src_indices: list[int] = []
        batch_coefs: list[float] = []
        batch_offsets = np.zeros(len(self.state_list) + 1, dtype=np.int32)

        for idx, state in enumerate(self.state_list):
            i_vec, d, r = state.i_vec, state.d, state.r

            l = self.l_arr[idx]
            c = int((v - l) // b_min)
            q_prime = self.q_prime_arr[idx]

            # Incoming transitions caused by primary ET batch arrivals.
            # from state (i_vec, d0, r - m) to state (i_vec, d, r)
            #
            # a(d0) = maximum number of ET to accept
            # j(d0, s) = number of accepted ET requests from a batch of size s
            # z(d0, s) = number of rejected ET requests from that batch
            # m <= z(d0, s) = number of retried ET requests
            for d0 in range(d + 1):
                for s, f_s in zip(batch_sizes, batch_probs):
                    accepted = min(s, max(0, c - d0))
                    rejected = s - accepted

                    if d != d0 + accepted:
                        continue

                    for m in range(min(rejected, r) + 1):
                        # Exclude self-transition: no accepted requests and no new retries.
                        if accepted == 0 and m == 0:
                            continue

                        src_idx = self.state_to_idx.get(State(i_vec, d0, r - m), -1)
                        coef = lambda_e * f_s * comb(rejected, m) * (H**m) * ((1 - H) ** (rejected - m))
                        if coef > 0 and src_idx >= 0:
                            batch_src_indices.append(src_idx)
                            batch_coefs.append(float(coef))

            batch_offsets[idx + 1] = len(batch_src_indices)

            self.data_serv_n_coef[idx] = (mu_e / b_min) * min(v - l, (d + 1 - q_prime) * b_max)

            self.freeze_n_coef[idx] = q_prime * sigma * H * (q_prime > 0 and r > 0)
            self.freeze_out_n_coef[idx] = q_prime * sigma * (1 - H) * (q_prime > 0)

            self.retry_accept_n_coef[idx] = (r + 1) * nu * (d > 0 and l + (d - 1) * b_min + b_min <= v)
            self.retry_reject_n_coef[idx] = (r + 1) * nu * (1 - H) * (l + d * b_min + b_min > v)

            self.real_time_arr_n_coefs[idx] = np.array(
                [lamb[k] * (i_vec[k] > 0) for k in range(n_flows)], dtype=np.float64
            )
            self.real_time_serv_n_coefs[idx] = np.array(
                [(i_vec[k] + 1) * mu[k] * (l + b[k] <= v) for k in range(n_flows)], dtype=np.float64
            )

        self.batch_arr_n_offsets = batch_offsets
        self.batch_arr_n_src_indices = np.array(batch_src_indices, dtype=np.int32)
        self.batch_arr_n_coefs = np.array(batch_coefs, dtype=np.float64)

    def get_possible_states(self) -> list[State]:
        """Get possible states for markov process."""
        beam_capacity = self.params.beam_capacity
        b_min = self.params.data_resources_min

        # d = 0, 1, ..., beam_capacity // b_min
        data_flow_states = np.arange(beam_capacity // b_min + 1, dtype=int)
        # r = 0, 1, ..., State.r_max
        retries_states = np.arange(State.r_max, dtype=int)
        # i_k = 0, 1, ..., beam_capacity // b_k
        real_time_flows_states = [np.arange(beam_capacity // b + 1, dtype=int) for b in self.params.real_time_resources]

        # product of possible values
        states = itertools.product(*real_time_flows_states, data_flow_states, retries_states)
        # filter by capacity
        states_lst = [
            State(s[:-2], int(s[-2]), int(s[-1]))
            for s in states
            if np.dot(s[:-2], self.params.real_time_resources) <= beam_capacity
        ]

        return states_lst

    def solve_with_r_max(self, r_min: int, step: int, max_attempts: int) -> bool:
        """Solve the model with different r_max values."""
        is_valid = False
        attempt = 0

        self.init_state_list(r_max=r_min + attempt * step)

        while not is_valid and attempt < max_attempts:
            logger.info("--------------------------------")
            logger.info("Attempt %d", attempt)

            if attempt > 0:
                logger.info("Using previous probabilities as initial guess")
                # save previous iterations probabilities
                old_state_to_prob = dict(zip(self.state_list, self.p))
                # new state list with new r_max
                self.init_state_list(r_min + attempt * step)
                # update known probabilities
                self.p = np.array([old_state_to_prob.get(state, 1e-20) for state in self.state_list], dtype=np.float64)
                self.p /= self.p.sum()

            it, error = self.solve()
            logger.info("\tFinal: it=%d, error=%2.10f", it, error)

            metrics = self.calculate_metrics()
            logger.info("\t%s", metrics)
            is_valid = self.check_solution(metrics)
            logger.info("\tSolution is valid: %s", is_valid)
            attempt += 1

        return bool(is_valid)

    def solve(self):
        """Solve the model."""
        logger.info("Start solving the model")
        start_time = time.time()

        self.precompute_state_indices()
        self.precompute_denominator()
        self.precompute_numerator_coefs()

        iteration, error = solve_numba(
            self.p,
            self.max_eps,
            self.max_iter,
            self.denominator,
            self.idx_rt_minus,
            self.idx_rt_plus,
            self.idx_d_plus_1,
            self.idx_r_plus_1,
            self.idx_d_plus_1_r_minus_1,
            self.idx_d_minus_1_r_plus_1,
            self.real_time_arr_n_coefs,
            self.real_time_serv_n_coefs,
            self.batch_arr_n_offsets,
            self.batch_arr_n_src_indices,
            self.batch_arr_n_coefs,
            self.data_serv_n_coef,
            self.freeze_n_coef,
            self.freeze_out_n_coef,
            self.retry_accept_n_coef,
            self.retry_reject_n_coef,
        )

        logger.info(
            "Model solved in %d iterations for %4.2f sec",
            iteration,
            time.time() - start_time,
        )

        self.p = self.p / self.p.sum()

        return iteration, error

    def retry_boundary_mass(self) -> float:
        batch_size_max = len(self.params.data_batch_probs)
        r_arr = np.array([state.r for state in self.state_list], dtype=np.int32)

        # State.r_max currently means r = 0, ..., State.r_max - 1
        boundary_start = max(0, State.r_max - batch_size_max)

        return float(np.sum(self.p * (r_arr >= boundary_start)))

    def calculate_metrics(self) -> Metrics:
        """Calculate metrics for the model."""

        rt_flows = self.params.real_time_flows

        v = self.params.beam_capacity
        lambda_e = self.params.data_lambda
        nu = self.params.retry_intensity
        sigma = self.params.queue_intensity
        b_min = self.params.data_resources_min
        b_max = self.params.data_resources_max
        H = self.params.retry_probability
        mu_e = self.params.data_mu
        batch_probs = self.params.data_batch_probs
        batch_sizes = np.arange(1, len(batch_probs) + 1, dtype=np.float64)

        d_arr = np.array([state.d for state in self.state_list], dtype=np.int32)
        r_arr = np.array([state.r for state in self.state_list], dtype=np.int32)
        i_vec_arr = np.array([np.array(state.i_vec, dtype=np.int32) for state in self.state_list], dtype=np.int32)

        b_k = self.params.real_time_resources
        pi_k = [np.sum(self.p * (self.l_arr + b_k[k] > v)) for k in range(rt_flows)]
        y_k = [np.sum(self.p * i_vec_arr[:, k]) for k in range(rt_flows)]
        m_k = [y_k[k] * b_k[k] for k in range(rt_flows)]

        y_r = np.sum(self.p * r_arr)
        y_q = np.sum(self.p * self.q_arr)
        y_d = np.sum(self.p * d_arr)

        y_e = np.sum(self.p * (d_arr - self.q_arr))
        m_e = np.sum(self.p * np.minimum(v - self.l_arr, (d_arr - self.q_arr) * b_max))
        b_e = m_e / y_e

        mean_batch_size = float(np.sum(batch_probs * batch_sizes))
        c_arr = (v - self.l_arr) // b_min
        F_l_d_arr = np.array(
            [
                np.sum(
                    [
                        f_s * (s * (d_arr >= c_arr) + np.maximum(0, s + d_arr - c_arr) * (d_arr < c_arr))
                        for f_s, s in zip(batch_probs, batch_sizes)
                    ],
                    axis=0,
                )
            ],
        )
        # primary intensity
        Lambda_e_p = lambda_e * mean_batch_size
        # primary blocked intensity
        Lambda_e_p_b = float(np.sum(self.p * lambda_e * F_l_d_arr))

        # retry intensity
        Lambda_e_r = y_r * nu
        # retry blocked intensity
        Lambda_e_r_b = float(np.sum(self.p * r_arr * nu * (self.l_arr + (d_arr + 1) * b_min > v)))

        # total_data_intensity
        Lambda_e = Lambda_e_r + Lambda_e_p
        # total blocked data intensity
        Lambda_e_b = Lambda_e_p_b + Lambda_e_r_b

        pi_e_0 = Lambda_e_p_b / Lambda_e_p
        pi_e_a = Lambda_e_b / Lambda_e
        pi_e_r = (1.0 - H) * (Lambda_e_b + y_q * sigma) / Lambda_e_p

        W_sess = y_d / (m_e * mu_e / b_min + y_q * sigma)

        A = Lambda_e / Lambda_e_p

        util = sum(m_k) + m_e

        logger.info("\tRetry boundary mass: %.12f", self.retry_boundary_mass())

        return Metrics(
            rt_request_rej_prob=[float(x) for x in pi_k],
            mean_rt_requests_in_service=[float(x) for x in y_k],
            mean_resources_per_rt_flow=[float(x) for x in m_k],
            mean_retry_requests=float(y_r),
            mean_freeze_requests=float(y_q),
            mean_data_requests_in_system=float(y_d),
            mean_data_requests_in_service=float(y_e),
            mean_resources_per_data_flow=float(m_e),
            mean_resources_per_data_request=float(b_e),
            primary_intensity=float(Lambda_e_p),
            primary_blocked_intensity=float(Lambda_e_p_b),
            retry_intensity=float(Lambda_e_r),
            retry_blocked_intensity=float(Lambda_e_r_b),
            total_blocked_data_intensity=float(Lambda_e_b),
            primary_request_reject_prob=float(pi_e_0),
            attempt_request_reject_prob=float(pi_e_a),
            not_serviced_request_prob=float(pi_e_r),
            mean_data_request_in_system_time=float(W_sess),
            retry_amplification_factor=float(A),
            beam_utilization=float(util),
        )

    @staticmethod
    def check_balance(name: str, lhs: float, rhs: float, rtol: float = 5e-4, atol: float = 1e-8) -> bool:
        abs_err = abs(lhs - rhs)
        rel_err = abs_err / max(abs(lhs), abs(rhs), 1.0)
        ok = bool(np.isclose(lhs, rhs, rtol=rtol, atol=atol))

        logger.info(
            "%s balance lhs=%.5f rhs=%.5f abs_err=%.10e rel_err=%.10e ok=%s",
            name,
            lhs,
            rhs,
            abs_err,
            rel_err,
            ok,
        )
        return ok

    def check_solution(self, metrics: Metrics):
        """Check conservation laws for the stationary distribution."""

        mu_e = self.params.data_mu
        sigma = self.params.queue_intensity
        H = self.params.retry_probability
        b_min = self.params.data_resources_min

        y_q = metrics.y_q
        m_e = metrics.m_e

        Lambda_e_r = metrics.Lambda_e_r
        Lambda_e_b = metrics.Lambda_e_b
        Lambda_e = metrics.Lambda_e
        Lambda_e_p = metrics.Lambda_e_p

        # Real-time flow balances:
        # lambda_k * (1 - pi_k) * b_k = m_k * mu_k
        real_time_balances = [
            self.check_balance(
                name=f"Real-time flow {k}",
                lhs=lambda_k * (1.0 - metrics.pi_k[k]) * b_k,
                rhs=metrics.m_k[k] * mu_k,
            )
            for k, (lambda_k, mu_k, b_k) in enumerate(
                zip(self.params.real_time_lambdas, self.params.real_time_mus, self.params.real_time_resources)
            )
        ]

        # Retry orbit balance:
        # Lambda_e_r = H * (Lambda_e_b + y_q * sigma)
        retry_balance = self.check_balance(
            name="Retry flow",
            lhs=Lambda_e_r,
            rhs=H * (Lambda_e_b + y_q * sigma),
        )

        # Total elastic flow balance:
        # Lambda_e = Lambda_e_b + y_q * sigma + m_e * mu_e / b_min
        elastic_balance = self.check_balance(
            name="Elastic flow",
            lhs=Lambda_e,
            rhs=Lambda_e_b + y_q * sigma + m_e * mu_e / b_min,
        )

        # Primary elastic flow balance:
        # Lambda_e_p = m_e * mu_e / b_min + (1 - H) * (Lambda_e_b + y_q * sigma)
        primary_elastic_balance = self.check_balance(
            name="Primary elastic flow",
            lhs=Lambda_e_p,
            rhs=m_e * mu_e / b_min + (1.0 - H) * (Lambda_e_b + y_q * sigma),
        )

        return all(real_time_balances) and retry_balance and elastic_balance and primary_elastic_balance


@njit(cache=True)
def solve_numba(
    p,
    max_eps,
    max_iter,
    denominator,
    idx_rt_minus,
    idx_rt_plus,
    idx_d_plus_1,
    idx_r_plus_1,
    idx_d_plus_1_r_minus_1,
    idx_d_minus_1_r_plus_1,
    real_time_arr_n_coefs,
    real_time_serv_n_coefs,
    batch_arr_n_offsets,
    batch_arr_n_src_indices,
    batch_arr_n_coefs,
    data_serv_n_coef,
    freeze_n_coef,
    freeze_out_n_coef,
    retry_accept_n_coef,
    retry_reject_n_coef,
):
    # Gaus-Seidel procedure
    iteration = 0
    error = 1e10

    while (error > max_eps or iteration < 10) and iteration < max_iter:
        iteration += 1
        n_states = p.shape[0]
        n_flows = idx_rt_minus.shape[1]

        max_diff = 0.0

        for idx in range(n_states):
            denr = denominator[idx]
            num = 0.0

            # RT terms
            for k in range(n_flows):
                j = idx_rt_minus[idx, k]
                if j >= 0:
                    num += p[j] * real_time_arr_n_coefs[idx, k]

                j = idx_rt_plus[idx, k]
                if j >= 0:
                    num += p[j] * real_time_serv_n_coefs[idx, k]

            # Primary ET batch arrivals.
            # Incoming transitions are precomputed because one batch can change
            # both d and r: (d0, r - m) -> (d, r).
            for t in range(batch_arr_n_offsets[idx], batch_arr_n_offsets[idx + 1]):
                j = batch_arr_n_src_indices[t]
                num += p[j] * batch_arr_n_coefs[t]

            # ET service
            j = idx_d_plus_1[idx]
            if j >= 0:
                num += p[j] * data_serv_n_coef[idx]
                # frozen -> leave system
                num += p[j] * freeze_out_n_coef[idx]

            # frozen -> retry
            j = idx_d_plus_1_r_minus_1[idx]
            if j >= 0:
                num += p[j] * freeze_n_coef[idx]

            # retry accepted
            j = idx_d_minus_1_r_plus_1[idx]
            if j >= 0:
                num += p[j] * retry_accept_n_coef[idx]

            # retry rejected and leaves
            j = idx_r_plus_1[idx]
            if j >= 0:
                num += p[j] * retry_reject_n_coef[idx]

            new_prob = num / denr
            diff = abs(new_prob - p[idx])
            if diff > max_diff:
                max_diff = diff

            p[idx] = new_prob

        error = max_diff

    return iteration, error


def main():
    """Main function. Parses args from command line and runs simulation"""
    # parser = get_argparser()
    # args = parser.parse_args()

    params = ParametersSet(
        real_time_flows=2,
        real_time_lambdas=[4, 2],
        real_time_mus=[1, 1],
        real_time_resources=[4, 8],
        data_resources_min=2,
        data_resources_max=3,
        data_lambda=10,
        data_mu=2,
        queue_intensity=1,
        retry_intensity=1,
        retry_probability=0.8,
        beam_capacity=80,
        data_batch_probs=[0.2, 0.2, 0.1, 0.1, 0.2, 0.2],
    )

    logger.info("%s", params)
    solver = Solver(params, 1e-9, 2500)
    is_valid = solver.solve_with_r_max(r_min=90, step=10, max_attempts=10)
    logger.info("Solution is valid: %s", is_valid)


if __name__ == "__main__":
    main()
