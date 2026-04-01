"""Analytical model for the satellite communication system with real-time and data flows."""

from __future__ import annotations

import logging
import itertools
import time
from dataclasses import dataclass
import numpy as np
from utils import ParametersSet, Metrics
from numba import njit
from typing import ClassVar

logger = logging.getLogger(__name__)
# logging.basicConfig(filename="analytical.log", filemode="w", level=logging.INFO, encoding="utf-8")


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

        self.idx_d_minus_1 = np.array([index_of(state.d_(-1)) for state in self.state_list], dtype=np.int32)

        self.idx_r_plus_1 = np.array([index_of(state.r_(1)) for state in self.state_list], dtype=np.int32)

        self.idx_r_minus_1 = np.array([index_of(state.r_(-1)) for state in self.state_list], dtype=np.int32)

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

        v = self.params.beam_capacity
        sigma = self.params.queue_intensity
        nu = self.params.retry_intensity
        H = self.params.retry_probability

        self.denominator = np.zeros(len(self.state_list), dtype=np.float64)

        rt_resources = np.array(self.params.real_time_resources)

        self.l_arr = np.array(
            [np.dot(state.i_vec, rt_resources) for state in self.state_list],
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

            # accept ET request
            data_arr_accept_d = lambda_e * (l + d * b_min + b_min <= v)
            # reject ET request and it is retried
            data_arr_reject_d = lambda_e * H * (l + d * b_min + b_min > v)

            # serve ET request
            data_serv_d = mu_e * (v - l) * (d - q > 0)
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
        """Precompute numerator coefs for SEE for each state"""

        n_flows = self.params.real_time_flows
        lamb = np.array(self.params.real_time_lambdas)
        mu = np.array(self.params.real_time_mus)
        b = np.array(self.params.real_time_resources)

        lambda_e = self.params.data_lambda
        mu_e = self.params.data_mu
        b_min = self.params.data_resources_min

        v = self.params.beam_capacity
        sigma = self.params.queue_intensity
        nu = self.params.retry_intensity
        H = self.params.retry_probability

        self.data_arr_accept_n_coef = np.zeros(len(self.state_list), dtype=np.float64)
        self.data_arr_reject_n_coef = np.zeros(len(self.state_list), dtype=np.float64)
        self.data_serv_n_coef = np.zeros(len(self.state_list), dtype=np.float64)
        self.freeze_n_coef = np.zeros(len(self.state_list), dtype=np.float64)
        self.freeze_out_n_coef = np.zeros(len(self.state_list), dtype=np.float64)
        self.retry_accept_n_coef = np.zeros(len(self.state_list), dtype=np.float64)
        self.retry_reject_n_coef = np.zeros(len(self.state_list), dtype=np.float64)
        self.real_time_arr_n_coefs = np.zeros((len(self.state_list), n_flows), dtype=np.float64)
        self.real_time_serv_n_coefs = np.zeros((len(self.state_list), n_flows), dtype=np.float64)

        for idx, state in enumerate(self.state_list):
            i_vec, d, r = state.i_vec, state.d, state.r

            l = self.l_arr[idx]
            q_prime = self.q_prime_arr[idx]

            self.data_arr_accept_n_coef[idx] = lambda_e * (l + (d - 1) * b_min + b_min <= v and d > 0)
            self.data_arr_reject_n_coef[idx] = lambda_e * H * (r > 0 and l + d * b_min + b_min > v)

            self.data_serv_n_coef[idx] = mu_e * (v - l) * (d + 1 - q_prime > 0)

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
        while not is_valid and attempt < max_attempts:
            logger.info("--------------------------------")
            logger.info("Attempt %d", attempt)
            self.init_state_list(r_min + attempt * step)
            logger.info("\tNumber of states: %d", len(self.state_list))
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
            self.idx_d_minus_1,
            self.idx_d_plus_1,
            self.idx_r_minus_1,
            self.idx_r_plus_1,
            self.idx_d_plus_1_r_minus_1,
            self.idx_d_minus_1_r_plus_1,
            self.real_time_arr_n_coefs,
            self.real_time_serv_n_coefs,
            self.data_arr_accept_n_coef,
            self.data_arr_reject_n_coef,
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

    def calculate_metrics(self) -> Metrics:
        """Calculate metrics for the model."""

        rt_flows = self.params.real_time_flows

        v = self.params.beam_capacity
        lambda_e = self.params.data_lambda
        nu = self.params.retry_intensity
        sigma = self.params.queue_intensity
        b_min = self.params.data_resources_min
        H = self.params.retry_probability
        mu_e = self.params.data_mu

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

        y_e = np.sum(self.p * (d_arr - self.q_arr) * (d_arr - self.q_arr > 0))
        m_e = np.sum(self.p * (v - self.l_arr) * (d_arr - self.q_arr > 0))
        b_e = m_e / y_e

        Lambda_e_b = np.sum(self.p * (lambda_e + r_arr * nu) * (self.l_arr + d_arr * b_min + b_min > v))
        Lambda_e = lambda_e + y_r * nu

        pi_e_0 = np.sum(self.p * (self.l_arr + (d_arr + 1) * b_min > v))
        pi_e_a = Lambda_e_b / Lambda_e
        pi_e_r = (1 - H) * (Lambda_e_b + y_q * sigma) / lambda_e
        W_sess = np.sum(self.p * d_arr) / (m_e * mu_e + y_q * sigma)
        A = Lambda_e / lambda_e

        util = sum(m_k) + m_e

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
            intensity_all_requests=float(Lambda_e),
            intensity_blocked_requests=float(Lambda_e_b),
            primary_data_request_reject_prob=float(pi_e_0),
            data_request_attempt_reject_prob=float(pi_e_a),
            data_request_not_serviced_prob=float(pi_e_r),
            mean_data_request_in_system_time=float(W_sess),
            retry_amplification_factor=float(A),
            beam_utilization=float(util),
        )

    def check_solution(self, metrics: Metrics):
        """Check the solution."""

        # real-time traffic flows
        real_time_balances = []
        for k in range(self.params.real_time_flows):
            lambda_k = self.params.real_time_lambdas[k]
            mu_k = self.params.real_time_mus[k]
            b_k = self.params.real_time_resources[k]

            pi_k = metrics.pi_k[k]
            m_k = metrics.m_k[k]
            real_time_balance_k = np.isclose(lambda_k * (1 - pi_k) * b_k, m_k * mu_k)
            real_time_balances.append(real_time_balance_k)
            logger.info(
                "Real-time flow %d balance (%2.5f, %2.5f): %s",
                k,
                lambda_k * (1 - pi_k) * b_k,
                m_k * mu_k,
                real_time_balance_k,
            )

        # retry flow
        y_r = metrics.y_r
        y_q = metrics.y_q
        Lambda_e_b = metrics.Lambda_e_b
        H = self.params.retry_probability
        sigma = self.params.queue_intensity
        nu = self.params.retry_intensity

        retry_balance = np.isclose(y_r * nu, Lambda_e_b * H + y_q * sigma * H)
        logger.info(
            "Retry flow balance (%2.5f, %2.5f): %s",
            y_r * nu,
            Lambda_e_b * H + y_q * sigma * H,
            retry_balance,
        )

        # elastic data flow
        Lambda_e = metrics.Lambda_e
        mu_e = self.params.data_mu
        m_e = metrics.mean_resources_per_data_flow
        elastic_balance_1 = np.isclose(Lambda_e, Lambda_e_b + y_q * sigma + m_e * mu_e)
        logger.info(
            "Elastic flow balance (%2.5f, %2.5f): %s",
            Lambda_e,
            Lambda_e_b + y_q * sigma + m_e * mu_e,
            elastic_balance_1,
        )

        # elastic data flow
        L_e = (1 - H) * (Lambda_e_b + y_q * sigma)
        lambda_e = self.params.data_lambda
        elastic_balance_2 = np.isclose(lambda_e, L_e + m_e * mu_e)
        logger.info(
            "Primary elastic flow balance (%2.5f, %2.5f): %s",
            lambda_e,
            L_e + m_e * mu_e,
            elastic_balance_2,
        )

        return all(real_time_balances) and retry_balance and elastic_balance_1 and elastic_balance_2


@njit(cache=True)
def solve_numba(
    p,
    max_eps,
    max_iter,
    denominator,
    idx_rt_minus,
    idx_rt_plus,
    idx_d_minus_1,
    idx_d_plus_1,
    idx_r_minus_1,
    idx_r_plus_1,
    idx_d_plus_1_r_minus_1,
    idx_d_minus_1_r_plus_1,
    real_time_arr_n_coefs,
    real_time_serv_n_coefs,
    data_arr_accept_n_coef,
    data_arr_reject_n_coef,
    data_serv_n_coef,
    freeze_n_coef,
    freeze_out_n_coef,
    retry_accept_n_coef,
    retry_reject_n_coef,
):
    # Gaus-Seidel procedure
    iteration = 0
    error = 1e10

    while error > max_eps and iteration < max_iter:
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

            # ET accept
            j = idx_d_minus_1[idx]
            if j >= 0:
                num += p[j] * data_arr_accept_n_coef[idx]

            # ET reject -> retry
            j = idx_r_minus_1[idx]
            if j >= 0:
                num += p[j] * data_arr_reject_n_coef[idx]

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
        data_lambda=10,
        data_mu=1,
        queue_intensity=1,
        retry_intensity=1,
        retry_probability=0.8,
        beam_capacity=80,
    )

    logger.info("%s", params)
    solver = Solver(params, 1e-7, 7500)
    is_valid = solver.solve_with_r_max(r_min=20, step=10, max_attempts=10)
    logger.info("Solution is valid: %s", is_valid)


if __name__ == "__main__":
    main()
