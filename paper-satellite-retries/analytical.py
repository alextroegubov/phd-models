"""Analytical model for the satellite communication system with real-time and data flows."""

from __future__ import annotations

import argparse
import logging
import itertools
import time
from dataclasses import dataclass
import numpy as np
from utils import ParametersSet, Metrics

from typing import ClassVar

logger = logging.getLogger(__name__)
logging.basicConfig(filename="analytical.log", filemode="w", level=logging.INFO, encoding="utf-8")


@dataclass(frozen=True, slots=True)
class State:
    """State of the markov process"""

    # (i_1, ..., i_n) - number of RT requests for k-th RT flow
    i_vec: tuple[int, ...]
    # number of frozen and being served ET requests
    d: int
    # number of UT to retransmit
    r: int

    # v: ClassVar[int] = 0

    r_max: ClassVar[int] = 40

    def ik_(self, k: int, delta=1):
        """Get state (i_1, ..., i_k + delta, ..., i_n, d, r)"""
        i_vec = list(self.i_vec)
        i_vec[k] += delta
        return State(tuple(i_vec), self.d, self.r)

    def d_(self, delta=1):
        """Get state (i_1, ..., i_n, d + delta, r)"""
        return State(self.i_vec, self.d + delta, self.r)

    def r_(self, delta=1):
        """Get state (i_1, ..., i_n, d, r + delta)"""
        return State(self.i_vec, self.d, self.r + delta)

    def dr_(self, d_d=1, d_r=1):
        """Get state (i_1, ..., i_n, d + d_d, r + d_r)"""
        return State(self.i_vec, self.d + d_d, self.r + d_r)

    def __hash__(self):
        # Combine the hash of both fields for a unique hash value
        return hash((self.i_vec, self.d, self.r))


class Solver:
    """Analytical model for the satellite communication system with real-time and data flows."""

    def __init__(self, params: ParametersSet, max_eps: float, max_iter: int):
        """Initialize the solver

        :param params: ParametersSet
        :param max_eps: maximum error
        :param max_iter: maximum number of iterations
        """
        self.params: ParametersSet = params
        self.max_eps: float = max_eps
        self.max_iter: int = max_iter

        logger.info("%s", self.params)

        self.states: dict[State, float] = {
            state: 1e-2 for state in self.get_possible_states()
        }

        logger.info("Number of states: %d", len(self.states))

    def get_possible_states(self) -> list[State]:
        """Get possible states for markov process."""
        beam_capacity = self.params.beam_capacity
        b_min = self.params.data_resources_min    

        max_data_flows = beam_capacity
        data_flow_states = np.arange(max_data_flows // b_min + 1, dtype=int)

        real_time_flows_states = [
            np.arange(beam_capacity // b + 1, dtype=int) for b in self.params.real_time_resources
        ]

        retries_states = np.arange(State.r_max, dtype=int)

        # product of possible values
        states = itertools.product(*real_time_flows_states, data_flow_states, retries_states)
        # filter by capacity
        states_lst = [
            State(s[:-2], s[-2], s[-1])
            for s in states
            if np.dot(s[:-2], self.params.real_time_resources) <= beam_capacity
        ]

        return states_lst

    def solve(self):
        """Solve the model."""
        n = self.params.real_time_flows
        lamb = np.array(self.params.real_time_lambdas)
        mu = np.array(self.params.real_time_mus)
        b = np.array(self.params.real_time_resources)

        lambda_e = self.params.data_lambda
        mu_e = self.params.data_mu
        b_min = self.params.data_resources_min

        v = self.params.beam_capacity
        sigma = self.params.queue_intensity
        nu = self.params.retry_intensity
        H = self.params.leave_probability

        flows_idx = list(range(n))

        error = 1000
        iteration = 0

        logger.info("Start solving the model")
        start_time = time.time()

        while error > self.max_eps and iteration < self.max_iter:
            iteration += 1
            error = 0

            for st, prev_prob in self.states.items():
                i_vec, d, r = np.array(st.i_vec), st.d, st.r
                l = np.dot(i_vec, b)
                q = max(0, d - (v - l) // b_min)
                q_prime = max(0, d + 1 - (v - l) // b_min)
                # accept new RT request
                real_time_arrival_d = sum(lamb[k] * (l + b[k] <= v) for k in range(n))
                # serve RT request
                real_time_serv_d = sum(i_vec[k] * mu[k] * (i_vec[k] > 0) for k in range(n))

                # accept ET request
                data_arr_accept_d = lambda_e * (l + d * b_min + b_min <= v)
                # reject ET request and it is retried
                data_arr_reject_d = lambda_e * H * (l + d * b_min + b_min > v)

                # serve ET request
                data_serv_d = mu_e * (v - l) * (d - q > 0)
                # go to retries from freeze queue
                freeze_d = q * sigma * (q > 0)

                # accept retry request
                retry_accept_d = r * nu * (l + d * b_min + b_min <= v)
                # reject retry request and it leaves the system
                retry_reject_d = r * nu * (1 - H) * (l + d * b_min + b_min > v)

                # denominator
                denr = (
                    real_time_arrival_d
                    + real_time_serv_d
                    + data_arr_accept_d
                    + data_arr_reject_d
                    + data_serv_d
                    + freeze_d
                    + retry_accept_d
                    + retry_reject_d
                )

                # accept new RT request
                real_time_arr_n = sum(
                    self.states.get(st.ik_(k, -1), 0) * lamb[k] * (i_vec[k] > 0) for k in flows_idx
                )
                # serve RT request
                real_time_serv_n = sum(
                    self.states.get(st.ik_(k, +1), 0) * (i_vec[k] + 1) * mu[k] * (l + b[k] <= v)
                    for k in flows_idx
                )

                # accept ET request
                data_arr_accept_n = (
                    self.states.get(st.d_(-1), 0) * lambda_e * (l + (d - 1) * b_min + b_min <= v and d > 0)
                )
                # reject ET request and it is retried
                data_arr_reject_n = (
                    self.states.get(st.r_(-1), 0) * lambda_e * H * (r > 0 and l + d * b_min + b_min > v)
                )

                # serve ET request
                data_serv_n = self.states.get(st.d_(1), 0) * mu_e * (v - l) * (d + 1 - q_prime > 0)
                # go to retries from freeze queue
                freeze_n = (
                    q_prime
                    * sigma
                    * H
                    * self.states.get(st.dr_(d_d=1, d_r=-1), 0)
                    * (q_prime > 0 and r > 0)
                )
                # go out of the system from freeze queue
                freeze_out_n = (
                    q_prime
                    * sigma
                    * (1 - H)
                    * self.states.get(st.d_(1), 0)
                    * (q_prime > 0)
                )                

                # accept retry request
                retry_accept_n = (
                    (r + 1)
                    * nu
                    * self.states.get(st.dr_(d_d=-1, d_r=1), 0)
                    * (d > 0 and l + (d - 1) * b_min + b_min <= v)
                )
                # reject retry request and it leaves the system
                retry_reject_n = (r + 1) * nu * (1 - H) * self.states.get(st.r_(1), 0) * (l + d * b_min + b_min > v)

                numr = (
                    real_time_arr_n
                    + real_time_serv_n
                    + data_arr_accept_n
                    + data_arr_reject_n
                    + data_serv_n
                    + freeze_n
                    + freeze_out_n
                    + retry_accept_n
                    + retry_reject_n
                )
                error += abs(prev_prob - numr / denr) / prev_prob
                self.states[st] = numr / denr

            if iteration % 50 == 0:
                logger.info(
                    "Iteration %d: error = %2.8f, time = %4.2f",
                    iteration,
                    error,
                    time.time() - start_time,
                )

        logger.info(
            "Model solved in %d iterations for %4.2f sec",
            iteration,
            time.time() - start_time,
        )
        norm = sum(self.states.values())

        for key, value in self.states.items():
            self.states[key] = value / norm

        metrics = self.calculate_metrics()
        logger.info("%s", metrics)

        self.check_solution(metrics)

        return iteration, error

    def calculate_metrics(self) -> Metrics:
        """Calculate metrics for the model."""

        rt_flows = self.params.real_time_flows

        metrics = Metrics(
            rt_request_rej_prob=[0] * rt_flows,
            mean_rt_requests_in_service=[0] * rt_flows,
            mean_resources_per_rt_flow=[0] * rt_flows,
        )

        v = self.params.beam_capacity
        lambda_e = self.params.data_lambda
        nu = self.params.retry_intensity
        b_min = self.params.data_resources_min

        for state in self.states.keys():
            i_vec, d, r = state.i_vec, state.d, state.r
            l = np.dot(i_vec, self.params.real_time_resources)
            q = max(0, d - (v - l) // b_min)

            for k in range(rt_flows):
                b_k = self.params.real_time_resources[k]
                metrics.mean_rt_requests_in_service[k] += i_vec[k] * self.states[state]

                metrics.rt_request_rej_prob[k] += self.states[state] * (l + b_k > v)

            metrics.mean_freeze_requests += self.states[state] * q
            metrics.mean_retry_requests += self.states[state] * r
            metrics.intensity_blocked_requests += (
                self.states[state] * (l + d * b_min + b_min > v) * (lambda_e + r * nu)
            )
            metrics.mean_data_requests_in_service += self.states[state] * (d - q) * (d - q > 0)
            metrics.mean_resources_per_data_flow += self.states[state] * (v - l) * (d - q > 0)

        metrics.intensity_all_requests = lambda_e + metrics.mean_retry_requests * nu

        for k in range(rt_flows):
            metrics.mean_resources_per_rt_flow[k] = (
                metrics.mean_rt_requests_in_service[k] * self.params.real_time_resources[k]
            )

        metrics.mean_resources_per_data_request = (
            metrics.mean_resources_per_data_flow / metrics.mean_data_requests_in_service
        )

        metrics.beam_utilization = (
            sum(metrics.mean_resources_per_rt_flow) + metrics.mean_resources_per_data_flow
        )
        return metrics

    def check_solution(self, metrics: Metrics):
        """Check the solution."""

        # real-time traffic flows
        for k in range(self.params.real_time_flows):
            lambda_k = self.params.real_time_lambdas[k]
            mu_k = self.params.real_time_mus[k]
            b_k = self.params.real_time_resources[k]

            pi_k = metrics.rt_request_rej_prob[k]
            m_k = metrics.mean_resources_per_rt_flow[k]

            logger.info(
                "Real-time flow %d balance (%2.5f, %2.5f): %s",
                k,
                lambda_k * (1 - pi_k) * b_k,
                m_k * mu_k,
                np.isclose(lambda_k * (1 - pi_k) * b_k, m_k * mu_k),
            )
        # retry flow
        y_r = metrics.mean_retry_requests
        y_q = metrics.mean_freeze_requests
        Lambda_b = metrics.intensity_blocked_requests
        H = self.params.leave_probability
        sigma = self.params.queue_intensity
        nu = self.params.retry_intensity

        logger.info(
            "Retry flow balance (%2.5f, %2.5f): %s",
            y_r * nu,
            Lambda_b * H + y_q * sigma * H,
            np.isclose(y_r * nu, Lambda_b * H + y_q * sigma * H),
        )

        # elastic data flow
        Lambda = metrics.intensity_all_requests
        mu_e = self.params.data_mu
        m_e = metrics.mean_resources_per_data_flow
        logger.info(
            "Elastic flow balance (%2.5f, %2.5f): %s",
            Lambda,
            Lambda_b + y_q * sigma + m_e * mu_e,
            np.isclose(Lambda, Lambda_b + y_q * sigma + m_e * mu_e),
        )

        # elastic data flow
        L_e = (1 - H) * (metrics.intensity_blocked_requests + y_q * sigma)
        mu_e = self.params.data_mu
        m_e = metrics.mean_resources_per_data_flow
        lambda_e = self.params.data_lambda
        logger.info(
            "Primary elastic flow balance (%2.5f, %2.5f): %s",
            lambda_e,
            L_e + m_e * mu_e,
            np.isclose(lambda_e, L_e + m_e * mu_e),
        )


def main():
    """Main function. Parses args from command line and runs simulation"""
    # parser = get_argparser()
    # args = parser.parse_args()

    params = ParametersSet(
        real_time_flows=1,
        real_time_lambdas=[2],
        real_time_mus=[1],
        real_time_resources=[4],
        data_resources_min=2,
        data_resources_max=100,
        data_lambda=10,
        data_mu=1,
        beam_capacity=30,
        queue_intensity=1,
        retry_intensity=1,
        leave_probability=0.8,
    )

    solver = Solver(params, 1e-7, 800)
    print(len(solver.states))
    it, error = solver.solve()
    logger.info("Final: it=%d, error=%2.10f", it, error)

    # metrics, it, error = solver.solve()


if __name__ == "__main__":
    main()
