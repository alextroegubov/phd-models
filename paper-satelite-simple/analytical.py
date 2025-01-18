"""Analytical model for the satellite communication system with real-time and data flows."""

from __future__ import annotations

import argparse
import logging
import itertools
import time
from dataclasses import dataclass
import numpy as np
from utils import ParametersSet, Metrics

logger = logging.getLogger(__name__)
logging.basicConfig(filename="analytical.log", filemode="w", level=logging.INFO, encoding="utf-8")


class Solver:
    """Analytical model for the satellite communication system with real-time and data flows."""

    @dataclass(frozen=True)
    class State:
        """State of the markov process"""

        i_vec: tuple[int, ...]
        d: int

        def ik_(self, k: int, delta=1):
            i_vec = list(self.i_vec)
            i_vec[k] += delta
            return Solver.State(tuple(i_vec), self.d)

        def d_(self, delta=1):
            return Solver.State(self.i_vec, self.d + delta)

        def __hash__(self):
            # Combine the hash of both fields for a unique hash value
            return hash((self.i_vec, self.d))

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

        self.states: dict[Solver.State, float] = {
            state: 1e-2 for state in self.get_possible_states()
        }

        logger.info("Number of states: %d", len(self.states))

    def get_possible_states(self) -> list[State]:
        """Get possible states for markov process."""
        beam_capacity = self.params.beam_capacity
        b_min = self.params.data_resources_min

        max_data_flows = beam_capacity // self.params.data_resources_min
        data_flow_states = np.arange(max_data_flows + 1, dtype=int)

        real_time_flows_states = [
            np.arange(beam_capacity // b + 1, dtype=int) for b in self.params.real_time_resources
        ]

        # product of possible values
        states = itertools.product(*real_time_flows_states, data_flow_states)
        # filter by capacity
        states_lst = [
            Solver.State(s[:-1], s[-1])
            for s in states
            if np.dot(s[:-1], self.params.real_time_resources) + s[-1] * b_min <= beam_capacity
        ]

        return states_lst

    def solve(self):
        """Solve the model."""
        n = self.params.real_time_flows
        lamb = np.array(self.params.real_time_lambdas)
        mu = np.array(self.params.real_time_mus)
        b = np.array(self.params.real_time_resources)

        b_min = self.params.data_resources_min
        b_max = self.params.data_resources_max
        lambda_e = self.params.data_lambda
        mu_e = self.params.data_mu

        v = self.params.beam_capacity

        flows_idx = list(range(n))

        error = 1000
        iteration = 0

        logger.info("Start solving the model")
        start_time = time.time()
        while error > self.max_eps and iteration < self.max_iter:
            iteration += 1
            error = 0

            for st, prev_prob in self.states.items():
                i_vec, d = st.i_vec, st.d
                i_vec = np.array(i_vec)

                l = np.dot(i_vec, b)
                l_tot = l + d * b_min

                real_time_arrival_d = sum(lamb[k] * (l_tot + b[k] <= v) for k in range(n))
                real_time_serv_d = sum(i_vec[k] * mu[k] * (i_vec[k] > 0) for k in range(n))

                data_arr_d = lambda_e * (l_tot + b_min <= v)
                data_serv_d = mu_e / b_min * (d > 0) * min(v - l, d * b_max)

                denr = real_time_arrival_d + real_time_serv_d + data_arr_d + data_serv_d

                real_time_arr_n = sum(
                    self.states.get(st.ik_(k, -1), 0) * lamb[k] * (i_vec[k] > 0) for k in flows_idx
                )
                real_time_serv_n = sum(
                    self.states.get(st.ik_(k, +1), 0) * (i_vec[k] + 1) * mu[k] * (l_tot + b[k] <= v)
                    for k in range(n)
                )
                data_arr_n = self.states.get(st.d_(-1), 0) * lambda_e * (l_tot <= v and d > 0)
                data_serv_n = (
                    self.states.get(st.d_(+1), 0.0)
                    * (l_tot + b_min <= v and d + 1 > 0)
                    * mu_e
                    / b_min
                    * min(v - l, b_max * (d + 1))
                )

                numr = real_time_arr_n + real_time_serv_n + data_arr_n + data_serv_n
                error += abs(prev_prob - numr / denr) / prev_prob
                self.states[st] = numr / denr

            if iteration % 500 == 0:
                # use lazy formatting
                logger.info(
                    "Iteration %d: error = %2.8f, time = %4.2f",
                    iteration,
                    error,
                    time.time() - start_time,
                )

        logger.info(
            "Model solved in %d iterations for %4.2f sec", iteration, time.time() - start_time
        )
        norm = sum(self.states.values())

        for key, value in self.states.items():
            self.states[key] = value / norm

        metrics = self.calculate_metrics()
        logger.info("%s", metrics)

        self.check_solution(metrics)

        return metrics, iteration, error

    def calculate_metrics(self) -> Metrics:
        """Calculate metrics for the model."""

        rt_flows = self.params.real_time_flows

        metrics = Metrics(
            rt_request_rej_prob=[0] * rt_flows,
            mean_rt_requests_in_service=[0] * rt_flows,
            mean_resources_per_rt_flow=[0] * rt_flows,
        )

        for state in self.states.keys():
            i_vec, d = state.i_vec, state.d
            b_min = self.params.data_resources_min

            v = self.params.beam_capacity
            l = np.dot(i_vec, self.params.real_time_resources) + b_min * d

            for k in range(rt_flows):
                b_k = self.params.real_time_resources[k]
                metrics.mean_rt_requests_in_service[k] += i_vec[k] * self.states[state]

                if l + b_k > v:
                    metrics.rt_request_rej_prob[k] += self.states[state]

            if l + b_min > v:
                metrics.data_request_rej_prob += self.states[state]

            b_max = self.params.data_resources_max
            data_res = min(v - np.dot(i_vec, self.params.real_time_resources), d * b_max)
            metrics.mean_resources_per_data_flow += (d > 0) * data_res * self.states[state]
            metrics.mean_data_requests_in_service += d * self.states[state]

        for k in range(rt_flows):
            metrics.mean_resources_per_rt_flow[k] = (
                metrics.mean_rt_requests_in_service[k] * self.params.real_time_resources[k]
            )

        metrics.mean_resources_per_data_request = (
            metrics.mean_resources_per_data_flow / metrics.mean_data_requests_in_service
        )

        metrics.mean_data_request_service_time = metrics.mean_data_requests_in_service / (
            self.params.data_lambda * 1 * (1 - metrics.data_request_rej_prob)
        )

        metrics.beam_utilization = (
            sum(metrics.mean_resources_per_rt_flow) + metrics.mean_resources_per_data_flow
        )
        return metrics

    def check_solution(self, metrics: Metrics):
        """Check the solution."""

        # elastic data flow
        lambda_e = self.params.data_lambda
        pi_e = metrics.data_request_rej_prob
        mu_e = self.params.data_mu
        m_e = metrics.mean_resources_per_data_flow
        b_min = self.params.data_resources_min

        logger.info(
            "Data flow balance: %s", np.isclose(lambda_e * (1 - pi_e) * b_min, m_e * mu_e)
        )

        # real-time traffic flows
        for k in range(self.params.real_time_flows):
            lambda_k = self.params.real_time_lambdas[k]
            mu_k = self.params.real_time_mus[k]
            b_k = self.params.real_time_resources[k]

            pi_k = metrics.rt_request_rej_prob[k]
            m_k = metrics.mean_resources_per_rt_flow[k]

            logger.info(
                "Real-time flow %d balance: %s",
                k,
                np.isclose(lambda_k * (1 - pi_k) * b_k, m_k * mu_k),
            )


def get_argparser():
    """Return the argument parser."""
    parser = argparse.ArgumentParser(
        description="Satellite network simulation: n real-time flows, 1 elastic data flow"
    )

    # Real-time flow parameters
    parser.add_argument(
        "--real_time_flows",
        type=int,
        default=2,
        help="Number of real-time data flows in the network",
    )
    parser.add_argument(
        "--real_time_lambdas",
        type=float,
        nargs="+",
        default=[15.0, 3.0],
        help="Arrival rates (lambda) for real-time data flows (space-separated list)",
    )
    parser.add_argument(
        "--real_time_mus",
        type=float,
        nargs="+",
        default=[0.2, 0.2],
        help="Service rates (mu) for real-time data flows (space-separated list)",
    )
    parser.add_argument(
        "--real_time_resources",
        type=int,
        nargs="+",
        default=[1, 5],
        help="Resource units required by each real-time flow (space-separated list)",
    )

    # Elastic data flow parameters
    parser.add_argument(
        "--data_resources_min",
        type=int,
        default=1,
        help="Minimum resource units allocated for elastic data flow",
    )
    parser.add_argument(
        "--data_resources_max",
        type=int,
        default=10,
        help="Maximum resource units allocated for elastic data flow",
    )
    parser.add_argument(
        "--data_lambda",
        type=float,
        default=2.5,
        help="Arrival rate (lambda) for elastic data flow",
    )
    parser.add_argument(
        "--data_mu",
        type=float,
        default=1.0,
        help="Service rate (mu) for elastic data flow",
    )

    # Beam capacity parameter
    parser.add_argument(
        "--beam_capacity",
        type=int,
        default=100,
        help="Total beam capacity in resource units",
    )

    # Simulation parameters
    parser.add_argument(
        "--max_error",
        type=float,
        default=1e-7,
        help="Maximum error in the iterative solution",
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=35e3,
        help="Maximum number of iterations in the iterative solution",
    )

    return parser


if __name__ == "__main__":
    parser = get_argparser()
    args = parser.parse_args()

    params = ParametersSet(
        real_time_flows=args.real_time_flows,
        real_time_lambdas=args.real_time_lambdas,
        real_time_mus=args.real_time_mus,
        real_time_resources=args.real_time_resources,
        data_resources_min=args.data_resources_min,
        data_resources_max=args.data_resources_max,
        data_lambda=args.data_lambda,
        data_mu=args.data_mu,
        beam_capacity=args.beam_capacity,
    )

    # params = ParametersSet(1, [0.042], [1 / 300], [3], 1, 5, 4.2, 1 / 16, 50)
    solver = Solver(params, args.max_error, args.max_iter)
    metrics, it, error = solver.solve()

    print(metrics)