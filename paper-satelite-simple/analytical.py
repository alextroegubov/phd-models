"""Analytical model for the satellite communication system with real-time and data flows."""

from __future__ import annotations

import argparse
import logging
import itertools
import time
from dataclasses import dataclass
import numpy as np
from numba import njit
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
            """Get state (i_1, ..., i_k + delta, ..., i_n, d)"""
            i_vec = list(self.i_vec)
            i_vec[k] += delta
            return Solver.State(tuple(i_vec), self.d)

        def d_(self, delta=1):
            """Get state (i_1, ..., i_n, d + delta)"""
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

        self.state_list: list[Solver.State] = self.get_possible_states()
        self.state_to_idx: dict[Solver.State, int] = {
            state: idx for idx, state in enumerate(self.state_list)
        }
        self.p: np.ndarray = np.full(len(self.state_list), 1e-2, dtype=np.float64)

        logger.info("Number of states: %d", len(self.state_list))

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

    def precompute_state_indices(self):
        """Precompute state neighbor indices for fast array-based lookup."""
        n_flows = self.params.real_time_flows
        S = len(self.params.data_requests_batch_probs)
        index_of = lambda state: self.state_to_idx.get(state, -1)

        self.idx_d_plus_1 = np.array(
            [index_of(st.d_(+1)) for st in self.state_list], dtype=np.int32
        )
        self.idx_d_minus_s = np.array(
            [[index_of(st.d_(-s)) for s in range(1, S + 1)] for st in self.state_list],
            dtype=np.int32,
        )
        self.idx_rt_minus = np.array(
            [[index_of(st.ik_(k, -1)) for k in range(n_flows)] for st in self.state_list],
            dtype=np.int32,
        )
        self.idx_rt_plus = np.array(
            [[index_of(st.ik_(k, +1)) for k in range(n_flows)] for st in self.state_list],
            dtype=np.int32,
        )

    def precompute_denominator(self):
        """Precompute denominator for the Gauss-Seidel equation for each state."""
        n = self.params.real_time_flows
        lamb = np.array(self.params.real_time_lambdas)
        mu = np.array(self.params.real_time_mus)
        b = np.array(self.params.real_time_resources)
        b_min = self.params.data_resources_min
        b_max = self.params.data_resources_max
        lambda_e = self.params.data_lambda
        mu_e = self.params.data_mu
        v = self.params.beam_capacity
        rt_resources = np.array(self.params.real_time_resources)

        n_states = len(self.state_list)
        self.denominator = np.zeros(n_states, dtype=np.float64)
        self.l_arr = np.zeros(n_states, dtype=np.float64)

        for idx, st in enumerate(self.state_list):
            i_vec = np.array(st.i_vec)
            d = st.d

            l = np.dot(i_vec, rt_resources)
            l_tot = l + d * b_min
            self.l_arr[idx] = l

            rt_arrival_d = sum(lamb[k] * (l_tot + b[k] <= v) for k in range(n))
            rt_serv_d = sum(i_vec[k] * mu[k] for k in range(n))
            data_arr_d = lambda_e * (l_tot + b_min <= v)
            data_serv_d = mu_e / b_min * (d > 0) * min(v - l, d * b_max)

            self.denominator[idx] = rt_arrival_d + rt_serv_d + data_arr_d + data_serv_d

    def precompute_numerator_coefs(self):
        """Precompute numerator coefficients for the Gauss-Seidel equation."""
        n = self.params.real_time_flows
        lamb = np.array(self.params.real_time_lambdas)
        mu = np.array(self.params.real_time_mus)
        b = np.array(self.params.real_time_resources)
        b_min = self.params.data_resources_min
        b_max = self.params.data_resources_max
        lambda_e = self.params.data_lambda
        mu_e = self.params.data_mu
        v = self.params.beam_capacity
        f = np.array(self.params.data_requests_batch_probs)
        S = len(f)
        rt_resources = np.array(self.params.real_time_resources)

        n_states = len(self.state_list)
        self.real_time_arr_n_coefs = np.zeros((n_states, n), dtype=np.float64)
        self.real_time_serv_n_coefs = np.zeros((n_states, n), dtype=np.float64)
        self.data_arr_n_coefs = np.zeros((n_states, S), dtype=np.float64)
        self.data_serv_n_coef = np.zeros(n_states, dtype=np.float64)

        f_tail_sums = np.zeros(S, dtype=np.float64)
        for s_idx in range(S):
            f_tail_sums[s_idx] = np.sum(f[s_idx + 1:]) if s_idx + 1 < S else 0.0

        for idx, st in enumerate(self.state_list):
            i_vec = np.array(st.i_vec)
            d = st.d
            l = self.l_arr[idx]
            l_tot = l + d * b_min

            for k in range(n):
                self.real_time_arr_n_coefs[idx, k] = lamb[k] * (i_vec[k] > 0)
                self.real_time_serv_n_coefs[idx, k] = (i_vec[k] + 1) * mu[k] * (l_tot + b[k] <= v)

            for s_idx in range(S):
                s = s_idx + 1
                if s <= d:
                    self.data_arr_n_coefs[idx, s_idx] = lambda_e * (
                        f[s_idx] + (v - l_tot < b_min) * f_tail_sums[s_idx]
                    )

            d_plus_1 = d + 1
            l_tot_plus_1 = l + d_plus_1 * b_min
            self.data_serv_n_coef[idx] = (
                mu_e / b_min
                * min(v - l, b_max * d_plus_1)
                * (l_tot_plus_1 <= v and d_plus_1 > 0)
            )

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
            self.idx_d_minus_s,
            self.idx_d_plus_1,
            self.real_time_arr_n_coefs,
            self.real_time_serv_n_coefs,
            self.data_arr_n_coefs,
            self.data_serv_n_coef,
        )

        logger.info(
            "Model solved in %d iterations for %4.2f sec", iteration, time.time() - start_time
        )

        self.p = self.p / self.p.sum()

        self.states = {st: self.p[idx] for idx, st in enumerate(self.state_list)}

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

        b_min = self.params.data_resources_min
        f = self.params.data_requests_batch_probs
        v = self.params.beam_capacity

        pi_e_num = 0
        #d_s
        mean_batch_size = sum(s * f[s-1] for s in range(1, len(f) + 1))

        metrics.mean_data_requests_per_batch = mean_batch_size

        for state in self.states.keys():
            i_vec, d = state.i_vec, state.d
            l_hat = np.dot(i_vec, self.params.real_time_resources) + b_min * d

            for k in range(rt_flows):
                b_k = self.params.real_time_resources[k]
                metrics.mean_rt_requests_in_service[k] += i_vec[k] * self.states[state]

                metrics.rt_request_rej_prob[k] += self.states[state] * (l_hat + b_k > v)

            pi_e_num += self.states[state] * sum(
                f[s-1] * max(s - (v - l_hat) // b_min, 0) for s in range(1, len(f) + 1)
            )

            b_max = self.params.data_resources_max
            data_res = min(v - np.dot(i_vec, self.params.real_time_resources), d * b_max)
            metrics.mean_resources_per_data_flow += (d > 0) * data_res * self.states[state]
            metrics.mean_data_requests_in_service += d * self.states[state]

        metrics.data_request_rej_prob = pi_e_num / mean_batch_size

        for k in range(rt_flows):
            metrics.mean_resources_per_rt_flow[k] = (
                metrics.mean_rt_requests_in_service[k] * self.params.real_time_resources[k]
            )

        metrics.mean_resources_per_data_request = (
            metrics.mean_resources_per_data_flow / metrics.mean_data_requests_in_service
        )

        metrics.mean_data_request_service_time = metrics.mean_data_requests_in_service / (
            self.params.data_lambda * metrics.mean_data_requests_per_batch * (1 - metrics.data_request_rej_prob)
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
        d_s = metrics.mean_data_requests_per_batch

        logger.info(
            "Data flow balance: %s", np.isclose(lambda_e * (1 - pi_e) * b_min * d_s, m_e * mu_e)
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


@njit(cache=True)
def solve_numba(
    p,
    max_eps,
    max_iter,
    denominator,
    idx_rt_minus,
    idx_rt_plus,
    idx_d_minus_s,
    idx_d_plus_1,
    real_time_arr_n_coefs,
    real_time_serv_n_coefs,
    data_arr_n_coefs,
    data_serv_n_coef,
):
    iteration = 0
    error = 1e10
    n_states = p.shape[0]
    n_flows = idx_rt_minus.shape[1]
    n_batch = idx_d_minus_s.shape[1]

    while error > max_eps and iteration < max_iter:
        iteration += 1
        max_diff = 0.0

        for idx in range(n_states):
            num = 0.0

            for k in range(n_flows):
                j = idx_rt_minus[idx, k]
                if j >= 0:
                    num += p[j] * real_time_arr_n_coefs[idx, k]

                j = idx_rt_plus[idx, k]
                if j >= 0:
                    num += p[j] * real_time_serv_n_coefs[idx, k]

            for s_idx in range(n_batch):
                j = idx_d_minus_s[idx, s_idx]
                if j >= 0:
                    num += p[j] * data_arr_n_coefs[idx, s_idx]

            j = idx_d_plus_1[idx]
            if j >= 0:
                num += p[j] * data_serv_n_coef[idx]

            new_prob = num / denominator[idx]
            diff = abs(new_prob - p[idx])
            if diff > max_diff:
                max_diff = diff
            p[idx] = new_prob

        error = max_diff

    return iteration, error


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
    parser.add_argument(
        "--data_requests_batch_probs",
        type=float,
        nargs="+",
        default=[1 / 3, 1 / 3, 1 / 3],
        help="Batch probs f_s, s = 1, ..., B",
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


def main():
    """ Main function. Parses args from command line and runs simulation"""
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
        data_requests_batch_probs=args.data_requests_batch_probs,
        beam_capacity=args.beam_capacity,
    )

    solver = Solver(params, args.max_error, args.max_iter)
    metrics, it, error = solver.solve()

    print(metrics)


if __name__ == "__main__":
    main()