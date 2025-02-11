"""Approximate models and two special cases"""

from __future__ import annotations

import argparse
import logging
import numpy as np
from utils import ParametersSet, Metrics

logger = logging.getLogger(__name__)
logging.basicConfig(
    filename="approximation.log", filemode="w", level=logging.INFO, encoding="utf-8"
)


class SolverOnlyRealTime:
    """Solver for the real-time model w/o elastic traffic."""

    def __init__(self, params: ParameterSet):
        self.params = params
        self.p = []

    def solve(self):
        """Solve the real-time model"
        :return: Metrics
        """
        v = self.params.beam_capacity
        n = self.params.real_time_flows
        lamb = self.params.real_time_lambdas
        mu = self.params.real_time_mus
        b = self.params.real_time_resources
        p = np.ones(v + 1)

        for l in range(1, v + 1):
            p[l] = (1 / l) * sum(
                (lamb[k] / mu[k]) * b[k] * p[l - b[k]] * (l - b[k] >= 0) for k in range(n)
            )

        p = p / np.sum(p)
        self.p = p
        metrics = self.calculate_metrics(p)

        return metrics

    @staticmethod
    def solve_approx(a, b, v):
        """Solve real-time model for the given a = lambda/mu, b and v"""
        n = len(a)
        p = np.ones(v + 1)
        for l in range(1, v + 1):
            p[l] = (1 / l) * sum(a[k] * b[k] * p[l - b[k]] * (l - b[k] >= 0) for k in range(n))

        p = p / np.sum(p)

        return p

    def calculate_metrics(self, p: np.ndarray) -> Metrics:
        """Calculate metrics from the probabilities p"""
        rt_flows = self.params.real_time_flows
        v = self.params.beam_capacity
        b = self.params.real_time_resources
        lamb = self.params.real_time_lambdas
        mu = self.params.real_time_mus

        metrics = Metrics(
            rt_request_rej_prob=[0] * rt_flows,
            mean_rt_requests_in_service=[0] * rt_flows,
            mean_resources_per_rt_flow=[0] * rt_flows,
        )
        for k in range(rt_flows):
            metrics.rt_request_rej_prob[k] = sum(p[l] for l in range(v - b[k] + 1, v + 1))
            metrics.mean_resources_per_rt_flow[k] = (
                b[k] * lamb[k] / mu[k] * (1 - metrics.rt_request_rej_prob[k])
            )
            metrics.mean_rt_requests_in_service[k] = metrics.mean_resources_per_rt_flow[k] / b[k]

        metrics.beam_utilization = sum(metrics.mean_resources_per_rt_flow)
        return metrics


class SolverOnlyElasticData:
    """Solver for the elastic data model w/o real-time traffic."""

    def __init__(self, params: ParameterSet):
        self.params: ParametersSet = params
        self.p = []

    def solve(self):
        """Solve the elastic data model
        :return Metrics
        """
        v = self.params.beam_capacity
        lamb = self.params.data_lambda
        mu = self.params.data_mu
        b_min = self.params.data_resources_min
        b_max = self.params.data_resources_max
        f = self.params.data_requests_batch_probs

        p = np.ones(v // b_min + 1)

        for d in range(1, v // b_min + 1):
            mu_d = mu / b_min * min(v, d * b_max)
            p[d] = lamb / mu_d * sum(p[d - j] * sum(f[j - 1 :]) for j in range(1, d + 1))

        p = p / np.sum(p)
        self.p = p
        metrics = self.calculate_metrics(p)

        return metrics

    def solve_approx(self, v):
        """Solve elastic data model for the given v"""
        lamb = self.params.data_lambda
        mu = self.params.data_mu
        b_min = self.params.data_resources_min
        b_max = self.params.data_resources_max
        f = self.params.data_requests_batch_probs

        p = np.ones(v // b_min + 1)

        for d in range(1, v // b_min + 1):
            mu_d = mu / b_min * min(v, d * b_max)
            p[d] = lamb / mu_d * sum(p[d - j] * sum(f[j - 1 :]) for j in range(1, d + 1))

        p = p / np.sum(p)

        mean_data_requests_per_batch = sum(s * f[s - 1] for s in range(1, len(f) + 1))

        rej_numer = sum(
            p[d] * sum(f[s - 1] * max(s + d - (v // b_min), 0) for s in range(1, len(f) + 1))
            for d in range(0, v // b_min + 1)
        )

        data_request_rej_prob = 1 / mean_data_requests_per_batch * rej_numer

        mean_data_requests_in_service = sum(p[d] * d for d in range(0, v // b_min + 1))

        mean_data_request_service_time = mean_data_requests_in_service / (
            lamb * (1 - data_request_rej_prob) * mean_data_requests_per_batch
        )

        return mean_data_requests_in_service, mean_data_request_service_time

    def calculate_metrics(self, p: np.ndarray) -> Metrics:
        """Calculate metrics from the probabilities p"""

        v = self.params.beam_capacity
        lamb = self.params.data_lambda
        mu = self.params.data_mu
        b_min = self.params.data_resources_min
        b_max = self.params.data_resources_max
        f = self.params.data_requests_batch_probs

        metrics = Metrics(
            rt_request_rej_prob=[],
            mean_rt_requests_in_service=[],
            mean_resources_per_rt_flow=[],
        )

        metrics.mean_data_requests_per_batch = sum(s * f[s - 1] for s in range(1, len(f) + 1))

        rej_numer = sum(
            p[d] * sum(f[s - 1] * max(s + d - (v // b_min), 0) for s in range(1, len(f) + 1))
            for d in range(0, v // b_min + 1)
        )

        metrics.data_request_rej_prob = 1 / metrics.mean_data_requests_per_batch * rej_numer

        metrics.mean_resources_per_data_flow = sum(
            p[d] * min(v, d * b_max) for d in range(0, v // b_min + 1)
        )
        metrics.mean_data_requests_in_service = sum(p[d] * d for d in range(0, v // b_min + 1))
        metrics.mean_resources_per_data_request = (
            metrics.mean_resources_per_data_flow / metrics.mean_data_requests_in_service
        )
        metrics.mean_data_request_service_time = metrics.mean_data_requests_in_service / (
            lamb * (1 - metrics.data_request_rej_prob) * metrics.mean_data_requests_per_batch
        )

        metrics.beam_utilization = metrics.mean_resources_per_data_flow

        return metrics


class SolveApprox:
    """Approximate solver for multiservice model"""

    def __init__(self, params: ParametersSet):
        self.params = params

    def solve(self):
        n = self.params.real_time_flows
        v = self.params.beam_capacity
        b = self.params.real_time_resources
        lamb = self.params.real_time_lambdas
        mu = self.params.real_time_mus

        lamb_e = self.params.data_lambda
        mu_e = self.params.data_mu
        b_min = self.params.data_resources_min

        a = [lamb[k] / mu[k] for k in range(n)]

        # pi_e^(1)
        a_new = a.copy()
        a_new.append(lamb_e / mu_e)
        b_new = list(b)
        b_new.append(b_min)
        p = SolverOnlyRealTime.solve_approx(a_new, b_new, v)
        pi_e_1 = sum(p[l] for l in range(v - b_min + 1, v + 1))

        # W_2, y_2, b_e_2
        p = SolverOnlyRealTime.solve_approx(a, b, v)
        ed_solver = SolverOnlyElasticData(self.params)

        y_2 = 0
        W_2 = 0
        for l in range(0, v - b_min + 1):
            y_e, w = ed_solver.solve_approx(v - l)
            y_2 += p[l] * y_e
            W_2 += p[l] * w

        b_e_2 = b_min / (mu_e * W_2)
        return pi_e_1, y_2, W_2, b_e_2


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
        data_requests_batch_probs=args.data_requests_batch_probs,
    )

    solver_rt = SolverOnlyRealTime(params)
    metrics_rt = solver_rt.solve()

    solver_data = SolverOnlyElasticData(params)
    metrics_data = solver_data.solve()

    # approx = SolveApprox(params)

    # pi_e_1, y_2, W_2, b_e_2 = approx.solve()

    logger.info("%s", str(metrics_rt))
    logger.info("%s", str(metrics_data))
    # logger.info("%s", f"{pi_e_1=}, {y_2=}, {W_2=}, {b_e_2=}")
