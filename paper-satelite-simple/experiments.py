from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from utils import ParametersSet, Metrics
import simulation as sim
import analytical as ana
import approximation as appr
import numpy as np

LAMBDA_2 = 2
V_MAX = 100
test_params_1 = ParametersSet(
    real_time_flows=2,
    real_time_lambdas=[2 * LAMBDA_2, LAMBDA_2],
    real_time_mus=[1, 1],
    real_time_resources=[1, 4],
    data_resources_min=2,
    data_resources_max=8,
    data_lambda=5 * LAMBDA_2,
    data_mu=1,
    beam_capacity=V_MAX,
)


class ExperimentValidation:
    """Experiment to validate the analytical model against the simulation."""

    save_path = "results/validation"

    def __init__(self, max_error, max_iter, warmup, events, num_exec=8):
        self.max_error = max_error
        self.max_iter = max_iter
        self.warmup = warmup
        self.events = events
        self.num_exec = num_exec

        (Path.cwd() / self.save_path).mkdir(parents=True, exist_ok=True)

    def run(self, type_="ana"):
        assert type_ in ["ana", "sim"]

        task = None
        if type_ == "ana":
            task = self.task_analytical
        elif type_ == "sim":
            task = self.task_simulation

        assert task is not None

        futures = {}
        with ProcessPoolExecutor(self.num_exec) as executor:
            for v in range(10, test_params_1.beam_capacity + 1):
                run_params = ParametersSet(
                    real_time_flows=test_params_1.real_time_flows,
                    real_time_lambdas=test_params_1.real_time_lambdas,
                    real_time_mus=test_params_1.real_time_mus,
                    real_time_resources=test_params_1.real_time_resources,
                    data_resources_min=test_params_1.data_resources_min,
                    data_resources_max=test_params_1.data_resources_max,
                    data_lambda=test_params_1.data_lambda,
                    data_mu=test_params_1.data_mu,
                    beam_capacity=v,
                )
                futures[executor.submit(task, run_params)] = run_params

            for future in as_completed(futures):
                res_params = futures[future]
                res = future.result()
                self.save_result(res_params, res, type_)

    def save_result(self, params, res: Metrics, type_="ana"):
        assert type_ in ["ana", "sim"]
        filename = f"{type_}-v-{params.beam_capacity}.csv"

        full_path = Path.cwd() / self.save_path / filename

        with open(full_path, "w", encoding="utf-8") as f:
            f.write(
                f"{params.beam_capacity},"
                f"{res.rt_request_rej_prob[0]:.5f},"
                f"{res.rt_request_rej_prob[1]:.5f},"
                f"{res.data_request_rej_prob:.5f}"
            )

    def task_analytical(self, params):
        an_solver = ana.Solver(params, self.max_error, self.max_iter)
        an_metrics, _, _ = an_solver.solve()

        return an_metrics

    def task_simulation(self, params):

        sim.Simulator._instance = None
        sim.Simulator.num_instances = 0

        simulator = sim.Simulator.get_instance()
        network = sim.Network(params)
        network.add_flows(params.real_time_flows, 1)
        simulator.run(self.warmup)

        network.enable_stats_collection()
        for flow in network.flows_lookup.values():
            flow.enable_stats_collection()

        simulator.run(self.warmup + self.events)
        sim_metrics = sim.convert_stats_to_metrics(params, network)

        return sim_metrics


class ExperimentQoS:
    """Experiment to validate the analytical model against the simulation."""

    save_path = "results/qos"
    b_max_range = [8, 16, 100]

    def __init__(self, max_error, max_iter, num_exec=8):
        self.max_error = max_error
        self.max_iter = max_iter
        self.num_exec = num_exec

        (Path.cwd() / self.save_path).mkdir(parents=True, exist_ok=True)

    def run(self):

        task = self.task_analytical

        futures = {}
        with ProcessPoolExecutor(self.num_exec) as executor:
            for v in range(60, test_params_1.beam_capacity + 1):
                for b_max in self.b_max_range:
                    run_params = ParametersSet(
                        real_time_flows=test_params_1.real_time_flows,
                        real_time_lambdas=test_params_1.real_time_lambdas,
                        real_time_mus=test_params_1.real_time_mus,
                        real_time_resources=test_params_1.real_time_resources,
                        data_resources_min=test_params_1.data_resources_min,
                        data_resources_max=b_max,
                        data_lambda=test_params_1.data_lambda,
                        data_mu=test_params_1.data_mu,
                        beam_capacity=v,
                    )
                    futures[executor.submit(task, run_params)] = run_params

            for future in as_completed(futures):
                res_params = futures[future]
                res = future.result()
                self.save_result(res_params, res)

    def save_result(self, params: ParametersSet, res: Metrics):
        filename = f"ana-v-{params.beam_capacity}-bmax-{params.data_resources_max}.csv"

        full_path = Path.cwd() / self.save_path / filename

        with open(full_path, "w", encoding="utf-8") as f:
            f.write(
                f"{params.beam_capacity},"
                f"{params.data_resources_max},"
                f"{res.rt_request_rej_prob[0]:.5f},"
                f"{res.rt_request_rej_prob[1]:.5f},"
                f"{res.data_request_rej_prob:.5f},"
                f"{res.mean_data_request_service_time:.5f},"
                f"{res.mean_resources_per_data_request:.5f}"
            )

    def task_analytical(self, params):
        an_solver = ana.Solver(params, self.max_error, self.max_iter)
        an_metrics, _, _ = an_solver.solve()

        return an_metrics


class ExperimentResourcePlanning:
    """Experiment to validate the analytical model against the simulation."""

    save_path = "results/planning"
    pi_norm = 0.01
    W_norm = 1 / 3  # three times faster than at minimal rate
    B_MAX_RANGE = [8, 16, 200]
    MIN_V = 32  # rho = 1
    MAX_V = 100

    def __init__(self, max_error, max_iter, num_exec=8):
        self.max_error = max_error
        self.max_iter = max_iter
        self.num_exec = num_exec

        (Path.cwd() / self.save_path).mkdir(parents=True, exist_ok=True)

    def run(self, type_="ana"):
        assert type_ in ["ana", "approx"]

        task = None
        if type_ == "ana":
            task = self.task_analytical
        elif type_ == "approx":
            task = self.task_approximation

        assert task is not None

        futures = {}
        with ProcessPoolExecutor(self.num_exec) as executor:
            for v in range(self.MIN_V, self.MAX_V + 1):
                for b_max in self.B_MAX_RANGE:
                    run_params = ParametersSet(
                        real_time_flows=test_params_1.real_time_flows,
                        real_time_lambdas=test_params_1.real_time_lambdas,
                        real_time_mus=test_params_1.real_time_mus,
                        real_time_resources=test_params_1.real_time_resources,
                        data_resources_min=test_params_1.data_resources_min,
                        data_resources_max=b_max,
                        data_lambda=test_params_1.data_lambda,
                        data_mu=test_params_1.data_mu,
                        beam_capacity=v,
                    )
                    futures[executor.submit(task, run_params)] = run_params

            for future in as_completed(futures):
                res_params = futures[future]
                res = future.result()
                self.save_result(res_params, res, type_)

    def save_result(self, params: ParametersSet, res: Metrics | tuple, type_):
        filename = f"{type_}-v-{params.beam_capacity}-bmax-{params.data_resources_max}-bmin-{params.data_resources_min}.csv"
        full_path = Path.cwd() / self.save_path / filename

        if type_ == "ana" and isinstance(res, Metrics):
            W = res.mean_data_request_service_time
            max_pi = max(max(res.rt_request_rej_prob), res.data_request_rej_prob)

            res_str = (
                f"{params.beam_capacity},"
                f"{params.data_resources_max},"
                f"{W:.5f},"
                f"{max_pi:.5f}"
            )
        elif type_ == "approx" and isinstance(res, tuple):
            max_pi = res[0]
            W = res[1]

            res_str = f"{params.beam_capacity}, {params.data_resources_max}, {W:.5f}, {max_pi:.5f}"
        else:
            raise ValueError("Invalid type")

        if W <= self.W_norm and max_pi <= self.pi_norm:
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(res_str)

    def task_analytical(self, params):
        an_solver = ana.Solver(params, self.max_error, self.max_iter)
        an_metrics, _, _ = an_solver.solve()

        return an_metrics

    def task_approximation(self, params):

        lamb = params.real_time_lambdas
        mu = params.real_time_mus
        b = params.real_time_resources
        n = len(lamb)

        lamb_e = params.data_lambda
        mu_e = params.data_mu
        b_min = params.data_resources_min
        v = params.beam_capacity
        a = [lamb[k] / mu[k] for k in range(n)]

        # pi_e^(1)
        a_new = a.copy()
        a_new.append(lamb_e / mu_e)
        b_new = list(b)
        b_new.append(b_min)
        p = appr.SolverOnlyRealTime.solve_approx(a_new, b_new, v)

        pi_e_1 = sum(p[l] for l in range(v - b_min + 1, v + 1))
        pi_1_1 = sum(p[l] for l in range(v - b[0] + 1, v + 1))
        pi_2_1 = sum(p[l] for l in range(v - b[1] + 1, v + 1))

        max_pi = max([pi_e_1, pi_1_1, pi_2_1])

        solver = appr.SolveApprox(params)
        pi_e_1_, y_2, W_2, b_e_2 = solver.solve()

        assert np.isclose(pi_e_1, pi_e_1_)

        return max_pi, W_2


# class ExperimentApproximationAnalysis:
#     """Experiment to validate the analytical model against the simulation."""

#     save_path = "results/approx_analysis"

#     def __init__(self, max_error, max_iter, num_exec=8):
#         self.max_error = max_error
#         self.max_iter = max_iter
#         self.num_exec = num_exec

#         (Path.cwd() / self.save_path).mkdir(parents=True, exist_ok=True)

#     def run(self, type_="ana"):
#         assert type_ in ["ana", "approx"]

#         task = None
#         if type_ == "ana":
#             task = self.task_analytical
#         elif type_ == "approx":
#             task = self.task_approximation

#         assert task is not None

#         futures = {}
#         with ProcessPoolExecutor(self.num_exec) as executor:
#             for v in range(10, test_params_1.beam_capacity + 1):
#                 for b_max in [4, 8, v]:
#                     run_params = ParametersSet(
#                         real_time_flows=test_params_1.real_time_flows,
#                         real_time_lambdas=test_params_1.real_time_lambdas,
#                         real_time_mus=test_params_1.real_time_mus,
#                         real_time_resources=test_params_1.real_time_resources,
#                         data_resources_min=test_params_1.data_resources_min,
#                         data_resources_max=b_max,
#                         data_lambda=test_params_1.data_lambda,
#                         data_mu=test_params_1.data_mu,
#                         beam_capacity=v,
#                     )
#                     futures[executor.submit(task, run_params)] = run_params

#             for future in as_completed(futures):
#                 res_params = futures[future]
#                 res = future.result()
#                 self.save_result(res_params, res, type_)

#     def save_result(self, params: ParametersSet, res: Metrics | tuple, type_):
#         filename = f"{type_}-v-{params.beam_capacity}-bmax-{params.data_resources_max}.csv"
#         full_path = Path.cwd() / self.save_path / filename

#         if type_ == "ana" and isinstance(res, Metrics):
#             res_str = (
#                 f"{params.beam_capacity},"
#                 f"{params.data_resources_max},"
#                 f"{res.data_request_rej_prob:.5f},"
#                 f"{res.mean_data_requests_in_service:.5f},"
#                 f"{res.mean_data_request_service_time:.5f},"
#                 f"{res.mean_resources_per_data_request:.5f}"
#             )
#         elif type_ == "approx" and isinstance(res, tuple):
#             res_str = (
#                 f"{params.beam_capacity}, {params.data_resources_max}, {res[0]:.5f}, {res[1]:.5f}, {res[2]:.5f}, {res[3]:.5f}"
#             )
#         else:
#             raise ValueError("Invalid type")

#         with open(full_path, "w", encoding="utf-8") as f:
#             f.write(res_str)

#     def task_analytical(self, params):
#         an_solver = ana.Solver(params, self.max_error, self.max_iter)
#         an_metrics, _, _ = an_solver.solve()

#         return an_metrics

#     def task_approximation(self, params):
#         solver = appr.SolveApprox(params)
#         pi_e_1, y_2, W_2, b_e_2 = solver.solve()

#         return pi_e_1, y_2, W_2, b_e_2


if __name__ == "__main__":
    import time

    exp = ExperimentValidation(
        max_error=1e-7, max_iter=35_000, warmup=200_000, events=3_000_000, num_exec=4
    )
    start = time.time()
    exp.run("sim")
    end = time.time()
    print("simulation:", end - start, "seconds")

    start = time.time()
    exp.run("ana")
    end = time.time()
    print("analytical:", end - start, "seconds")

    exp = ExperimentQoS(max_error=1e-8, max_iter=35_000, num_exec=4)
    start = time.time()
    exp.run()
    end = time.time()
    print("analytical:", end - start, "seconds")

    exp = ExperimentResourcePlanning(max_error=1e-8, max_iter=35_000, num_exec=8)
    exp.run("approx")
    exp.run("ana")
