from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from utils import ParametersSet, Metrics
from typing import Callable
import itertools
import analytical as ana
import logging
logging.basicConfig(filename="analytical.log", filemode="w", level=logging.WARNING, encoding="utf-8")

EXP_1_COMPARE_PARAMS = ParametersSet(
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
    beam_capacity=50,
)
V_MIN = 30
V_MAX = 90


class ExperimentCompare:
    """Experiment to validate the analytical model against the simulation."""

    save_path = "results/compare_retries"
    retry_intensity_range = [1]
    retry_probability_range = [0.05, 0.5, 0.95]
    queue_intensity_range = [1]
    beam_capacity_range = range(V_MIN, V_MAX + 1)

    def __init__(self, max_error, max_iter, num_exec=8):
        self.max_error = max_error
        self.max_iter = max_iter
        self.num_exec = num_exec

        (Path.cwd() / self.save_path).mkdir(parents=True, exist_ok=True)

    def run(self, task: Callable[[ParametersSet], Metrics | None]):

        futures = {}
        with ProcessPoolExecutor(self.num_exec) as executor:
            for v, nu, sigma, H in itertools.product(
                self.beam_capacity_range,
                self.retry_intensity_range,
                self.queue_intensity_range,
                self.retry_probability_range,
            ):
                run_params = ParametersSet(
                    real_time_flows=EXP_1_COMPARE_PARAMS.real_time_flows,
                    real_time_lambdas=EXP_1_COMPARE_PARAMS.real_time_lambdas,
                    real_time_mus=EXP_1_COMPARE_PARAMS.real_time_mus,
                    real_time_resources=EXP_1_COMPARE_PARAMS.real_time_resources,
                    data_resources_min=EXP_1_COMPARE_PARAMS.data_resources_min,
                    data_lambda=EXP_1_COMPARE_PARAMS.data_lambda,
                    data_mu=EXP_1_COMPARE_PARAMS.data_mu,
                    beam_capacity=v,
                    queue_intensity=sigma,
                    retry_intensity=nu,
                    retry_probability=H,
                )
                futures[executor.submit(task, run_params)] = run_params

            for future in as_completed(futures):
                res_params = futures[future]
                res = future.result()
                self.save_result(res_params, res)

    def save_result(self, params: ParametersSet, res: Metrics | None):
        if res is None:
            return
        filename = f"v-{params.beam_capacity}-sigma-{params.queue_intensity}-nu-{params.retry_intensity}-H-{params.retry_probability}.csv"

        full_path = Path.cwd() / self.save_path / filename

        with open(full_path, "w", encoding="utf-8") as f:
            f.write(
                f"{params.beam_capacity},"
                f"{params.queue_intensity},"
                f"{params.retry_intensity},"
                f"{params.retry_probability},"
                f"{res.beam_utilization:.7f},"
                f"{res.pi_k[0]:.7f},"
                f"{res.pi_k[1]:.7f},"
                f"{res.y_k[0]:.7f},"
                f"{res.y_k[1]:.7f},"
                f"{res.m_k[0]:.7f},"
                f"{res.m_k[1]:.7f},"
                f"{res.y_r:.7f},"
                f"{res.y_q:.7f},"
                f"{res.y_d:.7f},"
                f"{res.y_e:.7f},"
                f"{res.m_e:.7f},"
                f"{res.b_e:.7f},"
                f"{res.Lambda_e:.7f},"
                f"{res.Lambda_e_b:.7f},"
                f"{res.pi_e_0:.7f},"
                f"{res.pi_e_a:.7f},"
                f"{res.pi_e_r:.7f},"
                f"{res.W_sess:.7f},"
                f"{res.A:.7f},"
            )

    def task_analytical(self, params: ParametersSet) -> Metrics | None:
        an_solver = ana.Solver(params, self.max_error, self.max_iter)
        is_valid = an_solver.solve_with_r_max(r_min=20, step=10, max_attempts=10)
        if not is_valid:
            print(f"Solution is not valid for parameters: {params}")
            return None
        return an_solver.calculate_metrics()


def main():
    exp = ExperimentCompare(max_error=1e-7, max_iter=3000, num_exec=4)
    exp.run(exp.task_analytical)

if __name__ == "__main__":
    main()