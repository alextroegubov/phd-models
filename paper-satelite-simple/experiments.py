from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from utils import ParametersSet, Metrics
import simulation as sim
import analytical as ana
import approximation as appr

LAMBDA_2 = 2
V_MAX = 50
test_params_1 = ParametersSet(
    real_time_flows=2,
    real_time_lambdas=[2 * LAMBDA_2, LAMBDA_2],
    real_time_mus=[1, 1],
    real_time_resources=[1, 4],
    data_resources_min=2,
    data_resources_max=6,
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


if __name__ == "__main__":
    import time

    exp = ExperimentValidation(
        max_error=1e-8, max_iter=35_000, warmup=100_000, events=1_000_000, num_exec=8
    )
    start = time.time()
    exp.run("sim")
    end = time.time()
    print("simulation:", end - start, "seconds")

    start = time.time()
    exp.run('ana')
    end = time.time()
    print("analytical:", end - start, "seconds")