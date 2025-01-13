from __future__ import annotations

from utils import ParametersSet, Metrics
import itertools
import time
import numpy as np

# all three eqs are ok
test_params = ParametersSet(2, [15, 3], [1, 1], [1, 5], 1, 50, 1, 1, 50)

# ok
test_params = ParametersSet(1, [3], [1], [5], 1, 50, 1, 1, 50)


test_params = ParametersSet(1, [0.042], [1 / 300], [3], 1, 5, 4.2, 1 / 16, 50)


def get_possible_states(parameters: ParametersSet):
    beam_capacity = parameters.beam_capacity
    b_min = parameters.data_resources_min

    max_data_flows = beam_capacity // parameters.data_resources_min
    data_flow_states = np.arange(max_data_flows + 1, dtype=int)

    real_time_flows_states = [
        np.arange(beam_capacity // b + 1, dtype=int) for b in parameters.real_time_resources
    ]

    # product of possible values
    states = itertools.product(*real_time_flows_states, data_flow_states)
    # filter by capacity
    states_lst = [
        s
        for s in states
        if np.dot(s[:-1], parameters.real_time_resources) + s[-1] * b_min <= beam_capacity
    ]

    return states_lst


class State:
    def __init__(self, i_vec: list[int], d: int):
        self.i_vec = i_vec
        self.d = d

    def ik_(self, k, delta=1):
        i_vec = self.i_vec.copy()
        i_vec[k] += delta

        return State(i_vec, self.d)

    def d_(self, delta=1):
        return State(self.i_vec, self.d + delta)


class Solver:
    def __init__(self, params: ParametersSet, possible_states, max_eps: float, max_iter: int):
        self.params: ParametersSet = params
        self.possible_states: list[tuple] = possible_states
        self.max_eps: float = max_eps
        self.max_iter: int = max_iter

        self.states: dict[tuple, float] = {state: 1.0 for state in possible_states}

    def solve(self):
        n = self.params.real_time_flows
        lamb = self.params.real_time_lambdas
        mu = self.params.real_time_mus
        b = self.params.real_time_resources

        b_min = self.params.data_resources_min
        b_max = self.params.data_resources_max
        lambda_e = self.params.data_lambda
        mu_e = self.params.data_mu

        v = self.params.beam_capacity

        error = 1000
        iteration = 0
        while not (error < self.max_eps or iteration > self.max_iter):
            iteration += 1
            error = 0

            for state in self.states.keys():
                *i_vec, d = state

                state_ = State(i_vec, d)

                l = np.dot(i_vec, b)
                l_tot = l + d * b_min
                d_lim = (v - l) // b_max

                real_time_arrival_d = sum(lamb[k] * (l_tot + b[k] <= v) for k in range(n))
                real_time_serv_d = sum(i_vec[k] * mu[k] * (i_vec[k] > 0) for k in range(n))

                data_arr_d = lambda_e * (l_tot + b_min <= v)
                data_serv_d = mu_e * (d > 0) * ((v - l) * (d > d_lim) + d * b_max * (d <= d_lim))

                denr = real_time_arrival_d + real_time_serv_d + data_arr_d + data_serv_d

                real_time_arr_n = sum(
                    self.prob(state_.ik_(k, -1)) * lamb[k] * (i_vec[k] > 0) for k in range(n)
                )
                real_time_serv_n = sum(
                    self.prob(state_.ik_(k, +1)) * (i_vec[k] + 1) * mu[k] * (l_tot + b[k] <= v)
                    for k in range(n)
                )
                data_arr_n = self.prob(state_.d_(-1)) * lambda_e * (l_tot <= v and d > 0)
                data_serv_n = (
                    self.prob(state_.d_(+1))
                    * (l_tot + b_min <= v and d + 1 > 0)
                    * mu_e
                    * ((v - l) * (d + 1 > d_lim) + (d + 1) * b_max * (d + 1 <= d_lim))
                )

                numr = real_time_arr_n + real_time_serv_n + data_arr_n + data_serv_n
                error += (self.states[state] - numr / denr) / self.states[state]
                self.states[state] = numr / denr

        return self.states, iteration

    def prob(self, state: State):
        i_vec, d = state.i_vec, state.d

        state_ = (*i_vec, d)

        return self.states[state_] if (state_ in self.states) else 0

    def calculate_metrics(self, states: dict[tuple, float]) -> Metrics:
        metrics = Metrics(
            rt_request_rej_prob=[0] * self.params.real_time_flows,
            mean_rt_requests_in_service=[0] * self.params.real_time_flows,
            mean_resources_per_rt_flow=[0] * self.params.real_time_flows,
        )

        for state in states.keys():
            *i_vec, d = state
            b_min = self.params.data_resources_min

            v = self.params.beam_capacity
            l = np.dot(i_vec, self.params.real_time_resources) + b_min * d

            for k in range(self.params.real_time_flows):
                b_k = self.params.real_time_resources[k]
                metrics.mean_rt_requests_in_service[k] += i_vec[k] * states[state]

                if l + b_k > v:
                    metrics.rt_request_rej_prob[k] += states[state]

            if l + b_min > v:
                metrics.data_request_rej_prob += states[state]

            b_max = self.params.data_resources_max
            d_limit = (v - np.dot(i_vec, self.params.real_time_resources)) // b_max
            data_res = (v - np.dot(i_vec, self.params.real_time_resources)) * (
                d > d_limit
            ) + d * b_max * (d <= d_limit) * (d > 0)
            metrics.mean_resources_per_data_flow += data_res * states[state]
            metrics.mean_data_requests_in_service += d * states[state]

        for k in range(self.params.real_time_flows):
            metrics.mean_resources_per_rt_flow[k] = (
                metrics.mean_rt_requests_in_service[k] * self.params.real_time_resources[k]
            )

        metrics.mean_resources_per_data_request = (
            metrics.mean_resources_per_data_flow / metrics.mean_data_requests_in_service
        )

        metrics.mean_data_request_service_time = metrics.mean_data_requests_in_service / (
            self.params.data_lambda * 1 * (1 - metrics.data_request_rej_prob)
        )
        return metrics


possible_states = get_possible_states(test_params)
print(len(possible_states))

solver = Solver(test_params, possible_states, 1e-7, 35000)
start_time = time.time()
solution_states, it = solver.solve()
end_time = time.time()
print(f"{it=}, time = {end_time - start_time}")
# normalize

norm = sum(solution_states.values())

for key, value in solution_states.items():
    solution_states[key] = value / norm

print("sum probs", sum(solution_states.values()))


metrics = solver.calculate_metrics(solution_states)

m_e = metrics.mean_resources_per_data_flow
pi_e = metrics.data_request_rej_prob

e1 = test_params.data_lambda * 1
e2 = test_params.data_lambda * 1 * pi_e + m_e * test_params.data_mu

print(f"{e1:.5f}", f"{e2:.5f}", np.isclose(e1, e2))


lambda_1 = test_params.real_time_lambdas[0]
# lambda_2 = test_params.real_time_lambdas[1]

mu_1 = test_params.real_time_mus[0]
# mu_2 = test_params.real_time_mus[1]

b_1 = test_params.real_time_resources[0]
# b_2 = test_params.real_time_resources[1]

pi_1 = metrics.rt_request_rej_prob[0]
# pi_2 = metrics.rt_request_rej_prob[1]

m_1 = metrics.mean_resources_per_rt_flow[0]
# m_2 = metrics.mean_resources_per_rt_flow[1]

e1 = lambda_1 * b_1
e2 = lambda_1 * b_1 * pi_1 + m_1 * mu_1
print(f"{e1:.5f}", f"{e2:.5f}", np.isclose(e1, e2))

# e1 = lambda_2 * b_2
# e2 = lambda_2 * b_2 * pi_2 + m_2 * mu_2
print(f"{e1:.5f}", f"{e2:.5f}", np.isclose(e1, e2))

pi_1 = metrics.rt_request_rej_prob[0]
pi_2 = 0  # metrics.rt_request_rej_prob[1]
pi_e = metrics.data_request_rej_prob

print(f"{pi_1=:.5f}, {pi_2=:.5f}, {pi_e=:.5f}")


print(metrics)
