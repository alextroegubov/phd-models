from utils import ParametersSet
import itertools

import numpy as np

test_params = ParametersSet(2, [15, 3], [1, 1], [1, 5], 1, 50, 1, 1, 50)

# test_params_satelite = ParametersSet(
#     real_time_flows=2,
#     real_time_lambdas=[2.5 * 1 / 3, 1 / 15],
#     real_time_mus=[1 / 3, 1 / 15],
#     real_time_resources=[1, 4],
#     data_resources_min=2,
#     data_resources_max=8,
#     data_lambda=1 / 20,
#     data_mu=1 / 20,
#     beam_capacity=10,
# )

test_params = test_params


def get_possible_states(parameters: ParametersSet):
    beam_capacity = parameters.beam_capacity
    b_min = parameters.data_resources_min

    max_data_flows = beam_capacity / parameters.data_resources_min
    data_flow_states = np.arange(max_data_flows + 1, dtype=int)

    real_time_flows_states = [
        np.arange(int(beam_capacity / b) + 1, dtype=int) for b in parameters.real_time_resources
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

    def change_ik(self, k, delta=1):
        i_vec = self.i_vec.copy()
        i_vec[k] += delta

        return State(i_vec, self.d)

    def change_d(self, delta=1):
        return State(self.i_vec, self.d + delta)


class Solver:
    def __init__(self, params: ParametersSet, possible_states, epsilon: float):
        self.params: ParametersSet = params
        self.possible_states: list[tuple] = possible_states
        self.epsilon: float = epsilon

        self.prev_states_dct: dict[tuple, float] = {state: 1.0 / 1000 for state in possible_states}
        self.next_states_dct: dict[tuple, float] = {}

    def calculate_error(self):
        norm_delta = np.linalg.norm(
            [
                self.next_states_dct[s] - self.prev_states_dct[s]
                for s in self.next_states_dct.keys()
            ],
            ord=1,
        )
        norm_vect = np.linalg.norm(list(self.prev_states_dct.values()), ord=1)
        error = norm_delta / norm_vect
        return error

    def solve(self):
        n = self.params.real_time_flows
        lambda_ = self.params.real_time_lambdas
        mu = self.params.real_time_mus
        b = self.params.real_time_resources

        b_min = self.params.data_resources_min
        b_max = self.params.data_resources_max
        lambda_e = self.params.data_lambda
        mu_e = self.params.data_mu

        v = self.params.beam_capacity

        I = lambda x: 1.0 if x else 0.0
        current_error = 1000
        iteration = 0

        while current_error > self.epsilon:
            iteration += 1
            for state in self.prev_states_dct.keys():
                *i_vec, d = state

                state_ = State(i_vec, d)

                l = sum(i_k * b_k for (i_k, b_k) in zip(i_vec, b))
                d_limit = (v - l) // b_max

                if l + d * b_min > v:
                    raise ValueError(f"{l=}, {d=}, {b_min=}")

                real_time_arrival_d = sum(
                    lambda_[k] * I(l + d * b_min + b[k] <= v) for k in range(n)
                )
                real_time_serv_d = sum(i_vec[k] * mu[k] * I(i_vec[k] >= 1) for k in range(n))
                data_arr_d = lambda_e * I(l + d * b_min + b_min <= v)
                data_serv_d = (v - l) * mu_e * I(d > d_limit) + d * b_max * mu_e * I(
                    1 <= d <= d_limit
                )

                denominator = real_time_arrival_d + real_time_serv_d + data_arr_d + data_serv_d

                real_time_arr_n = sum(
                    self.prob(state_.change_ik(k, delta=-1)) * lambda_[k] * I(i_vec[k] >= 1)
                    for k in range(n)
                )
                real_time_serv_n = sum(
                    self.prob(state_.change_ik(k, delta=+1))
                    * (i_vec[k] + 1)
                    * mu[k]
                    * I(l + d * b_min + b[k] <= v)
                    for k in range(n)
                )
                data_arr_n = (
                    self.prob(state_.change_d(-1)) * lambda_e * I(l + d * b_min <= v and d >= 1)
                )
                data_serv_n = (
                    self.prob(state_.change_d(+1))
                    * I(l + d * b_min + b_min <= v)
                    * (
                        (v - l) * mu_e * I(d + 1 > d_limit)
                        + (d + 1) * b_max * mu_e * I(d + 1 <= d_limit)
                    )
                )

                numerator = real_time_arr_n + real_time_serv_n + data_arr_n + data_serv_n
                self.next_states_dct[state] = numerator / denominator

            assert len(self.next_states_dct) == len(self.prev_states_dct)
            current_error = self.calculate_error()
            self.prev_states_dct = {key: value for (key, value) in self.next_states_dct.items()}

        return self.prev_states_dct, iteration

    def prob(self, state: State):
        i_vec, d = state.i_vec, state.d

        state_ = (*i_vec, d)

        if state_ in self.next_states_dct:
            return self.next_states_dct[state_]
        elif state_ in self.prev_states_dct:
            return self.prev_states_dct[state_]
        else:
            return 0


possible_states = get_possible_states(test_params)
print(len(possible_states))

solver = Solver(test_params, possible_states, 1e-7)
solution_states, it = solver.solve()
print(f"{it=}")
# normalize

norm = sum(solution_states.values())

for key, value in solution_states.items():
    solution_states[key] = value / norm

print(sum(solution_states.values()))


pi_1 = 0
pi_2 = 0
pi_e = 0

for state in solution_states.keys():
    v = test_params.beam_capacity
    b_min = test_params.data_resources_min
    i_1, i_2, d = state
    b_1, b_2 = test_params.real_time_resources

    l = i_1 * b_1 + i_2 * b_2

    if l + d * b_min + b_1 > v:
        pi_1 += solution_states[state]
    if l + d * b_min + b_2 > v:
        pi_2 += solution_states[state]
    if l + d * b_min + b_min > v:
        pi_e += solution_states[state]

print(f"{pi_1=:.4f}, {pi_2=:.4f}, {pi_e=:.4f}")


a = (
    np.isclose(pi_1, 0.005795995193366042)
    and np.isclose(pi_2, 0.03901275178165448)
    and np.isclose(pi_e, 0.005795995193366042)
)
print(a)
