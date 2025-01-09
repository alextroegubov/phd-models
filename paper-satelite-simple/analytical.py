from utils import ParametersSet
import itertools

import numpy as np

############ GOOD ###############
# def solve(possible_states, parameters: ParametersSet):

#     n = parameters.real_time_flows
#     lambda_1, lambda_2 = parameters.real_time_lambdas
#     mu_1, mu_2 = parameters.real_time_mus
#     b1, b2 = parameters.real_time_resources
#     ind = lambda x: 1.0 if x else 0.0

#     b_min = parameters.data_resources_min
#     b_max = parameters.data_resources_max
#     lambda_e = parameters.data_lambda
#     mu_e = parameters.data_mu

#     v = parameters.beam_capacity

#     prev_states_dict = {state: 1.0 / 5236 for state in possible_states}

#     epsilon = 1

#     it = 0

#     while epsilon > 1e-7:
#         it += 1
#         next_states_dict = {}
#         for state in prev_states_dict.keys():

#             i1, i2, d = state
#             l = i1 * b1 + i2 * b2

#             real_time_arr_serv_denom = lambda_1 * ind(l + d + b1 <= v) + lambda_2 * ind(l + d + b2 <= v)
#             real_time_arr_serv_denom += i1 * mu_1 * ind(i1 > 0) + i2 * mu_2 * ind(i2 > 0)
#             data_arr_serv_denom = lambda_e * ind(l + d + 1 <= v) + (v - l) * mu_e * ind(d > 0)

#             # real_time_arr_serv_num
#             real_time_arr_serv_num = 0

#             p_i1_minus = get_prob_from_states(d, (i1 - 1, i2), prev_states_dict, next_states_dict)
#             p_i1_plus = get_prob_from_states(d, (i1 + 1, i2), prev_states_dict, next_states_dict)

#             p_i2_minus = get_prob_from_states(d, (i1, i2 - 1), prev_states_dict, next_states_dict)
#             p_i2_plus = get_prob_from_states(d, (i1, i2 + 1), prev_states_dict, next_states_dict)

#             real_time_arr_serv_num = (
#                 p_i1_minus * lambda_1 * ind(i1 > 0) +
#                 p_i1_plus * (i1 + 1) * mu_1 * ind(l + d + b1 <= v) +
#                 p_i2_minus * lambda_2 * ind(i2 > 0) +
#                 p_i2_plus * (i2 + 1) * mu_2 * ind(l + d + b2 <= v)
#             )

#             data_arr_serv_num = (
#                 get_prob_from_states(d - 1, (i1, i2), prev_states_dict, next_states_dict) * lambda_e * ind(d > 0) +
#                 get_prob_from_states(d + 1, (i1, i2), prev_states_dict, next_states_dict) * (v - l) * mu_e * ind(l + d + 1 <= v)
#             )
#             new_P_state = (data_arr_serv_num + real_time_arr_serv_num) / (data_arr_serv_denom + real_time_arr_serv_denom)

#             next_states_dict[state] = new_P_state

#         assert len(next_states_dict) == len(prev_states_dict)

#         # residual = sum((next_states_dict[s] - prev_states_dict[s])**2 for s in next_states_dict.keys())
#         # norm = sum(map(lambda x: x**2, prev_states_dict.values()))

#         residual = sum(abs(next_states_dict[s] - prev_states_dict[s]) for s in next_states_dict.keys())
#         norm = sum(map(lambda x: abs(x), prev_states_dict.values()))

#         epsilon = residual / norm
#         prev_states_dict = {key: value for (key, value) in next_states_dict.items()}

#     print(f"{it=}")

#     return prev_states_dict

################# 

# test_params = ParametersSet(
#     2,
#     [15, 3],
#     [1, 1],
#     [1, 5],
#     1,
#     50,
#     1,
#     1,
#     50
# )

test_params_satelite = ParametersSet(
    2,
    [7.5, 15],
    [3, 15],
    [1, 4],
    2,
    8,
    5,
    5,
    24
)
test_params = test_params_satelite

# test_params = ParametersSet(
#     2,
#     [1],
#     [1],
#     [3],
#     1,
#     5,
#     1,
#     1,
#     5
# )


def get_possible_states(parameters: ParametersSet):
    beam_capacity = parameters.beam_capacity

    max_data_flows = beam_capacity / parameters.data_resources_min
    data_flow_states = np.arange(max_data_flows + 1, dtype=int)

    real_time_flows_states = [np.arange(int(beam_capacity / b) + 1, dtype=int) 
                              for b in parameters.real_time_resources]

    # product of possible values
    states = itertools.product(*real_time_flows_states, data_flow_states)
    # filter by capacity
    states_lst = [
        s for s in states 
        if np.dot(s[:-1], parameters.real_time_resources) + s[-1] <= beam_capacity
    ]

    return states_lst


def solve_single(possible_states, parameters: ParametersSet):

    lambda_1 = parameters.real_time_lambdas[0]
    mu_1 = parameters.real_time_mus[0]
    b1 = parameters.real_time_resources[0]

    lambda_e = parameters.data_lambda
    mu_e = parameters.data_mu

    v = parameters.beam_capacity

    prev_states_dict = {state: 1.0 / 20 for state in possible_states}
    ind = lambda x: 1.0 if x else 0.0
    epsilon = 1

    it = 0

    while epsilon > 1e-6:
        it += 1
        next_states_dict = {}
        for state in prev_states_dict.keys():

            print(f"{state=}")
            i, d = state
            denominator = (
                lambda_1 * ind(i * b1 + b1 + d <= v) + i * mu_1 * ind(i > 0) +
                lambda_e * ind(i * b1 + d + 1 <= v) + (v - i * b1) * mu_e * ind(d > 0)
            )
            l = i * b1

            numerator = (
                get_prob_from_states(d, i - 1, prev_states_dict, next_states_dict) * lambda_1 * ind((i > 0) and ((i - 1) * b1 + b1 + d <= v)) +
                get_prob_from_states(d, i + 1, prev_states_dict, next_states_dict) * (i + 1) * mu_1 * ind(i * b1 + b1 + d <= v) +
                get_prob_from_states(d - 1, i, prev_states_dict, next_states_dict) * lambda_e * ind((d > 0) and (l + d <= v))+
                get_prob_from_states(d + 1, i, prev_states_dict, next_states_dict) * (v - i * b1) * mu_e * ind(i*b1 + d + 1 <= v)
            )

            new_P_state = numerator / denominator

            print(f"{numerator=}")
            print(f"{denominator=}")

            next_states_dict[state] = new_P_state

        assert len(next_states_dict) == len(prev_states_dict)

        # residual = sum((next_states_dict[s] - prev_states_dict[s])**2 for s in next_states_dict.keys())
        # norm = sum(map(lambda x: x**2, prev_states_dict.values()))

        residual = sum(abs(next_states_dict[s] - prev_states_dict[s]) for s in next_states_dict.keys())
        norm = sum(map(lambda x: abs(x), prev_states_dict.values()))

        epsilon = residual / norm
        prev_states_dict = {key: value for (key, value) in next_states_dict.items()}

    print(f"{it=}")

    return prev_states_dict


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
    

def get_prob_from_states(d, i_vec, prev_states_dict, next_states_dict):
    
    state = (*i_vec, d)

    if state in next_states_dict:
        return next_states_dict[state]
    elif state in prev_states_dict:
        return prev_states_dict[state]
    else:
        return 0

def iter_prob(state: State, prev_states_dict, next_states_dict):
    i_vec, d = state.i_vec, state.d
    
    state_ = (*i_vec, d)

    if state_ in next_states_dict:
        return next_states_dict[state_]
    elif state_ in prev_states_dict:
        return prev_states_dict[state_]
    else:
        return 0


class Solver:
    def __init__(self, params: ParametersSet, possible_states, epsilon: float):
        self.params: ParametersSet = params
        self.possible_states: list[tuple] = possible_states
        self.epsilon: float = epsilon

        self.prev_states_dct: dict[tuple, float] = {state: 1.0 / 1000 for state in possible_states}
        self.next_states_dct: dict[tuple, float] = {}

    def calculate_error(self):
        norm_delta = np.linalg.norm(
            [self.next_states_dct[s] - self.prev_states_dct[s] for s in self.next_states_dct.keys()],
            ord=1
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
                d_limit = int((v - l) / b_max)

                real_time_arrival_d = sum(lambda_[k] * I(l + d*b_min + b[k] <= v) for k in range(n))
                real_time_serv_d = sum(i_vec[k] * mu[k] * I(i_vec[k] >= 1) for k in range(n))
                data_arr_d = lambda_e * I(l + d*b_min + b_min <= v)
                data_serv_d = (v - l) * mu_e * I(d > d_limit) + d * b_max * mu_e * I(1 <= d <= d_limit)

                denominator = real_time_arrival_d + real_time_serv_d + data_arr_d + data_serv_d

                real_time_arr_n = sum(
                    self.prob(state_.change_ik(k, delta=-1)) * lambda_[k] * I(i_vec[k] >= 1)
                    for k in range(n)
                )
                real_time_serv_n = sum(
                    self.prob(state_.change_ik(k, delta=+1)) * (i_vec[k] + 1) * mu[k] * I(l + d*b_min + b[k] <= v)
                    for k in range(n)
                )
                data_arr_n = self.prob(state_.change_d(-1)) * lambda_e * I(l + d*b_min <= v and d >= 1)
                data_serv_n = self.prob(state_.change_d(+1)) * I(l + d*b_min + b_min <= v) * (
                    (v - l) * mu_e * I(d + 1 > d_limit) +
                    (d + 1) * b_max * mu_e * I(d + 1 <= d_limit)
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


# def solve(possible_states, parameters: ParametersSet):

#     n = parameters.real_time_flows
#     lambda_ = parameters.real_time_lambdas
#     mu = parameters.real_time_mus
#     b = parameters.real_time_resources

#     b_min = parameters.data_resources_min
#     b_max = parameters.data_resources_max
#     lambda_e = parameters.data_lambda
#     mu_e = parameters.data_mu

#     v = parameters.beam_capacity

#     I = lambda x: 1.0 if x else 0.0
#     EPSILON = 1e-7
#     error = 1
#     iteration = 0

#     # dct = {state: probability}
#     prev_s_dct = {state: 1.0 / 1000 for state in possible_states}

#     while error > EPSILON:
#         iteration += 1
#         next_s_dct = {}
#         for state in prev_s_dct.keys():

#             *i_vec, d = state

#             state_ = State(i_vec, d)

#             l = sum(i_k * b_k for (i_k, b_k) in zip(i_vec, b))

#             real_time_arrival_d = sum(lambda_k * I(l + d + b_k <= v) for (lambda_k, b_k) in zip(lambda_, b))
#             real_time_serv_d = sum(i_k * mu_k * I(i_k >= 1) for (i_k, mu_k) in zip(i_vec, mu))
#             data_arr_serv_d = lambda_e * I(l + d + 1 <= v) + (v - l) * mu_e * I(d >= 1)

#             denominator = real_time_arrival_d + real_time_serv_d + data_arr_serv_d


#             real_time_arr_n = sum(
#                 iter_prob(state_.change_ik(k, delta=-1), prev_s_dct, next_s_dct) * lambda_[k] * I(i_vec[k] > 0)
#                 for k in range(n)
#             )

#             real_time_serv_n = sum(
#                 iter_prob(state_.change_ik(k, delta=+1), prev_s_dct, next_s_dct) * (i_vec[k] + 1) * mu[k] * I(l + d + b[k] <= v)
#                 for k in range(n)
#             )

#             data_arr_n = iter_prob(state_.change_d(-1), prev_s_dct, next_s_dct) * lambda_e * I(d > 0)
#             data_serv_n = iter_prob(state_.change_d(+1), prev_s_dct, next_s_dct) * (v - l) * mu_e * I(l + d + 1 <= v)

#             numerator = real_time_arr_n + real_time_serv_n + data_arr_n + data_serv_n

#             next_s_dct[state] = numerator / denominator

#         assert len(next_s_dct) == len(prev_s_dct)

#         norm_delta = np.linalg.norm(
#             [next_s_dct[s] - prev_s_dct[s] for s in next_s_dct.keys()],
#             ord=1
#         )
#         norm_vect = np.linalg.norm(list(prev_s_dct.values()), ord=1)
#         error = norm_delta / norm_vect

#         prev_s_dct = {key: value for (key, value) in next_s_dct.items()}

#     print(f"{iteration=}")

#     return prev_s_dct


possible_states = get_possible_states(test_params)
print(len(possible_states))

solver= Solver(test_params, possible_states, 1e-7)
solution_states, it = solver.solve()
print(f"{it=}")
# normalize

norm = sum(solution_states.values())

for (key, value) in solution_states.items():
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

    if l + d*b_min + b_1 > v:
        pi_1 += solution_states[state]
    if l + d*b_min + b_2 > v: 
        pi_2 += solution_states[state]
    if l + d*b_min + b_min > v:
        pi_e += solution_states[state]

print(f"{pi_1=}, {pi_2=}, {pi_e=}")


a = np.isclose(pi_1, 0.005795995193366042) and np.isclose(pi_2, 0.03901275178165448) and np.isclose(pi_e, 0.005795995193366042)
print(a)




