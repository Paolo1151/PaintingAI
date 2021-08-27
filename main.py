from enum import Enum

import numpy as np
import time


class States(Enum):
    DIRTY = 0
    CLEAN = 1
    PAINTED = 2


class Actions(Enum):
    WASH = 0
    PAINT = 1
    EJECT = 2


trans_tensor = np.array([
                        [[0.1, 0.9, 0],
                         [0.1, 0.9, 0],
                         [0.1, 0.9, 0]],

                        [[1.0, 0, 0],
                         [0.1, 0.1, 0.8],
                         [0, 0, 1]],

                        [[1, 0, 0],
                         [0, 1, 0],
                         [0, 0, 1]]
                        ])

state_set = np.array([States.DIRTY, States.CLEAN, States.PAINTED])
action_set = np.array([Actions.WASH, Actions.PAINT, Actions.EJECT])


def generate_q_matrix_list(reward_func, h, q_matrix_list=[], prev_q_matrix=None, n=1):
    if n > h:
        return q_matrix_list
    else:
        reward_matrix = generate_reward_matrix(reward_func)
        if prev_q_matrix is not None:
            q_max = np.amax(prev_q_matrix, axis=1)
            sum_q_matrix = []
            for state in state_set:
                row = []
                for action in action_set:
                    row.append(np.dot(trans_tensor[action.value, state.value], q_max))
                sum_q_matrix.append(row)
            new_q_matrix = np.add(sum_q_matrix, reward_matrix)
        else:
            new_q_matrix = reward_matrix
        q_matrix_list.append(new_q_matrix)
        return generate_q_matrix_list(reward_func, h, q_matrix_list, new_q_matrix, n + 1)


def generate_reward_matrix(reward_func):
    reward_matrix = []
    for state in state_set:
        state_row = []
        for action in action_set:
            state_row.append(reward_func(state.value, action.value))
        reward_matrix.append(state_row)
    return np.array(reward_matrix)


def reward_function(state, action):
    if state == States.PAINTED.value and action == Actions.EJECT.value:
        return 10
    elif action == Actions.EJECT.value:
        return 0
    else:
        return -3


def calculate_best_actions(initial_state=States.DIRTY, lifetime=3):
    q_matrix_list = generate_q_matrix_list(reward_function, lifetime)
    state = initial_state
    log_report = {}
    best_policy = []
    total_reward = 0
    for turn in range(lifetime, 0, -1):
        action, reward = generate_best_action(state, q_matrix_list[lifetime - 1])
        action = Actions(action)
        old_state = state
        state = new_state(state, action)
        log_report[f'turn {lifetime - turn + 1}'] = [old_state.name, action.name, state.name]
        best_policy.append(action)
        total_reward += reward
    return log_report, best_policy, total_reward


def generate_best_action(state, q_matrix):
    return np.argmax(q_matrix[state.value]), np.max(q_matrix[state.value])


ejected_painted = 0


def new_state(state, action):
    if action == Actions.EJECT and state == States.PAINTED:
        global ejected_painted
        ejected_painted += 1
        return States.DIRTY
    else:
        choice = np.random.choice(state_set, 1, p=trans_tensor[action.value][state.value])
        return choice[0]


def main(lifetime):
    global ejected_painted
    log_report, best_policy, total_reward = calculate_best_actions(lifetime=lifetime)
    print(f"Generated AI with {lifetime} Turns: ")
    for turn, log in log_report.items():
        time.sleep(2)
        print(f"{turn}: {log[0]} -> action: {log[1]} -> {log[2]}")
    time.sleep(2)
    print(f'Total Reward: {total_reward}, Total_Painted: {ejected_painted}')


if __name__ == '__main__':
    main(int(input("How many turns should the AI Live? ")))

