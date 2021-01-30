import gym
import numpy as np

env = gym.make("CartPole-v1")
env.reset()

LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 20000

epsilon = 1  # not a constant, going to be decayed
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2
epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)

bins = [
    np.linspace(-4.8, 4.8, 20),
    np.linspace(-4, 4, 20),
    np.linspace(.418, -.418, 20),
    np.linspace(-4, 4, 20)
]

q_table = np.random.uniform(low=-2, high=0, size=([20] * len(env.observation_space.high) + [env.action_space.n]))


def get_discrete_state(state):
    state_indexes = []
    for i in range(len(env.observation_space.high)):
        # digitize returns the indexes of the bins that the data fits into
        state_indexes.append(np.digitize(state[i], bins[i]) - 1)
    return tuple(state_indexes)


SHOW_EVERY = 2000


for run in range(EPISODES):
    discrete_state = get_discrete_state(env.reset())
    done = False
    movements = 0

    while not done:
        if run % SHOW_EVERY == 0:
            env.render()

        movements += 1

        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state])
        else:
            action = np.random.randint(0, env.action_space.n)

        new_state, reward, done, _ = env.step(action)
        new_discrete_state = get_discrete_state(new_state)

        if done and movements < 200:
            reward = -375

        max_future_q = np.max(q_table[new_discrete_state])
        current_q = q_table[discrete_state + (action,)]
        new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
        q_table[discrete_state + (action,)] = new_q

        discrete_state = new_discrete_state

    if END_EPSILON_DECAYING >= run >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value

env.close()
