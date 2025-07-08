import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque
import tensorflow as tf
from tensorflow.keras import models, layers
import asyncio
import platform

# 환경 설정: 5x5 그리드, 시작점(0,0), 배송지(4,4)
class LogisticsEnv:
    def __init__(self, grid_size=5):
        self.grid_size = grid_size
        self.state_size = grid_size * grid_size
        self.action_size = 4  # 상, 하, 좌, 우
        self.start = (0, 0)
        self.goal = (grid_size-1, grid_size-1)
        self.state = self.start
        self.obstacles = [(1, 1), (2, 2), (3, 3)]  # 장애물 위치
        self.max_steps = 50
        self.current_step = 0

    def reset(self):
        self.state = self.start
        self.current_step = 0
        return self._get_state_idx()

    def _get_state_idx(self):
        return self.state[0] * self.grid_size + self.state[1]

    def step(self, action):
        self.current_step += 1
        x, y = self.state
        if action == 0:  # 상
            x = max(0, x - 1)
        elif action == 1:  # 하
            x = min(self.grid_size - 1, x + 1)
        elif action == 2:  # 좌
            y = max(0, y - 1)
        elif action == 3:  # 우
            y = min(self.grid_size - 1, y + 1)

        next_state = (x, y)
        reward = -1  # 기본 이동 비용
        done = False

        if next_state in self.obstacles:
            reward = -10
            next_state = self.state  # 장애물에 부딪히면 이동하지 않음
        elif next_state == self.goal:
            reward = 100
            done = True
        elif self.current_step >= self.max_steps:
            done = True

        self.state = next_state
        return self._get_state_idx(), reward, done

# DQN 모델
class DQN:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # 할인율
        self.epsilon = 1.0  # 탐험률
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = self._build_model()

    def _build_model(self):
        model = models.Sequential()
        model.add(layers.Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(layers.Dense(24, activation='relu'))
        model.add(layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = np.zeros(self.state_size)
        state[state] = 1
        act_values = self.model.predict(state.reshape(1, -1), verbose=0)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state_vec = np.zeros(self.state_size)
            state_vec[state] = 1
            next_state_vec = np.zeros(self.state_size)
            next_state_vec[next_state] = 1
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state_vec.reshape(1, -1), verbose=0)[0])
            target_f = self.model.predict(state_vec.reshape(1, -1), verbose=0)
            target_f[0][action] = target
            self.model.fit(state_vec.reshape(1, -1), target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# 시각화 함수
def visualize_path(env, agent, episode):
    path = [env.start]
    state = env.reset()
    done = False
    steps = 0
    while not done and steps < env.max_steps:
        action = agent.act(state)
        next_state, _, done = env.step(action)
        x = next_state // env.grid_size
        y = next_state % env.grid_size
        path.append((x, y))
        state = next_state
        steps += 1

    # 그리드 시각화
    grid = np.zeros((env.grid_size, env.grid_size))
    for obs in env.obstacles:
        grid[obs] = -1
    grid[env.goal] = 2
    for x, y in path:
        if grid[x, y] == 0:
            grid[x, y] = 1

    plt.figure(figsize=(6, 6))
    plt.imshow(grid, cmap='hot', interpolation='nearest')
    plt.plot([y for _, y in path], [x for x, _ in path], 'b-', marker='o')
    plt.title(f'Episode {episode} Path')
    plt.colorbar(label='0: Empty, 1: Path, 2: Goal, -1: Obstacle')
    plt.grid(True)
    plt.show()

# 메인 학습 루프
async def main():
    env = LogisticsEnv()
    agent = DQN(env.state_size, env.action_size)
    episodes = 100
    batch_size = 32
    rewards = []

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        for _ in range(env.max_steps):
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if done:
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        rewards A'st.append(total_reward)
        if episode % 10 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")
            visualize_path(env, agent, episode)
        await asyncio.sleep(0.01)  # 비동기 실행을 위한 짧은 대기

    # 학습 결과 시각화
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.title('Training Rewards Over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    plt.show()

# Pyodide 호환 실행
if platform.system() == "Emscripten":
    asyncio.ensure_future(main())
else:
    if __name__ == "__main__":
        asyncio.run(main())
