import retro
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import cv2

# --- Hyperparameters and Constants ---
EPISODES = 1000
BATCH_SIZE = 32
GAMMA = 0.99
LEARNING_RATE = 1e-4
MEM_SIZE = 1000
REPLAY_START_SIZE = 1000
TARGET_UPDATE = 1000
MAX_STEPS = 10000

# --- NES Button Mapping and Action Set ---
BUTTONS = ['B', None, 'SELECT', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'A']
SIMPLE_MOVEMENT = [
    [],  # No action
    ['RIGHT'],
    ['RIGHT', 'A'],
    ['RIGHT', 'B'],
    ['RIGHT', 'A', 'B'],
    ['A'],
    ['LEFT'],
    ['LEFT', 'A'],
    ['LEFT', 'B'],
    ['LEFT', 'A', 'B'],
]

# --- Converts a discrete action index to NES button array ---
def action_to_array(action_index):
    arr = [0] * len(BUTTONS)
    for button in SIMPLE_MOVEMENT[action_index]:
        if button and button in BUTTONS:
            arr[BUTTONS.index(button)] = 1
    return np.array(arr)

# --- Preprocesses raw RGB observation: grayscale, resize, normalize ---
def preprocess(obs):
    obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    obs = cv2.resize(obs, (84, 84))
    obs = obs / 255.0
    return obs

# --- Stacks frames to provide temporal context for the agent ---
def stack_frames(stacked_frames, state, is_new_episode):
    frame = preprocess(state)
    if is_new_episode:
        stacked_frames = [frame] * 4
    else:
        stacked_frames.append(frame)
        stacked_frames.pop(0)
    return np.stack(stacked_frames, axis=0), stacked_frames

# --- Deep Q-Network: Convolutional Neural Network for image input ---
class DQNCNN(nn.Module):
    def __init__(self, action_space_n):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(7*7*64, 512),
            nn.ReLU(),
            nn.Linear(512, action_space_n)
        )
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# --- Experience Replay Buffer for DQN ---
class ReplayBuffer:
    def __init__(self, size):
        self.buffer = deque(maxlen=size)
    def add(self, *args):
        self.buffer.append(tuple(args))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        actions = np.array(actions)
        if len(actions.shape) > 1:
            actions = actions.squeeze()
        return np.stack(states), actions, rewards, np.stack(next_states), dones
    def __len__(self):
        return len(self.buffer)

# --- Initialize Environment and Networks ---
env = retro.make(game='SuperMarioBros-Nes')
action_space_n = len(SIMPLE_MOVEMENT)
policy_net = DQNCNN(action_space_n)
target_net = DQNCNN(action_space_n)
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
memory = ReplayBuffer(MEM_SIZE)

# --- Epsilon-Greedy Action Selection for Exploration/Exploitation ---
def select_action(state, eps):
    if random.random() < eps:
        return random.randrange(action_space_n)
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        q = policy_net(state)
    return q.argmax().item()

# --- Main Training Loop ---
eps = 1.0
eps_min = 0.1
eps_decay = 0.995
for episode in range(EPISODES):
    state = env.reset()
    if isinstance(state, tuple):
        state = state[0]
    stacked_frames = None
    state, stacked_frames = stack_frames(None, state, True)
    total_reward = 0
    for t in range(MAX_STEPS):
        action = select_action(state, eps)
        action_arr = action_to_array(int(action))
        step_result = env.step(action_arr)
        
        # --- Handle Gymnasium/Gym API differences ---
        if len(step_result) == 5:
            next_state, reward, terminated, truncated, info = step_result
        else:
            next_state, reward, done, info = step_result
            terminated, truncated = done, False
        if isinstance(next_state, tuple):
            next_state = next_state[0]
        done = terminated or truncated
        next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
        memory.add(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        # --- DQN Learning Step ---
        if len(memory) > REPLAY_START_SIZE:
            states, actions, rewards, next_states, dones = memory.sample(BATCH_SIZE)
            states = np.array(states)
            actions = np.array(actions)
            rewards = np.array(rewards)
            next_states = np.array(next_states)
            dones = np.array(dones)
            states = torch.tensor(states, dtype=torch.float32)
            actions = torch.tensor(actions, dtype=torch.long)
            rewards = torch.tensor(rewards, dtype=torch.float32)
            next_states = torch.tensor(next_states, dtype=torch.float32)
            dones = torch.tensor(dones, dtype=torch.float32)

            q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            next_q_values = target_net(next_states).max(1)[0]
            expected_q = rewards + GAMMA * next_q_values * (1 - dones)
            loss = nn.MSELoss()(q_values, expected_q.detach())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # --- Update Target Network ---
            if t % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())

        if done:
            break

    eps = max(eps_min, eps * eps_decay)
    print(f"Episode {episode+1}, Total Reward: {total_reward}, Epsilon: {eps:.3f}")
    
    # --- Periodically Save Model ---
    if (episode + 1) % 100 == 0:
        torch.save(policy_net.state_dict(), "src/Super_Mario_Bros_RL_Project/mario_dqn.pth")

env.close()