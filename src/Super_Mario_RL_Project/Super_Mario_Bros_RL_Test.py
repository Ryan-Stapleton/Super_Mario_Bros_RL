import retro
import numpy as np
import torch
import torch.nn as nn
import cv2

# --- Action Mapping Setup (must match training script) ---
BUTTONS = ['B', None, 'SELECT', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'A']
SIMPLE_MOVEMENT = [
    [],
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
    if is_new_episode or stacked_frames is None:
        stacked_frames = [frame] * 4
    else:
        stacked_frames.append(frame)
        stacked_frames.pop(0)
    return np.stack(stacked_frames, axis=0), stacked_frames

# --- Deep Q-Network: Convolutional Neural Network for image input (must match training) ---
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

# --- Load Environment and Trained Model ---
env = retro.make(game='SuperMarioBros-Nes')
action_space_n = len(SIMPLE_MOVEMENT)

policy_net = DQNCNN(action_space_n)
policy_net.load_state_dict(torch.load("src/Super_Mario_Bros_RL_Project/mario_dqn.pth"))
policy_net.eval()

# --- Run Test Episodes with Greedy Policy ---
EPISODES = 10
success_count = 0

for episode in range(EPISODES):
    state = env.reset()
    if isinstance(state, tuple):
        state = state[0]
    stacked_frames = None
    state, stacked_frames = stack_frames(None, state, True)
    total_reward = 0
    while True:
        
        # --- Greedy Action Selection (no exploration) ---
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            q_values = policy_net(state_tensor)
            action = torch.argmax(q_values).item()
        action_arr = action_to_array(action)
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
        state = next_state
        total_reward += reward

        # --- Success Condition: Mario reaches the flag ---
        if info.get('flag_get', False):
            print(f"Level completed in Test Episode {episode + 1} (Total Reward: {total_reward})")
            success_count += 1
            break
        if done:
            print(f"Episode {episode + 1} ended without success (Total Reward: {total_reward})")
            break

print(f"\nAverage Test Policy Success Rate: {(success_count / EPISODES) * 100:.1f}%")

env.close()