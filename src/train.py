from gymnasium.wrappers import TimeLimit
import torch
import torch.nn as nn


from env_hiv import HIVPatient

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!

class Network(torch.nn.Module):
    def __init__(self, state_dim, n_actions, n_layers=4, hidden_dim=256):
        super(Network, self).__init__()

        self.first_layer = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU())
        self.last_layer = nn.Linear(hidden_dim, n_actions)
        self.mlp = nn.ModuleList([nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()) for _ in range(n_layers)])

    def forward(self, x):
        x = self.first_layer(x)
        for layer in self.mlp:
            x = layer(x)
        x = self.last_layer(x)
        return x


config = {
    "n_layers":4,
    "hidden_dim":256
}


class ProjectAgent:
    def __init__(self, config=config):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = Network(env.observation_space.shape[0], env.action_space.n, config["n_layers"], config["hidden_dim"]).to(self.device)

        #Init means and stds
        self.state_mean = torch.tensor([3.6e5, 7.75e3, 287, 33, 36.8e3, 55], dtype=torch.float32).to(self.device)
        self.state_std = torch.tensor([1.28788e5, 1.4435e4, 345, 25, 70.083e3, 32], dtype=torch.float32).to(self.device)

    def normalize_state(self, state):
        out = (state - self.state_mean) / self.state_std
        return out


    def act(self, observation, use_random=False):
        observation = torch.tensor(observation, dtype=torch.float32).to(self.device)
        observation = self.normalize_state(observation).unsqueeze(0)
        with torch.no_grad():
            Q = self.model(observation)
            action = torch.argmax(Q).item()
        return action

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self):
        self.model.load_state_dict(
            torch.load("./models/best_model.pt", map_location=torch.device("cpu"))
        )
        self.model.eval()