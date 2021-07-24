# -*- coding: utf-8 -*-
"""
Created on Sat Jul 24 12:53:49 2021

@author: Arthur
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Life initialized on {}".format(device))


class LifeNet(nn.Module):
    """
    Neural network to perform the logic rules of the Life game.

    input: [x1, x2]
        x1: The state of the current cell (0: dead, 1: alive)
        x2: Number of neighbors

    output: x
        x:  Next state of the cell
    """

    values = torch.tensor([0], dtype=torch.float).to(device)

    def __init__(self):
        super(LifeNet, self).__init__()
        with torch.no_grad():
            # FC to draw the boundaries
            self.fc1 = nn.Linear(2, 2)
            self.fc1.weight.copy_(torch.tensor([[1, 1],
                                                [0, -1]]))
            self.fc1.bias.copy_(torch.tensor([-2.5,
                                              3.5]))

            # AND logic gate
            self.fc2 = nn.Linear(2, 1)
            self.fc2.weight.copy_(torch.tensor([[1, 1]]))
            self.fc2.bias.copy_(torch.tensor([-1.5]))

    def forward(self, x):
        with torch.no_grad():
            # Draw the 2 boundaries
            x = self.fc1(x)
            x = torch.heaviside(x, values=self.values)
            # AND logic gate
            x = self.fc2(x)
            x = torch.heaviside(x, values=self.values)

        return x


class GameOfLife:
    """
    Overall class to play the game of Life.
    """

    # Required tensor to count the neighbors
    check_neighbor = torch.tensor([[[[1, 1, 1],
                                     [1, 0, 1],
                                     [1, 1, 1]]]], dtype=torch.float)
    check_neighbor = check_neighbor.to(device)

    def __init__(self, seed):

        self.x_size = seed.shape[0]
        self.y_size = seed.shape[1]

        seed = seed.to(device)
        neighbor_tensor = self.count_neighbors(seed)

        # game_state: 100 (x), 100 (y), 2 (game map, nb of neighbors)
        self.game_state = torch.torch.stack((seed, neighbor_tensor))
        self.game_state = self.game_state.permute(1, 2, 0)

        # Network to perform the logic operations with a perceptron
        self.net = LifeNet()
        self.net = self.net.to(device)

    def count_neighbors(self, game_map):
        return F.conv2d(game_map.view(1, 1, self.x_size, self.y_size),
                        self.check_neighbor, padding=1).squeeze()

    def update(self):

        # Update game with logic
        self.game_state[:, :, 0] = self.net(self.game_state.view(-1, 2)).reshape(self.x_size, self.y_size)

        # Compute new neighbors
        self.game_state[:, :, 1] = self.count_neighbors(self.game_state[:, :, 0])

    def print(self):
        print(self.game_state[:, :, 0].numpy().squeeze())
