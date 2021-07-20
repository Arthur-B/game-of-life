# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 18:38:11 2021

To-do:
    - Get rid of gradients
    - reshape input of perceptron to process all pixels in parallel
    - Smallest dtype as possible
    - Compare time numpy vs pytorch

@author: Arthur Baucour
"""


from matplotlib import cm
from PIL import Image
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

import patterns


class LifeNet(nn.Module):
    """
    Neural network to perform the logic rules of the Life game.

    input: [x1, x2]
        x1: The state of the current cell (0: dead, 1: alive)
        x2: Number of neighbors

    output: x
        x:  Next state of the cell
    """

    values = torch.tensor([0], dtype=torch.float)

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

    # Device for GPU acceleration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Life initialized on {}".format(device))

    # Required tensor to count the neighbors
    check_neighbor = torch.tensor([[[[1, 1, 1],
                                     [1, 0, 1],
                                     [1, 1, 1]]]], dtype=torch.float).to(device)


    def __init__(self, seed):

        self.x_size = seed.shape[0]
        self.y_size = seed.shape[1]

        seed = seed.to(self.device)
        neighbor_tensor = self.count_neighbors(seed)

        # game_state: 100 (x), 100 (y), 2 (game map, nb of neighbors)
        self.game_state = torch.torch.stack((seed, neighbor_tensor))
        self.game_state = self.game_state.permute(1, 2, 0)

        # Network to perform the logic operations with a perceptron
        self.net = LifeNet()
        self.net = self.net.to(self.device)


    def count_neighbors(self, game_map):
        return F.conv2d(game_map.view(1, 1, self.x_size, self.y_size),
                        self.check_neighbor, padding=1).squeeze()


    # def update_numpy(self):

    #     # Count the neighbors
    #     neighbor_tensor = self.count_neighbors()

    #     # Follow the rules
    #     for i in range(self.x_size):
    #         for j in range(self.y_size):

    #             # Get the number of neighbors from the tensor
    #             nb_neighbors = neighbor_tensor[0, 0, i, j].item()

    #             # If we have a live cell (value is 1)
    #             if self.game_state[0, 0, i, j].item():
    #                 # If neighbor under two (underpopulation)
    #                 # or over three (overpopulation), then die
    #                 if (nb_neighbors < 2) or (nb_neighbors > 3):
    #                     self.game_state[0, 0, i, j] = 0
    #             # If we have a dead cell
    #             else:
    #                 # If exactly three neighbors then come alive
    #                 if (nb_neighbors == 3):
    #                     self.game_state[0, 0, i, j] = 1


    def update(self):

        # Update game with logic
        self.game_state[:, :, 0] = self.net(self.game_state.view(-1, 2)).reshape(self.x_size, self.y_size)

        # Compute new neighbors
        self.game_state[:, :, 1] = self.count_neighbors(self.game_state[:, :, 0])


    def get_image(self, x_image, y_image):
        im = Image.fromarray(cm.Blues(self.game_state[:, :, 0].numpy().T, bytes=True))
        im = im.resize((x_image, y_image), resample=0)
        return im


    def print(self):
        print(self.game_state[:, :, 0].numpy().squeeze())

# =============================================================================
# Init
# =============================================================================

# Parameters

epoch = 60
x_image, y_image = 1280, 640


# Make seed

x_size, y_size = 128, 64
seed = torch.rand(x_size, y_size, dtype=torch.float)
seed = torch.round(seed)

# x_size, y_size, seed = patterns.beacon()

# Start the game

life = GameOfLife(seed)


# =============================================================================
# Play the game
# =============================================================================

images = []

# Store the initial state
im = life.get_image(x_image, y_image)
images.append(im)

for i in range(epoch):

    # Update the game
    life.update()

    # Get the image
    im = life.get_image(x_image, y_image)
    images.append(im)

    if ((i + 1) % 10 == 0):
        print("[{} / {}]".format(i + 1, epoch))

# Duration is time per fram in ms
images[0].save('life.gif', save_all=True, append_images=images[1:],
               optimize=False, duration=250, loop=0)