# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 18:38:11 2021

To-do:
    - Get rid of gradients
    - Smallest dtype as possible
    - Compare time numpy vs pytorch

@author: Arthur Baucour
"""


from matplotlib import cm
from PIL import Image
import time
import torch
# import torch.nn as nn
# import torch.nn.functional as F

import patterns
from environment import GameOfLife


def get_image(game_state, x_image, y_image):
    """ Make """
    im = Image.fromarray(cm.Blues(game_state[:, :, 0].cpu().numpy().T,
                                  bytes=True))
    im = im.resize((x_image, y_image), resample=0)
    return im


# =============================================================================
# Init
# =============================================================================


# Make seed

x_size, y_size = 100, 100
seed = torch.rand(x_size, y_size, dtype=torch.float)
seed = torch.round(seed)

# glider_gun = patterns.glider_gun()
# seed = torch.zeros((40, 80))
# seed[20:31, 0:38] = glider_gun

# seed[17:20, 0:3] = glider
# seed[0:3, 17:20] = glider
# seed[17:20, 17:20] = glider

# Start the game

life = GameOfLife(seed)

# Parameters

epoch = 200
time_per_image = 160    # Time in ms
x_image, y_image = seed.shape[0] * 4, seed.shape[1] * 4


# =============================================================================
# Play the game
# =============================================================================


images = []

# Store the initial state
im = get_image(life.game_state, x_image, y_image)
images.append(im)

for i in range(epoch):

    # Update the game
    life.update()

    # Get the image
    im = get_image(life.game_state, x_image, y_image)
    images.append(im)

    if ((i + 1) % 10 == 0):
        print("[{} / {}]".format(i + 1, epoch))

# Duration is time per fram in ms
images[0].save('life.gif', save_all=True, append_images=images[1:],
               optimize=False, duration=time_per_image, loop=0)

# images[0].save('life.gif', save_all=True, append_images=images[1:],
#                optimize=False, duration=250, loop=0)