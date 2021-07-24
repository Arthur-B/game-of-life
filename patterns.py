# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 20:33:36 2021

Compared to wikipedia the x and y index are flipped
(image vs matrix index, yada, yada)

@author: Arthur Baucour
"""

import torch
import numpy as np
import matplotlib.pyplot as plt


# Still lifes


def block():
    x_size, y_size = 4, 4

    pattern = torch.zeros((x_size, y_size))
    pattern[1, 1:3] = 1
    pattern[2, 1:3] = 1
    return pattern


def beehive():
    x_size, y_size = 6, 5
    pattern = torch.zeros((x_size, y_size))
    pattern[1, 2:4] = 1
    pattern[2, 1] = 1
    pattern[2, 4] = 1
    pattern[3] = pattern[1]
    return pattern


def loaf():
    x_size, y_size = 6, 6
    pattern = torch.zeros((x_size, y_size))
    pattern[1, 3] = 1
    pattern[2, 2] = 1
    pattern[3, 1] = 1
    pattern[2:4, 4] = 1
    pattern[4, 2:4] = 1
    return pattern


def boat():
    x_size, y_size = 5, 5
    pattern = torch.zeros((x_size, y_size))
    pattern[1, 2] = 1
    pattern[2, 1] = 1
    pattern[2, 3] = 1
    pattern[3, 1:3] = 1
    return pattern


def tub():
    x_size, y_size = 5, 5
    pattern = torch.zeros((x_size, y_size))
    pattern[1, 2] = 1
    pattern[2, 1] = 1
    pattern[2, 3] = 1
    pattern[3, 2] = 1
    return pattern


# Oscillators


def blinker():
    x_size, y_size = 5, 5
    pattern = torch.zeros((x_size, y_size))
    pattern[2, 1:4] = 1
    return pattern


def toad():
    x_size, y_size = 6, 6
    pattern = torch.zeros((x_size, y_size))
    pattern[2, 1:4] = 1
    pattern[3, 2:5] = 1
    return pattern


def beacon():
    x_size, y_size = 6, 6
    pattern = torch.zeros((x_size, y_size))
    pattern[1, 3:5] = 1
    pattern[2, 4] = 1
    pattern[3, 1] = 1
    pattern[4, 2:4] = 1
    return pattern


def glider():
    x_size, y_size = 3, 3
    pattern = torch.zeros((x_size, y_size))
    pattern[0, 1] = 1
    pattern[1, 2] = 1
    pattern[2, 0:3] = 1
    return pattern


def glider_gun():
    x_size, y_size = 11, 38
    pattern = torch.zeros((x_size, y_size))

    # The two block on the sides
    pattern[3:7, 0:4] = block()
    pattern[5:9, 34:] = block()

    # left part
    pattern[3:6, 11] = 1
    pattern[2, 12] = 1
    pattern[6, 12] = 1
    pattern[1, 13:15] = 1
    pattern[7, 13:15] = 1
    pattern[4, 15] = 1
    pattern[2, 16] = 1
    pattern[6, 16] = 1
    pattern[3:6, 17] = 1
    pattern[4, 18] = 1

    # Right part
    pattern[5:8, 21:23] = 1
    pattern[4, 23] = 1
    pattern[8, 23] = 1
    pattern[3:5, 25] = 1
    pattern[8:10, 25] = 1

    return pattern

# =============================================================================
# Main


def main():
    seed = glider_gun().numpy()
    plt.imshow(seed, origin='lower')
    plt.xticks(np.arange(seed.shape[1]))
    plt.yticks(np.arange(seed.shape[0]))
    plt.grid()


if __name__ == "__main__":
    main()
