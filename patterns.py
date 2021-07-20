# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 20:33:36 2021

Compared to wikipedia the x and y index are flipped
(image vs matrix index, yada, yada)

@author: Arthur Baucour
"""

import torch

# Still lifes

def block():
    x_size, y_size = 4, 4

    pattern = torch.zeros((x_size, y_size))
    pattern[1, 1:3] = 1
    pattern[2, 1:3] = 1
    return x_size, y_size, pattern

def beehive():
    x_size, y_size = 6, 5
    pattern = torch.zeros((x_size, y_size))
    pattern[1, 2:4] = 1
    pattern[2, 1] = 1
    pattern[2, 4] = 1
    pattern[3] = pattern[1]
    return x_size, y_size, pattern

# Oscillators

def blinker():
    x_size, y_size = 5, 5
    pattern = torch.zeros((x_size, y_size))
    pattern[2, 1:4] = 1
    return x_size, y_size, pattern

def toad():
    x_size, y_size = 6, 6
    pattern = torch.zeros((x_size, y_size))
    pattern[2, 1:4] = 1
    pattern[3, 2:5] = 1
    return x_size, y_size, pattern

def beacon():
    x_size, y_size = 6, 6
    pattern = torch.zeros((x_size, y_size))
    pattern[1, 3:5] = 1
    pattern[2, 4] = 1
    pattern[3, 1] = 1
    pattern[4, 2:4] = 1
    return x_size, y_size, pattern