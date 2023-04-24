from agent import Agent
from cells import *
import numpy as np
import copy
import random
import enum

# Environment matrix
environment = [\
    [[None, -1, -1, None], [None, -1, -1, -1], [None, -1, -1, -1], [None, -1, -1, -1], [None, None, -1, -1]],
    [[-1, -1, -1, None], [-1, -1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1], [-1, None, -1, -1]],
    [[-1, -1, -1, None], [-1, -1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1], [-1, None, -1, -1]],
    [[-1, -1, -1, None], [-1, -1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1], [-1, None, -1, -1]],
    [[-1, -1, None, None], [-1, -1, None, -1], [-1, -1, None, -1], [-1, -1, None, -1], [-1, None, None, -1]]]

# Pickup q table
pickup_q_table = [\
    [[[1,1,1,3,2,3,0,0,0,0,0,0,10,10],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0]],
    [[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0]],     
    [[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0]],     
    [[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0]],     
    [[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0]]],
    [[[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0]],
    [[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0]],     
    [[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0]],     
    [[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0]],     
    [[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0]]],
    [[[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0]],
    [[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0]],     
    [[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0]],     
    [[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0]],     
    [[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0]]],
    [[[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0]],
    [[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0]],     
    [[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0]],     
    [[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0]],     
    [[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0]]],
    [[[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0]],
    [[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0]],     
    [[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0]],     
    [[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0]],     
    [[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0]]]]

# Dropoff q table
dropoff_q_table = [\
    [[[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0]],
    [[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0]],     
    [[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0]],     
    [[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0]],     
    [[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0]]],
    [[[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0]],
    [[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0]],     
    [[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0]],     
    [[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0]],     
    [[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0]]],
    [[[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0]],
    [[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0]],     
    [[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0]],     
    [[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0]],     
    [[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0]]],
    [[[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0]],
    [[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0]],     
    [[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0]],     
    [[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0]],     
    [[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0]]],
    [[[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0]],
    [[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0]],     
    [[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0]],     
    [[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0]],     
    [[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0]]]]

# Creating class Action and enumerating
class Actions(enum.Enum):
  north  = 0
  east   = 1
  south  = 2
  west   = 3
  up     = 4
  down   = 5
  PICKUP = 6
  DROP   = 7
  RESET  = 8

bank_account = []
num_operator = []

# Creating Agent
agent = Agent()

# List of pickup and dropoff cells
pickup_cells = []  
dropoff_cells = [] 

# Random Policy: If pickup and dropoff is applicable, choose this operator; otherwise, choose an applicable operator randomly.
def PRANDOM(actions):
    return random.choice(actions)

# Exploit Policy: If pickup and dropoff is applicable, choose this operator; otherwise, apply the applicable operator with the 
# highest q-value (break ties by rolling a dice for operators with the same q-value) with probability 0.85 and choose a different 
# applicable operator randomly with probability 0.15. 
def PEXPLOIT(actions, agent, row, col, height):
    duplicate = []
    q_table = dropoff_q_table if agent.blocked() else pickup_q_table

    # Choose action (operator) with highet q-value 85% of the time
    if random.random() <= 0.85:
        max_operator = actions[0]
        for num in actions:
            q_value = q_table[row][col][height][num]
            max_q_value = q_table[row][col][height][max_operator]

            if q_value > max_q_value:
                max_operator = num
                duplicate.clear()
                duplicate.append(num)
            if q_value == max_q_value:
                duplicate.append(num)

        exploit_choice = random.choice(duplicate) if len(duplicate) > 1 else max_operator
    # The rest of the time 15% choose action randomly
    else:
        exploit_choice = random.choice(actions)

    return exploit_choice

# Greedy Policy: If pickup and dropoff is applicable, choose this operator; otherwise, apply the applicable operator with the 
# highest q-value (break ties by rolling a dice for operators with the same q-value). 
def PGREEDY(actions, agent, row, col, height):
    duplicate = []
    q_table = dropoff_q_table if agent.blocked() else pickup_q_table

    # Choose action (operator) with best q-value 100% of the time
    max_operator = actions[0]
    for num in actions:
        q_value = q_table[row][col][height][num]
        max_q_value = q_table[row][col][height][max_operator]

    if q_value > max_q_value:
        max_operator = num
        duplicate.clear()
        duplicate.append(num)
    if q_value == max_q_value:
        duplicate.append(num)

    greedy_choice = random.choice(duplicate) if len(duplicate) > 1 else max_operator

    return greedy_choice

# Function to reset bank account and operator lists
def clear_lists():
  bank_account.clear()
  num_operator.clear()

# Swap pickup and dropoff q tables
def swap_q_tables():
  global pickup_q_table
  global dropoff_q_table

  pickup_q_table, dropoff_q_table = dropoff_q_table, pickup_q_table

# Returns the cell in the given position 
def getCell(pos, cell_list):
  for cell in cell_list:
    if cell.position == pos:
      return cell

# Decrease number of blocks in Cell
def decreaseNumBlocks(pos):
  cell = getCell(pos, pickup_cells)
  cell.num_blocks -= 1

# Increase number of blocks in Cell
def increaseNumBlocks(pos):
  cell = getCell(pos, dropoff_cells)
  cell.num_blocks += 1

# Populate cells
def populateCells(pickup_states, dropoff_states):
  pickup_cells.clear()
  dropoff_cells.clear()

  for pos in pickup_states:
    #print(pos)
    pickup_cells.append(Cells(pos, 5))

  for pos in dropoff_states:
    #print(pos)
    dropoff_cells.append(Cells(pos, 0))

  for i in range(len(pickup_states)):
    print("Pickup cell ", i, " = ", pickup_cells[i].position)

  print()

  for i in range(len(dropoff_states)):
    print("Dropoff cell ", i, " = ", dropoff_cells[i].position)

# Function to define q-table
def initialize_Q_table():
  global pickup_q_table
  global dropoff_q_table

  for row in range(5):
    for col in range(5):
      for height in range(5):
        for i in range(5):
          pickup_q_table[row][col][height][i] = 0
          dropoff_q_table[row][col][height][i] = 0

# Calculate reward from given action
def calculateReward(action):
  negative_reward = ["north", "east", "south", "west", "up", "down"]
  positive_reward = ["PICKUP", "DROP"]

  operator = action.name
  if operator in positive_reward:
    reward = 14
  if operator in negative_reward:
    reward = -1
  # Need to apply condition for risk cell
  
  return reward

def isPickup(pos, pickup_states):
  return pos in pickup_states

def isDropOff(pos, dropoff_states):
  return pos in dropoff_states

# Returns action enum given possible action
def getPolicyAction(agent, state, actions_list, pickup_states, dropoff_states):
  row = state[0]
  col = state[1]
  height = state[2]
  pos = [row, col, height]

  # Returns position if cell in pickup or dropoff states
  if pos in pickup_states:
    cell = getCell(pos, pickup_cells)
    #print(cell)
  elif pos in dropoff_states:
    cell = getCell(pos, dropoff_cells)
    #print(cell)

  # Pickup or dropoff cell given the condition
  if pos in pickup_states and state[3] == 0 and not cell.isEmpty(): # Pickup if Agent has no block and cell is not empty
    action = Actions.PICKUP
    decreaseNumBlocks(pos)
    if cell.isEmpty():
      cell.isActive = False
    print('\nAgent picked up a block')
    print(f'Pickup cell {cell.position} = {cell.num_blocks}')
  elif pos in dropoff_states and state[3] == 1 and not cell.isFull(): # Dropoff if Agent has block and cell is not full
    action = Actions.DROP
    increaseNumBlocks(pos)
    if cell.isFull():
      cell.isActive = False
    print('\nAgent dropped off a block')
    print(f'Drop off cell {cell.position} = {cell.num_blocks}')
  else: # If neither case above, apply policy
    if agent.policy == "PRANDOM":
      action = Actions(PRANDOM(actions_list))
    if agent.policy == "PEXPLOIT":
      action = Actions(PEXPLOIT(actions_list, agent, row, col, height))
    if agent.policy == "PGREEDY":
      action = Actions(PGREEDY(actions_list, agent, row, col, height))

  agent.num_actions += 1
  #print(action)
  return action

# Returns new state when action is applied
def getNextState(old_state, action):
  new_state = copy.deepcopy(old_state)

  if action.name == "north":
    new_state[0] -= 1
  elif action.name == "east":
    new_state[1] += 1
  elif action.name == "south":
    new_state[0] += 1
  elif action.name == "west":
    new_state[1] -= 1
  elif action.name == "up":
    new_state[2] += 1
  elif action.name == "down":
    new_state[2] -= 1
  elif action.name == "PICKUP":
    new_state[3] = 1
  elif action.name == "DROP":
    new_state[3] = 0

  return new_state

# Returns max q value for given list of possible Actions
def getMaxQvalue(actions_list, state):
  
  if state[3] == 1:
    blocked = True 
  else:
    blocked = False

  if blocked: 
    q_table = dropoff_q_table 
  else:
    q_table = pickup_q_table

  row = state[0]
  col = state[1]
  height = state[2]

  possible_q_values = []
  #print(actions_list)
  for i in actions_list:
    possible_q_values.append(q_table[row][col][height][i]) 

  return np.max(possible_q_values)

# Returns all possible Actions for a given position
def getNextActions(state, pickup_states, dropoff_states):

  row = state[0]
  col = state[1]
  height = state[2]
  pos = [row, col, height]

  if state[3] == 1:
    blocked = True
  else:
    blocked = False

  actions_list = []

  for i in range(4):
    if environment[row][col][i] != None:
      actions_list.append(i)

  if pos in pickup_states:
    cell = getCell(pos, pickup_cells)
  elif pos in dropoff_states:
    cell = getCell(pos, dropoff_cells)

  if pos in pickup_states and not blocked and not cell.isEmpty():
    actions_list.append(Actions.PICKUP.value)
  elif pos in dropoff_states and blocked and not cell.isFull():
    actions_list.append(Actions.DROP.value)

  return (actions_list)


# Returns when terminal state reached
def Q_learning(learning_rate, discount_rate, agent, pickup_states, dropoff_states):
  position = agent.position
  row = position[0]
  col = position[1]
  height = position[2]

  actions_list = getNextActions(agent.state, pickup_states, dropoff_states) # Gets all possible Actions in state
  #print(actions_list)
  action = getPolicyAction(agent, agent.state, actions_list, pickup_states, dropoff_states) # Choose action
  #print(action)
  agent.action = action
  next_state = getNextState(agent.state, action) # Next state after action is applied
  #print(next_state)
  reward = calculateReward(action) # Calculates Rewards
  #print(reward)
  agent.updateRewards(reward)

  new_row = next_state[0]
  #print(new_row)
  new_col = next_state[1]
  #print(new_col)
  new_height = next_state[2]
  #print(new_height)

  q_table = dropoff_q_table if agent.blocked() else pickup_q_table
  #print(q_table)
  old_value = q_table[row][col][height][action.value]
  new_pos = [new_row, new_col, new_height]
  next_possible_action = getNextActions(next_state, pickup_states, dropoff_states)
  # next_action = getPolicyAction(agent, next_state, next_possible_action, pickup_states, dropoff_states)

  # if pickup or drop off
  if isPickup(new_pos, pickup_states):
    cell = getCell(new_pos, pickup_cells)
  elif isDropOff(new_pos, dropoff_states):
    cell = getCell(new_pos, dropoff_cells)

  # if (isPickup(new_pos, pickup_states) or isDropOff(new_pos, dropoff_states)) and not cell.isActive and not cell.available:
  #   next_max = np.max(q_table[new_row][new_col][0:4])
  # else:
  #   next_max = np.max(q_table[new_row][new_col])

  next_max = getMaxQvalue(next_possible_action, next_state)

  new_q_value = ((1 - learning_rate) * old_value) + (learning_rate * (agent.reward + discount_rate * next_max))
  q_table[row][col][height][action.value] = round(new_q_value, 2)

  print("Q-table: ", q_table[row][col][height])

  agent.updateState(next_state)
  agent.updatePosition()

  is_terminal = False
  val = 0
  
  # It finishes this loop succesfully
  count=0
  for cell in dropoff_cells:
    count += 1
    #print(count)
    val += cell.num_blocks
    #print(val)
    if val == 15:
      is_terminal = True

  if is_terminal:
    bank_account.append(agent.bank_account)
    num_operator.append(agent.num_actions)
    agent.initialize()
    agent.final_state += 1
    agent.action = Actions.RESET
    populateCells(pickup_states, dropoff_states)
    print("\n-----------INITIALIZED----------")

# SARSA update, returns next action #
def SARSA_Q_learning(learning_rate, discount_rate, next_action, agent, pickup_states, dropoff_states):

  is_terminal = False
  val = 0

  for cell in dropoff_cells:
    val += cell.num_blocks
    if val == 15:
      is_terminal = True

  if is_terminal:
    bank_account.append(agent.bank_account)
    num_operator.append(agent.num_actions)
    agent.initialize()
    agent.final_state += 1
    populateCells(pickup_states, dropoff_states)
    print("\n-----------INITIALIZED----------")
    agent.action = Actions.RESET
    return None

  position = agent.position
  row = position[0]
  col = position[1]
  height = position[2]

  if next_action is None:
    actions_list = getNextActions(agent.state, pickup_states, dropoff_states)
    action = getPolicyAction(agent, agent.state, actions_list, pickup_states, dropoff_states) 
  else:
    action = next_action                                                    

  # Calculate agent rewards and bank account
  reward = calculateReward(action) 
  agent.action = action
  agent.updateRewards(reward)

  next_state = getNextState(agent.state, action)                         
  new_row = next_state[0]
  new_col = next_state[1]
  new_height = next_state[2]
  new_pos = [new_row, new_col, new_height]

  next_actions = getNextActions(next_state, pickup_states, dropoff_states) 
  next_action = getPolicyAction(agent, next_state, next_actions, pickup_states, dropoff_states) 

  q_table = dropoff_q_table if agent.blocked() else pickup_q_table
  old_value = q_table[row][col][height][action.value]

  new_q_value = old_value + learning_rate * (reward + discount_rate * q_table[next_state[0]][next_state[1]][next_state[2]][next_action.value] - old_value)

  q_table[row][col][height][action.value] = round(new_q_value, 2)

  agent.updateState(next_state)
  agent.updatePosition()

  return next_action