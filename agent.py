# Agent Class

class Agent:

  def __init__(self):
    self.action = None
    self.state = [0, 0, 0] # [row, column, block]
    self.past_state = None
    self.final_state = 0
    self.position = [0, 0]
    self.policy = None
    self.bank_account = 0
    self.reward = 0
    self.num_actions = 0

  def initialize(self):
    self.action = None
    self.state = [0, 0, 0]
    self.past_state = None
    self.position = [0, 0]
    self.bank_account = 0
    self.reward = 0
    self.num_actions = 0

  def reset(self):
    self.initialize()
    self.policy = None
    self.final_state = 0

  def updatePosition(self):
    self.position[0] = self.state[0]
    self.position[1] = self.state[1]

  def updateState(self, new_state):
    self.past_state = self.state
    self.state = new_state

  def updateRewards(self, reward):
    self.reward = reward
    self.bank_account += reward

  def canPickup(self):
    return self.state[2] == 0

  def blocked(self):
    return self.state[2] == 1

  def getFinalStates(self):
    return self.final_state

  def resetFinalState(self):
    self.final_state = 0