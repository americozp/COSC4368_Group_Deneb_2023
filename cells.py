# Cells Class

class Cells:

  def __init__(self, position, blocks):
    self.position = position
    self.num_blocks = blocks
    self.isActive = True
    self.isAvailable = True

  def isEmpty(self):
    return self.num_blocks == 0

  def isFull(self):
    return self.num_blocks == 5