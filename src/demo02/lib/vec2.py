import math


class Vec2():
  def __init__(self, x: float, y: float, predicted: bool = False) -> None:
    self.x = x
    self.y = y
    self.predicted = predicted

  def getDistance(self, vec):
    pow_dx = (self.x - vec.x) * (self.x - vec.x)
    pow_dy = (self.y - vec.y) * (self.y - vec.y)
    return math.sqrt(pow_dx + pow_dy)
