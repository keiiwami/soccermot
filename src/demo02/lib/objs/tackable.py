from vec2 import Vec2
from typing import Dict


class Trackable():

  def __init__(self, id: int) -> None:
    self.obj_id: int = id
    self.frames: Dict[int, Vec2] = {}

  def addPosition(self, i: int, x: float, y: float) -> None:
    self.frames[i] = Vec2(x, y)
    j = i
    while (j - 1) not in self.frames and j > 0:
      j -= 1
    if j != 0 and j != i:
      sx = (self.frames[i].x - self.frames[j - 1].x) / (i - j + 1)
      sy = (self.frames[i].y - self.frames[j - 1].y) / (i - j + 1)
      for k in range(j, i):
        self.frames[k] = Vec2(self.frames[k - 1].x + sx, self.frames[k - 1].y + sy, True)

  def hasFrame(self, i) -> bool:
    return i in self.frames
