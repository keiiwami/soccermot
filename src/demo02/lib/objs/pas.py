
from matplotlib.pyplot import cla
from player import Player

from vec2 import Vec2


class Pas():
  def __init__(self, start_frame: int, end_frame: int, start_pos: Vec2, end_pos: Vec2, from_p: Player, to_p: Player) -> None:
    self.start_frame = start_frame
    self.end_frame = end_frame
    self.start_pos = start_pos
    self.end_pos = end_pos
    self.from_p = from_p
    self.to_p = to_p
