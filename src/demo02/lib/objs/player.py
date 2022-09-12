from tackable import Trackable
from vec2 import Vec2
from typing import Dict


class Player(Trackable):

  def __init__(self, name, id: int, number, role, is_home) -> None:
    self.name: str = name
    self.number: int = number
    self.role: str = role
    self.is_home: bool = is_home
    super().__init__(id)

  def isHome(self) -> bool:
    return self.is_home
