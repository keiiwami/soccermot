
import json
from typing import Dict, List

from match import Match
from pas import Pas
from vec2 import Vec2


class Track():
  def __init__(self, id, x, y) -> None:
    self.id: int = id
    self.x: int = x
    self.y: int = y


class Possession():

  def __init__(self, data, match: Match) -> None:
    self.time = data['time']
    self.frame = data['frame']
    self.count = len(data['data'])
    for d in data['data']:
      if 'trackable_object' in d:
        id = d['trackable_object']
        if match.isPlayer(id):
          match.getPlayer(id).addPosition(self.frame, d['x'], d['y'])
        elif match.isBall(id):
          match.ball.addPosition(self.frame, d['x'], d['y'])

    self.keep_player_id = match.getPlayerNearestToBall(self.frame)


class Tracking():

  def __init__(self, dir, match: Match) -> None:
    with open(f"{dir}/structured_data.json") as f:
      data = json.load(f)

    self.frames: Dict[int, Possession] = {}
    self.pases: Dict[int, Pas] = {}

    self.frame_list: List[int] = []

    keep_player = None
    pass_start_pos: Vec2 = None
    pass_end_pos: Vec2
    pass_start_frame: int = None
    pass_end_frame: int
    for d in data:
      if 'time' in d:
        i = d['frame']
        self.frames[i] = possession = Possession(d, match)
        if possession.count > 0:
          self.frame_list.append(i)

        if possession.keep_player_id != -1:
          if keep_player is None:
            keep_player = match.getPlayer(possession.keep_player_id)
          elif keep_player.obj_id != possession.keep_player_id:
            next_player = match.getPlayer(possession.keep_player_id)
            if keep_player.is_home == next_player.is_home:  # 味方同士のパス
              pass_end_pos = next_player.frames[i]
              pass_end_frame = i
              if pass_start_frame is not None and pass_start_pos is not None:
                pas = Pas(pass_start_frame, pass_end_frame, pass_start_pos, pass_end_pos, keep_player, next_player)
                for j in range(pass_start_frame, pass_end_frame):
                  self.pases[j] = pas
            pass_start_frame = None
            pass_start_pos = None
            keep_player = next_player
          else:
            if keep_player.hasFrame(i):
              pass_start_pos = keep_player.frames[i]
              pass_start_frame = i

        i += 1

  def getFrame(self, i) -> Possession:
    if i in self.frames:
      return self.frames[i]
