import json
from typing import Dict

from objs.ball import Ball
from objs.player import Player


class Match():

  def __init__(self, dir) -> None:
    with open(f"{dir}/match_data.json") as f:
      mi = json.load(f)

    self.players: Dict[int, Player] = {}
    self.pitch_length: int = mi['pitch_length']
    self.pitch_width: int = mi['pitch_width']

    self.home_team_id: int = mi["home_team"]["id"]
    self.away_team_id: int = mi["away_team"]["id"]

    for ms in mi['players']:
      self.players[ms['trackable_object']] = Player(
          ms['first_name'] + " " + ms['last_name'],
          ms['trackable_object'],
          ms['number'],
          ms['player_role']['name'],
          ms['team_id'] is self.home_team_id)

    self.ball_id = mi["ball"]["trackable_object"]
    self.ball = Ball(self.ball_id)

  def isPlayer(self, id) -> bool:
    return id in self.players

  def isBall(self, id) -> bool:
    return id == self.ball_id

  def getPlayer(self, id) -> Player:
    return self.players[id]

  def getPlayerNearestToBall(self, frame) -> Player:
    min_distance = 1.5
    p = -1
    if self.ball.hasFrame(frame):
      for i, player in self.players.items():
        if player.hasFrame(frame):
          d = player.frames[frame].getDistance(self.ball.frames[frame])
          if d < min_distance:
            min_distance = d
            p = i
    return p
