import os.path as osp
import sys

osp.dirname(__file__)
sys.path.append(osp.dirname(__file__))
sys.path.append(osp.join(osp.dirname(__file__), "lib"))
sys.path.append(osp.join(osp.dirname(__file__), "lib/objs"))

from matplotlib import pyplot
from matplotlib.animation import FuncAnimation

from match import Match
from field import Field
from tracking import Possession, Tracking


if __name__ == '__main__':
  dir: str = "lib/data/matches/2068"
  fig = pyplot.figure()

  match: Match = Match(dir)
  tracking: Tracking = Tracking(dir, match)
  field: Field = Field(match.pitch_length, match.pitch_width)

  def update(i):
    possession: Possession = tracking.getFrame(i)
    math_time = possession.time
    frame = possession.frame

    # 描写のリセット
    pyplot.cla()

    # フィールドの描写
    field.draw()

    # パスの描写
    if i in tracking.pases:
      pas = tracking.pases[i]
      lx = []
      ly = []
      for j in range(pas.start_frame, pas.end_frame):
        pos = match.ball.frames[j]
        lx.append(pos.x)
        ly.append(pos.y)
      pyplot.plot(lx, ly, label="pas_line", color="cyan")
      # pyplot.plot([pas.from_p.frames[i].x, pas.to_p.frames[i].x], [pas.from_p.frames[i].y, pas.to_p.frames[i].y], color="cyan", linewidth=2.0, label="pass")

    # プレイヤーの描写
    for id, player in match.players.items():
      if player.hasFrame(i):
        pos = player.frames[i]
        if id == possession.keep_player_id:
          pyplot.plot(pos.x, pos.y, label=f'{id}k', color="yellow", marker='o', markersize=16)
        pyplot.plot(pos.x, pos.y, label=f'{id}b', color="red" if player.isHome() else "green", marker='o', markersize=12, alpha=1.0 if not pos.predicted else 0)
        pyplot.plot(pos.x, pos.y, label=id, color="white", marker=f'${player.number}$', markersize=8 if player.number > 9 else 5)

    # ボールの描写
    if match.ball.hasFrame(i):
      pos = match.ball.frames[i]
      pyplot.plot(pos.x, pos.y, label=match.ball.obj_id, marker='o', color="black", markersize=6, alpha=1.0 if not pos.predicted else 0.3)

    # タイトルの描写
    title_text = f'{math_time} '
    title_text = title_text + f'[{frame}] '
    # title_text = title_text + f'{poss.count} objects '
    if possession.keep_player_id != -1:
      keep_player = match.getPlayer(possession.keep_player_id)
      HorA = 'H' if keep_player.is_home else 'A'
      title_text = title_text + f'keep=({HorA}{keep_player.number})'

    pyplot.title(f'{title_text}')
    print(f'{math_time} [{frame}]')

  anim = FuncAnimation(fig, update, frames=tracking.frame_list, interval=150)
  # anim.save('animation.gif', writer='imagemagick')
  pyplot.show()
