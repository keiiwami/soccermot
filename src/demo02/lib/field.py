from matplotlib import pyplot


class Field():

  def __init__(self, l, w) -> None:
    self.p_l = l
    self.p_w = w

  def draw(self):
    line_color = "black"
    pyplot.xlim(-(self.p_l * 1.2 / 2), self.p_l * 1.2 / 2)
    pyplot.ylim(-(self.p_w * 1.2 / 2), self.p_w * 1.2 / 2)
    pyplot.plot([-(self.p_l / 2), (self.p_l / 2)], [-(self.p_w / 2), -(self.p_w / 2)], color=line_color, label="a")
    pyplot.plot([-(self.p_l / 2), (self.p_l / 2)], [(self.p_w / 2), (self.p_w / 2)], color=line_color, label="b")
    pyplot.plot([-(self.p_l / 2), -(self.p_l / 2)], [-(self.p_w / 2), (self.p_w / 2)], color=line_color, label="c")
    pyplot.plot([(self.p_l / 2), (self.p_l / 2)], [-(self.p_w / 2), (self.p_w / 2)], color=line_color, label="d")
    pyplot.plot([0, 0], [-(self.p_w / 2), (self.p_w / 2)], color=line_color, label="e")
