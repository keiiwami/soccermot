
import _init_paths

from detector import Detector
from opts import opts

def demo(opt):
  detector = Detector(opt)
	

if __name__ == '__main__':
	opt = opts().init()
	demo(opt)