from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import sys
import cv2
import json
import copy
import numpy as np
from opts import opts
from detector import Detector
import datetime


image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge', 'display']


def demo(opt):
  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  opt.debug = max(opt.debug, 1)
  detector = Detector(opt)
  cap = cv2.VideoCapture(opt.demo)
  cap_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
  cap_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
  print("width : ", cap_width)
  print("height : ", cap_height)
  # Initialize output video
  dt_now = datetime.datetime.now()
  out = None
  out_name = dt_now.strftime('%Y:%m:%d:%H:%M:%S_') + (os.path.splitext(opt.demo)[0]).split('/')[-1]
  # out_name = opt.demo[opt.demo.rfind('/') + 1:]
  print('out_name', out_name)
  if opt.save_video:
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter('../results/{}.mp4'.format(
        opt.exp_id + '_' + out_name), fourcc, opt.save_framerate, (
        cap_width, cap_height))

  cnt = 0
  results = {}

  print('read img from : ', opt.demo)

  while (cap.isOpened()):
    ret, img = cap.read()

    img = cv2.resize(img, (cap_width, cap_height))

    if img is None:
      break

    ret = detector.run(img)

    # log run time
    time_str = 'frame {} |'.format(cnt)
    for stat in time_stats:
      time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
    print(time_str)

    results[cnt] = ret['results']
    out.write(ret['generic'])
    cnt += 1

  print('finish!! write to ', out_name)
  save_and_exit(opt, out, results, out_name)

  # while True:
  #   if cnt < len(image_names):
  #     img = cv2.imread(image_names[cnt])
  #   else:
  #     save_and_exit(opt, out, results, out_name)
  #   cnt += 1

  #   # resize the original video for saving video results
  #   if opt.resize_video:
  #     img = cv2.resize(img, (opt.video_w, opt.video_h))

  #   # skip the first X frames of the video
  #   # if cnt < opt.skip_first:
  #     # continue

  #   # cv2.imshow('input', img)

  #   # track or detect the image.
  #   ret = detector.run(img)

  #   # log run time
  #   time_str = 'frame {} |'.format(cnt)
  #   for stat in time_stats:
  #     time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
  #   print(time_str)

  #   # results[cnt] is a list of dicts:
  #   #  [{'bbox': [x1, y1, x2, y2], 'tracking_id': id, 'category_id': c, ...}]
  #   results[cnt] = ret['results']

  #   # save debug image to video
  #   if opt.save_video:
  #     out.write(ret['generic'])
  #     # cv2.imwrite('../results/demo{}.jpg'.format(cnt), ret['generic'])

  #   # esc to quit and finish saving video
  #   if cv2.waitKey(1) == 27:
  #     save_and_exit(opt, out, results, out_name)
  #     return
  # # save_and_exit(opt, out, results)


def save_and_exit(opt, out=None, results=None, out_name=''):
  print(opt.save_results)
  print(results)
  if (results is not None):
    save_dir = '../results/{}_results.json'.format(opt.exp_id + '_' + out_name)
    print('saving results to', save_dir)
    json.dump(_to_list(copy.deepcopy(results)),
              open(save_dir, 'w'))
  if opt.save_video and out is not None:
    out.release()
  sys.exit(0)


def _to_list(results):
  for img_id in results:
    for t in range(len(results[img_id])):
      for k in results[img_id][t]:
        if isinstance(results[img_id][t][k], (np.ndarray, np.float32)):
          results[img_id][t][k] = results[img_id][t][k].tolist()
  return results


if __name__ == '__main__':
  opt = opts().init()
  demo(opt)
