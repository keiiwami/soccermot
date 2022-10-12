import os.path as osp
import os
from statistics import mode
import sys


osp.dirname(__file__)
sys.path.append(osp.dirname(__file__))
sys.path.append(osp.join(osp.dirname(__file__), "lib"))
sys.path.append(osp.join(osp.dirname(__file__), "lib/twogan"))
sys.path.append(osp.join(osp.dirname(__file__), "lib/twogan/util"))
sys.path.append(osp.join(osp.dirname(__file__), "lib/sccvsd"))
sys.path.append(osp.join(osp.dirname(__file__), "lib/sccvsd/util"))

from lib.twogan.two_pix2pix_model import TwoPix2PixModel
from lib.twogan.test_options import TestOptions
from lib.twogan.custom_dataset_data_loader import CustomDatasetDataLoader
from lib.twogan.util.visualizer import Visualizer
from lib.twogan.util import html

from lib.sccvsd.camera_dataset import CameraDataset
from lib.sccvsd.siamese import BranchNetwork, SiameseNetwork
from lib.sccvsd.util.projective_camera import ProjectiveCamera
from lib.sccvsd.util.iou_util import IouUtil
from lib.sccvsd.util.synthetic_util import SyntheticUtil

import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import scipy.io as sio
import cv2 as cv
from collections import OrderedDict
import pyflann


# TODO 入力：放送映像 ( jpeg ) の読み込み

# TODO two-GANでエッジ(mat)生成
opt = TestOptions()
opt.dataroot = "./data/soccer_seg_detection"

opt.how_many = 100

# opt.dataroot = "./data/SNMOT-060/img"
opt.which_direction = "AtoB"
opt.model = "two_pix2pix"
opt.name = "soccer_seg_detection_pix2pix"
opt.output_nc = 1
opt.dataset_model = "aligned"
opt.which_model_netG = "unet_256"
opt.norm = "batch"
opt.loadSize = 256
opt.gpu_ids = []  # not gpu
opt.which_epoch = 150
opt.phase = "test"
opt.resize_or_crop = "resize_and_crop"
opt.checkpoints_dir = "../../refarence/pytorch-two-GAN/checkpoints"
opt.input_nc = 3
opt.ngf = 64
opt.no_dropout = "store_true"
opt.init_type = "normal"
opt.fineSize = 256
opt.max_dataset_size = float("inf")
opt.results_dir = "./result"
opt.display_id = 1
opt.display_winsize = 256
opt.display_port = 8097
opt.aspect_ratio = 1.0
# opt.aspect_ratio = 180 / 320
opt.parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip
opt.continue_train = False

# DataSet
data_loader = CustomDatasetDataLoader()
data_loader.initialize(opt)
dataset = data_loader.load_data()

# Visualizer
visualizer = Visualizer(opt)
web_dir = os.path.join(opt.results_dir, "webpage")
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

# モデル生成
model: TwoPix2PixModel = TwoPix2PixModel()
model.initialize(opt)

pivot_images = np.zeros((10000, 1, 180, 320), np.float32)
edge_map = np.zeros((10000, 180, 320, 3), np.float32)
gan_visuals = []
for i, data in enumerate(dataset):
    if i >= opt.how_many:
        break
    model.set_input(data)
    img_path = model.get_image_paths()
    print('%04d: process image... %s' % (i, img_path))
    model.test()
    gan_visuals.append(model.get_current_visuals())

    im = model.get_im_fake_D()
    im = np.array(Image.fromarray(im).resize((180, 320)))

    for j, d1 in enumerate(im):
        for k, rgb in enumerate(d1):
            g = ((rgb[0] + rgb[1] + rgb[2]) / 256)
            pivot_images[i][0][k][j] = g
            edge_map[i][k][j][0] = g
            edge_map[i][k][j][1] = g
            edge_map[i][k][j][2] = g


# TODO networkでfeature変換
normalize = transforms.Normalize(mean=[0.0188],
                                 std=[0.128])

data_transform = transforms.Compose(
    [transforms.ToTensor(),
     normalize,
     ]
)
data_loader = CameraDataset(pivot_images,
                            1,  # batch size for test
                            -1,  # not gpu
                            data_transform,
                            is_train=False)

# 2: load network
device = 'cpu'
model_name = "./models/network.pth"
branch = BranchNetwork()
net = SiameseNetwork(branch)
checkpoint = torch.load(model_name, map_location=lambda storage, loc: storage)
net.load_state_dict(checkpoint['state_dict'])

data = sio.loadmat('./models/feature_camera_10k.mat')
database_features = data['features']
database_cameras = data['cameras']
flann = pyflann.FLANN()
print(database_features.shape)

data = sio.loadmat('./models/worldcup2014.mat')
model_points = data['points']
model_line_index = data['line_segment_index']
template_h = 74  # yard, soccer template
template_w = 115

for i in range(opt.how_many):
    # for i in range(len(data_loader)):
    print('%04d: retrieved image...' % (i))

    x, _ = data_loader[i]
    x = x.to(device)
    feat = net.feature_numpy(x)  # N x C
    # print(feat[0].shape)
    # print(database_features[0].shape)

    # Step 2: retrieve a camera using deep features
    result, _ = flann.nn(database_features, feat[0], 1, algorithm="kdtree", trees=8, checks=64)
    retrieved_index = result[0]
    retrieved_camera_data = database_cameras[retrieved_index]
    u, v, fl = retrieved_camera_data[0:3]
    rod_rot = retrieved_camera_data[3:6]
    cc = retrieved_camera_data[6:9]
    retrieved_camera = ProjectiveCamera(fl, u, v, cc, rod_rot)
    retrieved_h = IouUtil.template_to_image_homography_uot(retrieved_camera, template_h, template_w)
    retrieved_image = SyntheticUtil.camera_to_edge_image(retrieved_camera_data, model_points, model_line_index,
                                                         im_h=720, im_w=1280, line_width=4)
    query_image = edge_map[i]
    # cv.imwrite('result/sccvsd/result{}_A.jpg'.format(i), query_image)
    # cv.imwrite('result/sccvsd/result{}_Q.jpg'.format(i), retrieved_image)

    # dist_threshold = 50
    # query_dist = SyntheticUtil.distance_transform(query_image)
    # retrieved_dist = SyntheticUtil.distance_transform(retrieved_image)
    # cv.imwrite('result/sccvsd/result{}_D.jpg'.format(i), retrieved_dist.astype(np.uint8))
    # h_retrieved_to_query = SyntheticUtil.find_transform(retrieved_dist, query_dist)
    # refined_h = h_retrieved_to_query @ retrieved_h

    im = np.ones((720, 1280, 3), dtype=np.uint8) * 255
    h_img = IouUtil.homography_warp(np.linalg.inv(retrieved_h), im, (template_w, template_h), (0))
    # cv.imwrite('result/sccvsd/result{}_H.jpg'.format(i), h_img)

    visuals = gan_visuals[i]
    visuals["retrieved_image"] = retrieved_image
    visuals["template_H"] = h_img
    visualizer.save_images2(webpage, visuals, i, 180, 320)

webpage.save()
