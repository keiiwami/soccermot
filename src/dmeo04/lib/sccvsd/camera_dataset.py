import random
from PIL import Image

import torch
from torch.utils.data.dataset import Dataset


class CameraDataset(Dataset):
    def __init__(self,
                 pivot_data,
                 batch_size,
                 num_batch,
                 data_transform,
                 is_train=True):
        """
        :param pivot_data: N x 1 x H x W
        :param positive_data: N x 1 x H x W
        :param batch_size:
        :param num_batch:
        """
        super(CameraDataset, self).__init__()

        self.pivot_data = pivot_data
        self.batch_size = batch_size
        self.num_batch = num_batch
        self.data_transform = data_transform
        self.num_camera = pivot_data.shape[0]

        self.positive_index = []
        self.negative_index = []
        self.is_train = is_train

        if self.is_train:
            self._sample_once()

        if not self.is_train:
            # in testing, loop over all pivot cameras
            self.num_batch = self.num_camera // batch_size
            if self.num_camera % batch_size != 0:
                self.num_batch += 1

    def _get_test_item(self, index):
        """
        In testing, the label is hole-fill value, not used in practice.
        :param index:
        :return:
        """
        assert index < self.num_batch

        n, c, h, w = self.pivot_data.shape
        batch_size = self.batch_size

        start_index = batch_size * index
        end_index = min(start_index + batch_size, self.num_camera)
        bsize = end_index - start_index

        x = torch.zeros(bsize, c, h, w)
        label_dummy = torch.zeros(bsize)

        for i in range(start_index, end_index):
            pivot = self.pivot_data[i].squeeze()
            pivot = Image.fromarray(pivot)

            x[i - start_index, :] = self.data_transform(pivot)
            label_dummy[i - start_index] = 0

        #x = torch.tensor(x, requires_grad=True)
        x = x.clone().detach().requires_grad_(True)
        return x, label_dummy

    def __getitem__(self, index):
        if self.is_train:
            return self._get_train_item(index)
        else:
            return self._get_test_item(index)

    def __len__(self):
        return self.num_batch
