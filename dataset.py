import torch
import h5py
from PIL import Image
import numpy as np


class NYUDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, tfms):
        super(NYUDataset, self).__init__()
        self.data_dir = data_dir
        self.tfms = tfms

        self.ds_v_1 = h5py.File(self.data_dir)

        self.len = len(self.ds_v_1["images"])
        print(f'{self.len} !!!')

    def __getitem__(self, index):
        # index가 size보다 크는 것 예외처리 안하나?
        ds = self.ds_v_1
        i = index


        img = np.transpose(ds["images"][i], axes=[2, 1, 0])
        img = img.astype(np.uint8)

        depth = np.transpose(ds["depths"][i], axes=[1, 0])
        depth = (depth / depth.max()) * 255
        depth = depth.astype(np.uint8)

        if self.tfms:
            tfmd_sample = self.tfms({"image": img, "depth": depth})
            img, depth = tfmd_sample["image"], tfmd_sample["depth"]
        return (img, depth)

    def __len__(self):
        return self.len