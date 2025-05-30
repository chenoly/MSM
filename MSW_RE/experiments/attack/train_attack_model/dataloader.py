import os
import cv2
import numpy as np
from PIL import Image
from torch.utils import data
from torchvision import transforms


def find_all_images(data_dir):
    image_extensions = ['.bmp', '.png', '.jpg', '.jpeg', '.gif']  # 常见的图像文件格式扩展名
    image_files = []  # 存储找到的图像文件路径
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                relative_path = os.path.relpath(os.path.join(root, file), data_dir)
                image_files.append(os.path.join(data_dir, relative_path))  # 将相对路径添加到列表
    return image_files


class MSGDS(data.Dataset):
    def __init__(self, dataset_path, im_size):
        super().__init__()
        self.dataset_path = dataset_path
        self.im_size = im_size

        self.digital_msg_path = f"{self.dataset_path}/digital/"
        self.genuine_msg_path = f"{self.dataset_path}/scanned/"

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

        self.digital_msg = find_all_images(self.digital_msg_path)
        self.genuine_msg = find_all_images(self.genuine_msg_path)
        self.digital_msg.sort()
        self.genuine_msg.sort()

    def __len__(self):
        return len(self.digital_msg)

    def __getitem__(self, index):
        curr_digital_msg_path = self.digital_msg[index]
        curr_genuine_msg_path = self.genuine_msg[index]
        digital_msg = cv2.imread(curr_digital_msg_path, 0)
        genuine_msg = cv2.imread(curr_genuine_msg_path, 0)
        v = int(genuine_msg.shape[0] / digital_msg.shape[0])
        digital_msg = cv2.resize(digital_msg, dsize=genuine_msg.shape, interpolation=cv2.INTER_NEAREST)
        start_h = np.random.choice([_ for _ in range(0, digital_msg.shape[0] - self.im_size, v)], size=1)[0]
        start_w = np.random.choice([_ for _ in range(0, digital_msg.shape[1] - self.im_size, v)], size=1)[0]
        block_digital_msg = digital_msg[start_w:start_w + self.im_size, start_h:start_h + self.im_size]
        block_genuine_msg = genuine_msg[start_w:start_w + self.im_size, start_h:start_h + self.im_size]
        block_digital_msg = cv2.resize(block_digital_msg, (block_genuine_msg.shape[0], block_genuine_msg.shape[1]), interpolation=cv2.INTER_NEAREST)
        trans_digital = self.transform(Image.fromarray(np.uint8(block_digital_msg)))
        trans_genuine = self.transform(Image.fromarray(np.uint8(block_genuine_msg)))
        return trans_genuine, trans_digital
