import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils import data
from torchvision import transforms


class AttackDataset(data.Dataset):
    def __init__(self, root_path, dpi, ppi, img_size):
        super(AttackDataset, self).__init__()
        self.root_path = root_path
        self.v_u = int(ppi / dpi)
        self.img_size = img_size
        genuine_path = os.path.join(self.root_path, "Genuine_Channel")
        self.d_t_paths = self.find_all_images(os.path.join(genuine_path, "Template"))
        self.scan_paths = self.find_all_images(os.path.join(genuine_path, "Scan"))

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def find_all_images(self, data_dir):
        """

        :param data_dir:
        :return:
        """
        image_extensions = ['.bmp', '.png', '.jpg', '.jpeg', '.gif']  # 常见的图像文件格式扩展名
        image_files = []  # 存储找到的图像文件路径
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    if os.path.exists(os.path.abspath(os.path.join(root, file))):
                        image_files.append(os.path.abspath(os.path.join(root, file)))  # 将绝对路径添加到列表
        image_files.sort()
        return image_files

    def __len__(self):
        """

        :return:
        """
        return len(self.d_t_paths)



    def crop_img(self, d_t, scan):
        """

        :param d_t:
        :param scan:
        :return:
        """
        assert scan.shape[0] >= self.img_size and scan.shape[0] >= self.img_size
        d_t = cv2.resize(d_t, dsize=(scan.shape[0], scan.shape[1]), interpolation=cv2.INTER_NEAREST)

        start_h = np.random.choice(np.arange(0, max(1, scan.shape[0] - self.img_size), 2))
        start_w = np.random.choice(np.arange(0, max(1, scan.shape[1] - self.img_size), 2))

        end_h = start_h + self.img_size
        end_w = start_w + self.img_size

        d_t_block = d_t[start_h:end_h, start_w:end_w]
        scan_block = scan[start_h:end_h, start_w:end_w]

        return d_t_block, scan_block


    def __getitem__(self, index):
        """

        :param index:
        :return:
        """
        ''' 打印扫描的正品和拍摄的正品'''
        d_t_path = self.d_t_paths[index]
        scan_path = self.scan_paths[index]
        d_t = cv2.imread(d_t_path, 0)
        scan = cv2.imread(scan_path, 0)
        d_t_block, scan_bock = self.crop_img(d_t, scan)
        d_t_block = cv2.cvtColor(d_t_block, cv2.COLOR_GRAY2RGB)
        scan_bock = cv2.cvtColor(scan_bock, cv2.COLOR_GRAY2RGB)

        d_t_block_out = self.transform(Image.fromarray(np.uint8(d_t_block)))
        scan_block_out = self.transform(Image.fromarray(np.uint8(scan_bock)))
        return d_t_block_out, scan_block_out




class ScanDataset(data.Dataset):
    def __init__(self, root_path, dpi, ppi, img_size):
        super(ScanDataset, self).__init__()
        self.root_path = root_path
        self.v_u = int(ppi / dpi)
        self.img_size = img_size
        genuine_path = os.path.join(self.root_path, "Genuine_Channel")
        genuine_d_t_paths = self.find_all_images(os.path.join(genuine_path, "Template"))
        genuine_scan = self.find_all_images(os.path.join(genuine_path, "Scan"))


        counterfeit_path = os.path.join(self.root_path, "Counterfeit_Channel")
        counterfeit_f_inverse_path = os.path.join(counterfeit_path, "Counterfeit_F_Inverse")
        counterfeit_f_inverse_d_t = self.find_all_images(os.path.join(counterfeit_f_inverse_path, "Template"))
        counterfeit_f_inverse_scan = self.find_all_images(os.path.join(counterfeit_f_inverse_path, "Scan"))

        counterfeit_no_path = os.path.join(counterfeit_path, "Counterfeit_F_Inverse")
        counterfeit_no_d_t = self.find_all_images(os.path.join(counterfeit_no_path, "Template"))
        counterfeit_no_scan = self.find_all_images(os.path.join(counterfeit_no_path, "Scan"))

        counterfeit_otsu_unsharp_path = os.path.join(counterfeit_path, "Counterfeit_Otsu_Unsharp")
        counterfeit_otsu_unsharp_d_t = self.find_all_images(os.path.join(counterfeit_otsu_unsharp_path, "Template"))
        counterfeit_otsu_unsharp_scan = self.find_all_images(os.path.join(counterfeit_otsu_unsharp_path, "Scan"))


        self.ps_t_paths = genuine_d_t_paths + counterfeit_f_inverse_d_t + counterfeit_no_d_t + counterfeit_otsu_unsharp_d_t
        self.scan_paths = genuine_scan + counterfeit_f_inverse_scan + counterfeit_no_scan + counterfeit_otsu_unsharp_scan

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def find_all_images(self, data_dir):
        """

        :param data_dir:
        :return:
        """
        image_extensions = ['.bmp', '.png', '.jpg', '.jpeg', '.gif']  # 常见的图像文件格式扩展名
        image_files = []  # 存储找到的图像文件路径
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    if os.path.exists(os.path.abspath(os.path.join(root, file))):
                        image_files.append(os.path.abspath(os.path.join(root, file)))  # 将绝对路径添加到列表
        image_files.sort()
        return image_files

    def __len__(self):
        """

        :return:
        """
        return len(self.ps_t_paths)


    def crop_img(self, d_t, scan):
        """

        :param d_t:
        :param scan:
        :return:
        """
        assert scan.shape[0] >= self.img_size and scan.shape[0] >= self.img_size
        d_t = cv2.resize(d_t, dsize=(scan.shape[0], scan.shape[1]), interpolation=cv2.INTER_NEAREST)

        start_h = np.random.choice(np.arange(0, max(1, scan.shape[0] - self.img_size), 2))
        start_w = np.random.choice(np.arange(0, max(1, scan.shape[1] - self.img_size), 2))
        end_h = start_h + self.img_size
        end_w = start_w + self.img_size

        d_t_block = d_t[start_h:end_h, start_w:end_w]
        scan_block = scan[start_h:end_h, start_w:end_w]

        return d_t_block, scan_block


    def __getitem__(self, index):
        """

        :param index:
        :return:
        """
        ''' 打印扫描的正品和拍摄的正品'''
        d_t_path = self.ps_t_paths[index]
        scan_path = self.scan_paths[index]
        d_t = cv2.imread(d_t_path, 0)
        scan = cv2.imread(scan_path, 0)
        d_t_block, scan_bock = self.crop_img(d_t, scan)
        d_t_block_out = self.transform(Image.fromarray(np.uint8(d_t_block)))
        scan_block_out = self.transform(Image.fromarray(np.uint8(scan_bock)))
        return d_t_block_out, scan_block_out


class CaptureDataset(data.Dataset):
    def __init__(self, root_path, dpi, ppi, img_size):
        super(CaptureDataset, self).__init__()
        self.root_path = root_path
        self.v_u = int(ppi / dpi)
        self.img_size = img_size
        genuine_path = os.path.join(self.root_path, "Genuine_Channel")
        genuine_d_t_paths = self.find_all_images(os.path.join(genuine_path, "Template"))
        genuine_capture = self.find_all_images(os.path.join(genuine_path, "Capture"))


        counterfeit_path = os.path.join(self.root_path, "Counterfeit_Channel")
        counterfeit_f_inverse_path = os.path.join(counterfeit_path, "Counterfeit_F_Inverse")
        counterfeit_f_inverse_d_t = self.find_all_images(os.path.join(counterfeit_f_inverse_path, "Template"))
        counterfeit_f_inverse_capture = self.find_all_images(os.path.join(counterfeit_f_inverse_path, "Capture"))

        counterfeit_no_path = os.path.join(counterfeit_path, "Counterfeit_F_Inverse")
        counterfeit_no_d_t = self.find_all_images(os.path.join(counterfeit_no_path, "Template"))
        counterfeit_no_capture = self.find_all_images(os.path.join(counterfeit_no_path, "Capture"))

        counterfeit_otsu_unsharp_path = os.path.join(counterfeit_path, "Counterfeit_Otsu_Unsharp")
        counterfeit_otsu_unsharp_d_t = self.find_all_images(os.path.join(counterfeit_otsu_unsharp_path, "Template"))
        counterfeit_otsu_unsharp_capture = self.find_all_images(os.path.join(counterfeit_otsu_unsharp_path, "Capture"))


        self.ps_t_paths = genuine_d_t_paths + counterfeit_f_inverse_d_t + counterfeit_no_d_t + counterfeit_otsu_unsharp_d_t
        self.capture_paths = genuine_capture + counterfeit_f_inverse_capture + counterfeit_no_capture + counterfeit_otsu_unsharp_capture

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def find_all_images(self, data_dir):
        """

        :param data_dir:
        :return:
        """
        image_extensions = ['.bmp', '.png', '.jpg', '.jpeg', '.gif']  # 常见的图像文件格式扩展名
        image_files = []  # 存储找到的图像文件路径
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    if os.path.exists(os.path.abspath(os.path.join(root, file))):
                        image_files.append(os.path.abspath(os.path.join(root, file)))  # 将绝对路径添加到列表
        image_files.sort()
        return image_files

    def __len__(self):
        """

        :return:
        """
        return len(self.ps_t_paths)


    def crop_img(self, d_t, capture):
        """

        :param d_t:
        :param capture:
        :return:
        """
        assert capture.shape[0] >= self.img_size and capture.shape[0] >= self.img_size
        d_t = cv2.resize(d_t, dsize=(capture.shape[0], capture.shape[1]), interpolation=cv2.INTER_NEAREST)

        start_h = np.random.choice(range(0, np.arange(1, max(1, d_t.shape[0] - self.img_size)), 2))
        start_w = np.random.choice(range(0, np.arange(1, max(1, d_t.shape[1] - self.img_size)), 2))
        end_h = start_h + self.img_size
        end_w = start_w + self.img_size

        d_t_block = d_t[start_h:end_h, start_w:end_w]
        capture_block = capture[start_h:end_h, start_w:end_w]

        return d_t_block, capture_block


    def __getitem__(self, index):
        """

        :param index:
        :return:
        """
        ''' 打印扫描的正品和拍摄的正品'''
        d_t_path = self.ps_t_paths[index]
        capture_path = self.capture_paths[index]
        d_t = cv2.imread(d_t_path, 0)
        capture = cv2.imread(capture_path, 0)
        d_t_block, capture_bock = self.crop_img(d_t, capture)
        d_t_block_out = self.transform(Image.fromarray(np.uint8(d_t_block)))
        capture_block_out = self.transform(Image.fromarray(np.uint8(capture_bock)))
        return d_t_block_out, capture_block_out
