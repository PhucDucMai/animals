import os
import cv2
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import VOCDetection
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, RandomAffine, ColorJitter
from PIL import Image
import warnings
warnings.filterwarnings("ignore")


# Bộ dữ liệu CIFAR
class CIFARDataset(Dataset):
	def __init__(self, root="data", train=True, transform=None):
		data_path = os.path.join(root, "cifar-10-batches-py")
		if train:
			data_files = [os.path.join(data_path, "data_batch_{}".format(i)) for i in range(1, 6)]
		else:
			data_files = [os.path.join(data_path, "test_batch")]
		self.images = []
		self.labels = []
		for data_file in data_files:
			with open(data_file, 'rb') as fo:
				data = pickle.load(fo, encoding='bytes')
				self.images.extend(data[b'data'])
				self.labels.extend(data[b'labels'])
		self.transform = transform

	def __len__(self):
		return len(self.labels)

	def __getitem__(self, item):
		image = self.images[item].reshape((3, 32, 32)).astype(np.float32)
		if self.transform:
			image = np.transpose(image, (1, 2, 0))
			image = self.transform(image)
		else:
			image = torch.from_numpy(image)
		label = self.labels[item]
		return image, label


# Bộ dữ liệu Animals
class AnimalDataset(Dataset):
	# Hàm khởi tạo
	def __init__(self, root="data/animals", train=True, transform=None):
		self.categories = ["butterfly", "cat", "chicken", "cow", "dog", "elephant", "horse",
						   "sheep", "spider", "squirrel"]

		''''
			Trong folder animals có 2 thư mục là: train và test
		'''
		if train:
			data_path = os.path.join(root, "train")
		else:
			data_path = os.path.join(root, "test")

		# Lưu đường dẫn của ảnh
		self.image_paths = []
		# Lưu lables của ảnh ở dạng số (chỉ số trong categories)
		self.labels = []
		# ở trong image_paths chỉ thêm path của ảnh chứ không thêm ảnh (ngốn bộ nhớ)
		for category in self.categories:
			# lấy ra đường dẫn tới thư mục các ảnh
			category_path = os.path.join(data_path, category)
			# lấy ra đường dẫn tới từng ảnh
			for image_name in os.listdir(category_path):
				image_path = os.path.join(category_path, image_name)
				self.image_paths.append(image_path)
				# lấy ra index của lable trong categories (label ở dạng số)
				self.labels.append(self.categories.index(category))

		self.transform = transform

	# lấy ra độ dài
	def __len__(self):
		return len(self.labels)

	# trả về image và label của ảnh
	def __getitem__(self, item):
		image = cv2.imread(self.image_paths[item])
		if self.transform:
			image = self.transform(image)
		label = self.labels[item]
		return image, label


if __name__ == '__main__':
	''''
		Compose: Xây dựng 1 chuỗi biến đổi
		ToTenSor(): Chuyển đổi một hình ảnh từ định dạng ảnh về định dạng tensor để tính toán 
		trong deep learning trên GPU
		Resize(): Thay đổi kích thước ảnh
	'''
	transform = Compose([
		ToTensor(),
		Resize((224, 224))
	])

	dataset = AnimalDataset(root="./animals", train=True, transform=transform)
	image, label = dataset.__getitem__(12347)
	dataloader = DataLoader(dataset, batch_size=8, shuffle=True, drop_last=True, num_workers=6)
	for images, labels in dataloader:
		print(images.shape)
		print(labels)
