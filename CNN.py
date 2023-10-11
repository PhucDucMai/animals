import torch
import torch.nn as nn
from torchsummary import summary

''''
	Thường trong 1 khi sẽ có:
	conv -> batch_norm -> activation function
	Xong khối này sẽ cho thêm:
	pooling layer
	
	* Dùng padding trong các layer sẽ đảm bảo được việc kích thước giữ nguyên, tuy nhiên sẽ tốn 
	bộ nhớ. Vì vậy, nếu dùng padding thì sẽ dễ dàng hơn, không phải tính toán nhưng tốn bộ nhớ. 
	Còn nếu không dùng thì sẽ phải tính toán nhưng sẽ đỡ tốn bộ nhớ
'''


class CNN(nn.Module):
	# Dinh nghia xem co nhung layer nao
	def __init__(self, num_classes=10):
		super().__init__()
		self.conv1 = self.makeBlock(in_chanels=3, out_chanels=16)
		self.conv2 = self.makeBlock(in_chanels=16, out_chanels=32)
		self.conv3 = self.makeBlock(in_chanels=32, out_chanels=64)
		self.conv4 = self.makeBlock(in_chanels=64, out_chanels=64)
		self.conv5 = self.makeBlock(in_chanels=64, out_chanels=64)

		# fully connected layer: in_feature: kết quả cuối cùng sau khi flatten
		self.fc1 = nn.Sequential(
			nn.Dropout(0.5),
			nn.Linear(in_features=3136, out_features=1024),
			nn.LeakyReLU()
		)

		self.fc2 = nn.Sequential(
			nn.Dropout(0.5),
			nn.Linear(in_features=1024, out_features=512),
			nn.LeakyReLU()
		)

		self.fc3 = nn.Sequential(
			nn.Dropout(0.5),
			nn.Linear(in_features=512, out_features=num_classes)
		)

	def makeBlock(self, in_chanels, out_chanels):
		return nn.Sequential(
			# conv2d phai chi ra 3 thu: in_chanels , out_channels, kernel_size
			nn.Conv2d(in_channels=in_chanels, out_channels=out_chanels, kernel_size=3, padding=1),

			# giữa conv và action nên có thêm 1 layer(batchnorm) để tránh overfitting
			nn.BatchNorm2d(num_features=out_chanels),

			# activation không thay đổi kích thước, chỉ thay đổi giá trị
			nn.LeakyReLU(),

			# conv2d phai chi ra 3 thu: in_chanels , out_channels, kernel_size
			nn.Conv2d(in_channels=out_chanels, out_channels=out_chanels, kernel_size=3, padding=1),

			# giữa conv và action nên có thêm 1 layer(batchnorm) để tránh overfitting
			nn.BatchNorm2d(num_features=out_chanels),

			# activation không thay đổi kích thước, chỉ thay đổi giá trị
			nn.LeakyReLU(),

			# pooling layer
			nn.MaxPool2d(kernel_size=2),
		)

	# Cac layer ket hop voi nhau ra sao
	def forward(self, x):
		x = self.conv1(x)
		x = self.conv2(x)
		x = self.conv3(x)
		x = self.conv4(x)
		x = self.conv5(x)
		batch_size, chanels, height, width = x.shape
		x = x.view(batch_size, -1)
		x = self.fc1(x)
		x = self.fc2(x)
		x = self.fc3(x)
		return x


if __name__ == '__main__':
	model = CNN()
	model.train()
	sample_input = torch.rand(2, 3, 224, 224)
	result = model(sample_input)
	print(result.shape)

