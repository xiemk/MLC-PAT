from PIL import Image
from torch.utils.data.dataset import Dataset
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

class COCO2014_handler(Dataset):
    def __init__(self, X, Y, data_path, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform
        self.data_path = data_path

    def __getitem__(self, index):
        x = Image.open(self.data_path+'/'+self.X[index]).convert('RGB')
        x = self.transform(x)
        y = self.Y[index]
        return x, y

    def __len__(self):
        return len(self.X)

class VG256_handler(Dataset):
    def __init__(self, X, Y, data_path, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform
        self.data_path = data_path

    def __getitem__(self, index):
        x = Image.open(self.data_path+'/'+self.X[index]).convert('RGB')
        x = self.transform(x)
        y = self.Y[index]
        return x, y

    def __len__(self):
        return len(self.X)

class VOC2007_handler(Dataset):
	def __init__(self, X, Y, data_path, transform=None):
		self.X = X
		self.Y = Y
		self.transform = transform
		self.data_path = f'{data_path}/VOCdevkit/VOC2007'

	def __getitem__(self, index):
		x = Image.open(self.data_path + '/JPEGImages/' + self.X[index]).convert('RGB')
		x = self.transform(x)        
		y = self.Y[index]
		return x, y

	def __len__(self):
		return len(self.X)