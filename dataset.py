import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import os


data_augmentation = transforms.Compose([
    transforms.RandomResizedCrop(size=224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0))], p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_positive_pair(image):
    pos_1 = data_augmentation(image)
    pos_2 = data_augmentation(image)
    # import pdb;pdb.set_trace()
    # if pos_1.shape[0]==1:
    #     pos_1 = pos_1.repeat(3,1,1)
    # if pos_2.shape[0]==1:
    #     pos_2 = pos_2.repeat(3,1,1)
    return pos_1, pos_2

class dataset:
    def __init__(self,root='./dataset/train.txt',num=10000):
        self.root = root
        self.lis = self.prepare_lis()
        self.max_len = len(self.lis)
        self.num = min(num,self.max_len)

    def prepare_lis(self):
        root = self.root
        with open(root, 'r') as file:
            lis = [int(line.strip()) for line in file]
        return lis
    
    def __len__(self):
        return self.num
    
    def __getitem__(self,idx):
        assert 0<=idx<self.num
        image_name = os.path.join('./dataset/imagenet_val',f'ILSVRC2012_val_{self.lis[idx]:08d}.JPEG')
        # import pdb;pdb.set_trace()
        image = Image.open(image_name)
        image = image.convert('RGB')
        pos_1, pos_2 = get_positive_pair(image)
        return (pos_1,pos_2)

# dataset = dataset()

# image = Image.open('/home/tanweipeng/work/work3/dataset/imagenet_val/ILSVRC2012_val_00000002.JPEG')


# pos_1, pos_2 = get_positive_pair(image)

# def show_image(tensor, title=""):
#     image = tensor.permute(1, 2, 0).numpy()
#     mean = [0.485, 0.456, 0.406]
#     std = [0.229, 0.224, 0.225]
#     image = image * std + mean
#     image = image.clip(0, 1)
#     plt.imshow(image)
#     plt.title(title)
#     plt.axis('off')

# plt.figure(figsize=(8, 4))
# plt.subplot(1, 2, 1)
# show_image(pos_1, title="Positive Sample 1")
# plt.subplot(1, 2, 2)
# show_image(pos_2, title="Positive Sample 2")

# plt.savefig('test.jpg',format='jpg')