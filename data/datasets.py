import glob
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class CustomDataset(Dataset):
    def __init__(self, txt_path, lr_transforms, hr_transforms, hr_transforms2):
        self.transform1 = transforms.Compose(lr_transforms)  # 传入数据预处理
        self.transform2 = transforms.Compose(hr_transforms)
        self.hr_transform2 = transforms.Compose(hr_transforms2)
        with open(txt_path, 'r') as f:
            lines = f.readlines()

        self.img1_list = [i.split()[0] for i in lines]  # 得到所有的serve-ill image name
        self.img2_list = [i.split()[1] for i in lines]

    def __getitem__(self, idx):  # 根据 idx 取出其中一个
        img1 = Image.open(self.img1_list[idx % len(self.img1_list)])
        img2 = Image.open(self.img2_list[idx % len(self.img2_list)])
        img1lr = self.transform1(img1)
        img1hr = self.transform2(img1)
        imggtlr = self.transform2(img2)
        imggthr = self.transform2(img2)
        img_gt256 = self.hr_transform2(img2)
        return {'lrc': img1lr, 'hrc': img1hr, 'lrgt': imggtlr, 'hrgt': imggthr, 'hrgt_256': img_gt256}

    def __len__(self):  # 总数据的多少
        return len(self.img1_list)



class ImageDataset(Dataset):
    def __init__(self, root, lr_transforms, hr_transforms, hr_transforms2):
        self.lr_transform = transforms.Compose(lr_transforms)
        self.hr_transform = transforms.Compose(hr_transforms)
        self.hr_transform2 = transforms.Compose(hr_transforms2)
        self.files = sorted(glob.glob(root + '/*.*'))

    def __getitem__(self, index):
        img = Image.open(self.files[index % len(self.files)])
        imgslr = self.lr_transform(img)
        imgshr = self.hr_transform(img)
        imgs_gt256 = self.hr_transform2(img)
        return {'lrs': imgslr, 'hrs': imgshr, 'hrs256': imgs_gt256}

    def __len__(self):
        return len(self.files)

