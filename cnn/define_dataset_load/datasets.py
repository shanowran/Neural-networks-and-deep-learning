"""
@Time ： 2022/10/23 15:25
@Auth ： Luminous
@Contact:  3199409618@qq.com
@File ：datasets.py
@IDE ：PyCharm
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms
import PIL.Image
import os,glob
import random,csv
import torch.utils.data as datasets
from torch.utils.data import DataLoader


class numberdatasets(datasets.Dataset):
    """
    load dataset
    """
    def __init__(self, root, resize, mode):
        super(numberdatasets, self).__init__()
        self.root = root
        self.resize = resize
        self.mode = mode
        self.name2label = {}
        for name in sorted(os.listdir(os.path.join(root))):
            if not os.path.isdir(os.path.join(root, name)):
                continue
            self.name2label[name] = len(self.name2label.keys())
        self.images, self.labels = self.imagesname('images.csv')
        if mode == "train":
            self.images = self.images[:int(0.6 * len(self.images))]
            self.labels = self.labels[:int(0.6 * len(self.labels))]
        elif mode == "var":
            self.images = self.images[int(0.6 * len(self.images)):int(0.6 * len(self.images) + 0.2 * len(self.images))]
            self.labels = self.labels[int(0.6 * len(self.labels)):int(0.6 * len(self.labels) + 0.2 * len(self.labels))]
        else:
            self.images = self.images[int(0.6 * len(self.images) + 0.2 * len(self.images)):]
            self.labels = self.labels[int(0.6 * len(self.labels) + 0.2 * len(self.labels)):]

    def imagesname(self, filename):
        if not os.path.exists(os.path.join(self.root, filename)):
            images = []  # 存储每张图片篇路径
            for name in self.name2label.keys():
                images = images + glob.glob(os.path.join(self.root, name, '*.png'))
                images = images + glob.glob(os.path.join(self.root, name, '*.jpg'))
                images = images + glob.glob(os.path.join(self.root, name, '*.jpeg'))

            random.shuffle(images)
            with open(os.path.join(self.root, filename), mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                for imgs in images:
                    name = imgs.split(os.sep)[-2]
                    label = self.name2label[name]
                    writer.writerow([imgs, label])
                print("write into file:", os.path.join(self.root, filename))
        images1, labels1 = [], []
        with open(os.path.join(self.root, filename), encoding="utf-8") as F:
            reader = csv.reader(F)
            for row in reader:
                img, label1 = row
                label1 = int(label1)
                images1.append(img)
                labels1.append(label1)
            assert len(images1) == len(labels1)
            return images1, labels1

    def denormalize(self, x_hat):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        mean = torch.tensor(mean).unsqueeze(1).unsqueeze(1)
        std = torch.tensor(std).unsqueeze(1).unsqueeze(1)
        x = x_hat * std + mean
        return x

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img, label = self.images[idx], self.labels[idx]
        tranform = torchvision.transforms.Compose([
            lambda x:PIL.Image.open(img).convert("RGB"),
            torchvision.transforms.Resize((int(self.resize*1.25), int(self.resize*1.25))),
            torchvision.transforms.RandomRotation(15),
            torchvision.transforms.CenterCrop(self.resize),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225]),  # imagenet数据集总结出来的图片均值以及方差，可以直接用
        ])

        img = tranform(img)
        label = torch.tensor(label)
        return img, label


def main():

    import visdom
    import time
    root = "D:\卷积神经网络\data\pokeman"
    viz = visdom.Visdom()
    db = numberdatasets(root, 224, "train")
    x, y = next(iter(db))
    print("sample:", x.size(), y.size(), y)
    viz.images(db.denormalize(x), win="sample_x", opts=dict(title="sample_x"))
    loder = DataLoader(db, batch_size=32, shuffle=True)  # 取多张图片，一次32张图片
    for x, y in loder:
        viz.images(db.denormalize(x), win="batch", nrow=8, opts=dict(title='batch'))
        viz.text(str(y.numpy()), win="label", opts=dict(title='batch'))
        time.sleep(10)


if __name__ == '__main__':
    main()


# root = "D:\卷积神经网络\data\pokeman"
# label = numberdatasets(root, 224, "train")
# print(label.name2label)
