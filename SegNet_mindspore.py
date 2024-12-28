import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.dataset as ds
from mindspore import Tensor, context
import numpy as np
import cv2 as cv
import os

bn_momentum = 0.1  # BN层的momentum

# 编码器
class Encoder(nn.Cell):
    def __init__(self, input_channels):
        super(Encoder, self).__init__()
        self.enco1 = nn.SequentialCellCell([
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, pad_mode="pad", pad_mode="pad", padding=1),
            nn.BatchNorm2d(64, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, pad_mode="pad", pad_mode="pad", padding=1),
            nn.BatchNorm2d(64, momentum=bn_momentum),
            nn.ReLU()
        ])
        self.enco2 = nn.SequentialCellCell([
            nn.Conv2d(64, 128, kernel_size=3, stride=1, pad_mode="pad", pad_mode="pad", padding=1),
            nn.BatchNorm2d(128, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, pad_mode="pad", pad_mode="pad", padding=1),
            nn.BatchNorm2d(128, momentum=bn_momentum),
            nn.ReLU()
        ])
        self.enco3 = nn.SequentialCellCell([
            nn.Conv2d(128, 256, kernel_size=3, stride=1, pad_mode="pad", pad_mode="pad", padding=1),
            nn.BatchNorm2d(256, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, pad_mode="pad", pad_mode="pad", padding=1),
            nn.BatchNorm2d(256, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, pad_mode="pad", pad_mode="pad", padding=1),
            nn.BatchNorm2d(256, momentum=bn_momentum),
            nn.ReLU()
        ])
        self.enco4 = nn.SequentialCellCell([
            nn.Conv2d(256, 512, kernel_size=3, stride=1, pad_mode="pad", pad_mode="pad", padding=1),
            nn.BatchNorm2d(512, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, pad_mode="pad", pad_mode="pad", padding=1),
            nn.BatchNorm2d(512, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, pad_mode="pad", pad_mode="pad", padding=1),
            nn.BatchNorm2d(512, momentum=bn_momentum),
            nn.ReLU()
        ])
        self.enco5 = nn.SequentialCellCell([
            nn.Conv2d(512, 512, kernel_size=3, stride=1, pad_mode="pad", pad_mode="pad", padding=1),
            nn.BatchNorm2d(512, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, pad_mode="pad", pad_mode="pad", padding=1),
            nn.BatchNorm2d(512, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, pad_mode="pad", pad_mode="pad", padding=1),
            nn.BatchNorm2d(512, momentum=bn_momentum),
            nn.ReLU()
        ])

    def construct(self, x):
        id = []
        max_pool_with_argmax = ops.MaxPoolWithArgmax(kernel_size=2, strides=2)

        x = self.enco1(x)
        x, id1 = max_pool_with_argmax(x)
        id.append(id1)
        x = self.enco2(x)
        x, id2 = max_pool_with_argmax(x)
        id.append(id2)
        x = self.enco3(x)
        x, id3 = max_pool_with_argmax(x)
        id.append(id3)
        x = self.enco4(x)
        x, id4 = max_pool_with_argmax(x)
        id.append(id4)
        x = self.enco5(x)
        x, id5 = max_pool_with_argmax(x)
        id.append(id5)

        return x, id


# 编码器+解码器
class SegNet(nn.Cell):
    def __init__(self, input_channels, output_channels):
        super(SegNet, self).__init__()

        self.weights_new = self.state_dict()
        self.encoder = Encoder(input_channels)

        self.deco1 = nn.SequentialCell(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, pad_mode="pad", padding=1),
            nn.BatchNorm2d(512, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, pad_mode="pad", padding=1),
            nn.BatchNorm2d(512, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, pad_mode="pad", padding=1),
            nn.BatchNorm2d(512, momentum=bn_momentum),
            nn.ReLU()
        )
        self.deco2 = nn.SequentialCell(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, pad_mode="pad", padding=1),
            nn.BatchNorm2d(512, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, pad_mode="pad", padding=1),
            nn.BatchNorm2d(512, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, pad_mode="pad", padding=1),
            nn.BatchNorm2d(256, momentum=bn_momentum),
            nn.ReLU()
        )
        self.deco3 = nn.SequentialCell(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, pad_mode="pad", padding=1),
            nn.BatchNorm2d(256, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, pad_mode="pad", padding=1),
            nn.BatchNorm2d(256, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, pad_mode="pad", padding=1),
            nn.BatchNorm2d(128, momentum=bn_momentum),
            nn.ReLU()
        )
        self.deco4 = nn.SequentialCell(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, pad_mode="pad", padding=1),
            nn.BatchNorm2d(128, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, pad_mode="pad", padding=1),
            nn.BatchNorm2d(64, momentum=bn_momentum),
            nn.ReLU()
        )
        self.deco5 = nn.SequentialCell(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, pad_mode="pad", padding=1),
            nn.BatchNorm2d(64, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(64, output_channels, kernel_size=3, stride=1, pad_mode="pad", padding=1),
        )

    def forward(self, x):
        x, id = self.encoder(x)

        x = F.max_unpool2d(x, id[4], kernel_size=2, stride=2)
        x = self.deco1(x)
        x = F.max_unpool2d(x, id[3], kernel_size=2, stride=2)
        x = self.deco2(x)
        x = F.max_unpool2d(x, id[2], kernel_size=2, stride=2)
        x = self.deco3(x)
        x = F.max_unpool2d(x, id[1], kernel_size=2, stride=2)
        x = self.deco4(x)
        x = F.max_unpool2d(x, id[0], kernel_size=2, stride=2)
        x = self.deco5(x)

        return x

    # 删掉VGG-16后面三个全连接层的权重
    def load_weights(self, weights_path):
        weights = torch.load(weights_path)
        del weights["classifier.0.weight"]
        del weights["classifier.0.bias"]
        del weights["classifier.3.weight"]
        del weights["classifier.3.bias"]
        del weights["classifier.6.weight"]
        del weights["classifier.6.bias"]

        names = []
        for key, value in self.encoder.state_dict().items():
            if "num_batches_tracked" in key:
                continue
            names.append(key)

        for name, dict in zip(names, weights.items()):
            self.weights_new[name] = dict[1]

        self.encoder.load_state_dict(self.weights_new)


class DatasetLoader:
    def __init__(self, txt_path):
        # 读取图像和标签路径
        with open(txt_path, "r") as f:
            image_label = [line.strip().split() for line in f]

        self.image_label = image_label

    def __getitem__(self, index):
        # 获取图像和标签路径
        image_path, label_path = self.image_label[index]

        # 加载并预处理图像
        image = cv.imread(image_path)
        image = cv.resize(image, (224, 224))
        image = image / 255.0  # 归一化
        image = np.transpose(image, (2, 0, 1))  # 转换维度 (H, W, C) -> (C, H, W)
        image = Tensor(image, dtype=mindspore.float32)

        # 加载并预处理标签
        label = cv.imread(label_path, 0)  # 加载灰度图
        label = cv.resize(label, (224, 224))
        label = Tensor(label, dtype=mindspore.int32)

        return image, label

    def __len__(self):
        return len(self.image_label)

# 构造 MindSpore 数据集
def create_dataset(txt_path, batch_size=32, shuffle=True):
    dataset = DatasetLoader(txt_path)
    generator = ds.GeneratorDataset(
        source=dataset, 
        column_names=["image", "label"], 
        shuffle=shuffle
    )
    generator = generator.batch(batch_size)
    return generator