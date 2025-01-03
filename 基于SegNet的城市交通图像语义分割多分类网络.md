# 基于SegNet的城市交通图像语义分割多分类网络

<center>陈佳辉（02） 陈奕岐（02） 任思羽（04）</center>

## 实验简介

Cityscapes 是一个广泛用于计算机视觉领域的高质量数据集，专注于城市环境中的语义理解任务，特别是在自动驾驶和智能交通系统的研究中具有重要的应用价值。该数据集包含高分辨率的城市街景图像以及像素级精确标注，主要用于语义分割、实例分割、目标检测和场景理解等任务。Cityscapes 的数据采集覆盖了 50 座欧洲城市，涵盖不同的天气、季节和光照条件，以确保数据的多样性和模型的泛化能力。

这个数据集提供了 2048x1024 像素分辨率的图像，并定义了 30 个类别（如道路、建筑、行人、车辆等），其中 19 个类别被用于语义分割的评估任务。它包含 2975 张训练图像、500 张验证图像和 1525 张测试图像，其中测试集标注并未公开，需提交到官方测试服务器进行评估。Cityscapes 的标注不仅包括静态场景的语义信息，还提供了实例级的分割标注，部分数据甚至包含视频序列，适合用于动态场景分析和时序预测研究。

![image-20241230011921675](https://cdn.jsdelivr.net/gh/Jafekin/markdown_images@main/img/image-20241230011921675.png)

![image-20241230011942035](https://cdn.jsdelivr.net/gh/Jafekin/markdown_images@main/img/image-20241230011942035.png)

由于Cityscapes进行语义分割的先天优势，我们选择基于**MinSpore**使用一种经典的图像分割的**深度卷积编码器-解码器架构——SegNet**来对该数据进行学习，期望得到一个**语意分割多分类网络模型**。

**SegNet** 是一种经典的卷积神经网络（CNN）架构，主要用于 **语义分割任务**。它的设计重点是高效地提取特征并实现高质量的像素级分割，同时保证计算复杂度适中。SegNet 是基于全卷积网络（Fully Convolutional Network, FCN）的改进，并且常结合 VGG 网络的编码结构，形成编码器-解码器（Encoder-Decoder）的分割架构。

SegNet 的编码器部分借鉴了 VGG16 网络的前 13 层卷积网络，用于提取图像的高层语义特征，同时通过最大池化逐步降低特征图的分辨率。与传统的卷积网络不同，SegNet 在最大池化操作中保存了**池化索引**（Pooling Indices），即每次池化时的最大值位置，这些索引将在解码过程中被用于上采样操作。
 SegNet 在解码过程逐层恢复特征图的空间分辨率，与编码器结构对称。通过利用编码器中记录的池化索引，解码器能够准确地进行上采样，从而高效地恢复图像的空间结构。这种方法不仅避免了使用参数更多、计算量更大的转置卷积，同时也保留了关键的空间信息，使得解码过程更加高效和准确。解码器恢复的特征图经过卷积层进一步细化，最终通过一个 1×1 的卷积层输出每个像素对应的分类结果，实现像素级的语义分割。

![img](https://cdn.jsdelivr.net/gh/Jafekin/markdown_images@main/img/2a9f057e469374eda4088e8562bcc5de.png)

##  数据预处理

### 制作cityscapes标签数据集

首先在cityscapesscripts里的**label**文件里选好要用的分类，原始分类给的19类。在不感兴趣的类别`trainid`那栏改为-1（网络上查询的资料均为改为255，实际测试时会作为255类参与训练）、不感兴趣的`ignoreInEval`改为True， 然后重新按顺序将感兴趣的分类进行排序， 如下图：

![image-20241230015053504](/Users/jafekin/Library/Application Support/typora-user-images/image-20241230015053504.png)

最后使用cityscapesscripts里的**createlabelidimg**脚本将json文件生成trainid.png文件。

![img](https://cdn.jsdelivr.net/gh/Jafekin/markdown_images@main/img/42e9d6af4ea2628b9ed40fd9d145ed74.png)

### 数据集及标注路径构造

为了准备语义分割任务的训练数据，我们编写了一段 Python 脚本用于生成数据集路径对的文件。该脚本基于 Cityscapes 数据集的目录结构，自动提取训练集、验证集和测试集的图像路径以及对应的标注路径，并按照指定格式将这些路径保存到文本文件中，为后续模型训练提供方便的数据输入方式。下面仅给出训练集的代码供参考：

```python
def make_train_txt(num):
    global i
    paths = glob.glob(r"/root/autodl-tmp/leftImg8bit/train/*/*")
    txt = open("./dataset-list/train.txt", "w")

    for path in paths:
        data = path + " " + path.replace("leftImg8bit", "gtFine").replace(
            "gtFine.png", "gtFine_labelTrainIds.png") + "\n"
        txt.write(data)
        i = i + 1
        if i == num:
            break
    i = 0
    txt.close()
```

假设 `cityscapes/leftImg8bit/train/city1/image1.png` 是训练集中某一张图像，则对应的标注路径为 `cityscapes/gtFine/train/city1/image1_gtFine_labelTrainIds.png`。程序生成的 train.txt 文件中将包含以下内容：

```shell
cityscapes/leftImg8bit/train/city1/image1.png cityscapes/gtFine/train/city1/image1_gtFine_labelTrainIds.png
```

## 模型设计

### 导入必要的包

```python
# 导入 MindSpore 主模块
import mindspore as ms
# 导入 MindSpore 的神经网络模块，用于定义和构建模型
import mindspore.nn as nn
# 导入 MindSpore 的数据集模块，用于加载和处理数据集
import mindspore.dataset as ds
# 从 MindSpore 中导入操作、张量和上下文模块
from mindspore import ops, Tensor, context
# 导入 MindSpore 的训练回调模块，用于保存检查点文件和监控损失
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor
# 导入 Model 类，用于封装训练和推理的功能
from mindspore.train import Model

# 导入其他通用模块
import time  # 用于记录和测量时间
import numpy as np  # 用于数值计算
import os  # 用于文件和目录操作
from matplotlib import pyplot as plt  # 用于绘制图像
from PIL import Image  # 用于加载和处理图像文件

# 导入自定义的 SegNet 模型和 create_dataset 函数
from SegNet import create_dataset, SegNet
```

### 构造DataLoader

```python
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
```

### 语义分割中的类别不平衡的权重计算

在语义分割任务中，类别不平衡是一个常见的问题，尤其是在处理真实世界的数据集（如 Cityscapes）时。一些类别（如道路、建筑）在图像中占据了大量的像素，而其他类别（如行人、交通标志）则仅占据少量像素。这种类别分布的不均衡会导致模型更倾向于预测频繁出现的类别，从而对稀有类别的分割效果较差。为了解决这一问题，通常会为每个类别分配一个权重，用于在计算损失函数时对稀有类别赋予更高的关注度。这种方法称为类别权重计算。

**中位频率权重（Median Frequency Balancing）** 是一种在语义分割任务中应对类别不平衡问题的权重计算方法。它通过基于数据集中类别像素的分布来动态调整每个类别的权重，平衡频繁类别和稀有类别对损失函数的贡献，从而提升模型对稀有类别的预测能力。具体来说，中位频率权重的核心思想是利用类别频率的中位数作为基准，对每个类别的权重进行缩放。权重的计算公式为：
$$
w_c = \frac{\text{Median}(f)}{f_c}
$$
其中， $f_c$ 表示类别 $c$ 的像素频率，即类别 $c$ 的像素占数据集中所有像素的比例， $\text{Median}(f)$ 是所有类别频率的中位数。通过这种方式，类别的权重与其像素频率成反比。稀有类别（像素频率较低）的权重会比中位数大，从而在计算损失时获得更高的关注度。而对于频繁类别（像素频率较高），其权重会被压缩到小于 1，使得它们对损失的贡献相对减小。下面是具体实现代码：

```python
paths = open("./data-list/train.txt", "r")

CLASS_NUM = 19
SUM = [[] for i in range(CLASS_NUM)]
SUM_ = 0

for line in paths:
    line.rstrip("\n")
    line.lstrip("\n")
    path = line.split()
    img = cv.imread(path[1], 0)
    img_np = np.array(img)
    for i in range(CLASS_NUM):
        SUM[i].append(np.sum((img_np == i)))

for index, iter in enumerate(SUM):
    print("类别{}的数量：".format(index), sum(iter))

for iter in SUM:
    SUM_ += sum(iter)

median = 1/CLASS_NUM

for index, iter in enumerate(SUM):
    print(f"{median/(sum(iter)/SUM_)}, ")
```

### 设置超参数

~~~python
CLASS_NUM = 19  # 分类数量，Cityscapes 数据集中定义了 19 个类别
CATE_WEIGHT = [1.0] * CLASS_NUM  # 初始类别权重，每个类别初始权重设为 1.0
EPOCH = 20  # 训练轮数，决定模型在数据集上训练的次数
BATCH_SIZE = 4  # 每次训练的批量大小，用于控制模型训练的显存占用和梯度更新频率
LR = 0.01  # 学习率，控制梯度更新的步长大小
MOMENTUM = 0.9  # 动量系数，用于加速梯度下降并抑制震荡

# 加权类别权重（CATE_WEIGHT）
CATE_WEIGHT = [
    0.11921289124514069, 0.9772031489113517, 0.2606578051907899,
    9.068186030082103, 6.772279222279968, 4.845227365263553,
    28.52810833819015, 10.758335113118157, 0.3736856892826064,
    5.133604194351756, 1.4827554927399786, 4.886283011781665,
    44.10664265269921, 0.8495922090922964, 22.22468639649049,
    25.267384685741828, 25.526398354063044, 60.29661028702076,
    14.370822828405153
]  # 预先计算的类别权重，每个值对应一个类别，用于处理类别不平衡问题

# 数据路径与模型参数
TXT_PATH = "./dataset-list/train.txt"  # 训练数据列表文件路径，存储图像与标注文件的对应关系
PRE_TRAINING = "vgg16_bn-6c64b313.pth"  # 预训练模型路径，用于初始化网络权重
WEIGHTS = "./weights/"  # 模型训练过程中保存权重的目录

# 确保权重保存路径存在
# 如果权重保存目录不存在，则自动创建
if not os.path.exists(WEIGHTS):
    os.makedirs(WEIGHTS)

# 加载训练数据集
# 调用 create_dataset 函数，基于 train.txt 提供的数据路径和指定的批量大小加载训练数据
train_data = create_dataset(txt_path=TXT_PATH, batch_size=BATCH_SIZE)
~~~

### 编码器网络

在语义分割任务中，编码器的主要功能是从输入图像中提取高层语义特征，同时逐步降低特征图的空间分辨率，以捕获更大的感受野。本文实验中的编码器基于卷积神经网络（CNN），设计参考了 VGG 网络结构，利用堆叠的卷积层和最大池化操作实现特征提取和降采样。

编码器由 5 个阶段组成（enco1 至 enco5），每个阶段包含多个卷积层，采用批归一化（Batch Normalization）和 ReLU 激活函数，以提升训练稳定性和模型的非线性表达能力。阶段之间通过 MaxPoolWithArgmax 操作实现下采样，并记录池化索引，以便后续解码器中进行上采样。

**(1) 核心模块说明**

- **卷积层（Conv2d）**：每个卷积操作使用固定的 3 $\times$ 3 卷积核，步长为 1，填充模式为 “pad”，保证特征图尺寸保持不变。
- **批归一化层（BatchNorm2d）**：用于归一化卷积输出，稳定训练过程，加速收敛。
- **激活函数（ReLU）**：引入非线性能力，使网络能够学习更复杂的特征。
- **最大池化（MaxPoolWithArgmax）**：实现降采样，同时记录最大值索引，用于后续解码器的上采样。

**(2) 阶段结构**

每个阶段包含以下内容：

1. 第一阶段（enco1）：输入通道数为 input_channels，输出通道数为 64，两次卷积。
2. 第二阶段（enco2）：输入通道数为 64，输出通道数为 128，两次卷积。
3. 第三阶段（enco3）：输入通道数为 128，输出通道数为 256，三次卷积。
4. 第四阶段（enco4）：输入通道数为 256，输出通道数为 512，三次卷积。
5. 第五阶段（enco5）：输入通道数为 512，输出通道数为 512，三次卷积。

以下是编码器的关键实现代码及其解释：

```python
class Encoder(nn.Cell):
    def __init__(self, input_channels):
        super(Encoder, self).__init__()
        
        # 第一阶段：输入通道 -> 64，包含两次卷积、批归一化和激活函数
        self.enco1 = nn.SequentialCell([
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, pad_mode="pad", padding=1),  # 第一次卷积
            nn.BatchNorm2d(64, momentum=bn_momentum),  # 批归一化
            nn.ReLU(),  # 激活函数
            nn.Conv2d(64, 64, kernel_size=3, stride=1, pad_mode="pad", padding=1),  # 第二次卷积
            nn.BatchNorm2d(64, momentum=bn_momentum),  # 批归一化
            nn.ReLU()  # 激活函数
        ])
        
        # 第二阶段：64 -> 128，两次卷积
        self.enco2 = nn.SequentialCell([
            nn.Conv2d(64, 128, kernel_size=3, stride=1, pad_mode="pad", padding=1),  # 第一次卷积
            nn.BatchNorm2d(128, momentum=bn_momentum),  # 批归一化
            nn.ReLU(),  # 激活函数
            nn.Conv2d(128, 128, kernel_size=3, stride=1, pad_mode="pad", padding=1),  # 第二次卷积
            nn.BatchNorm2d(128, momentum=bn_momentum),  # 批归一化
            nn.ReLU()  # 激活函数
        ])
        
        # 第三阶段：128 -> 256，包含三次卷积
        self.enco3 = nn.SequentialCell([
            nn.Conv2d(128, 256, kernel_size=3, stride=1, pad_mode="pad", padding=1),  # 第一次卷积
            nn.BatchNorm2d(256, momentum=bn_momentum),  # 批归一化
            nn.ReLU(),  # 激活函数
            nn.Conv2d(256, 256, kernel_size=3, stride=1, pad_mode="pad", padding=1),  # 第二次卷积
            nn.BatchNorm2d(256, momentum=bn_momentum),  # 批归一化
            nn.ReLU(),  # 激活函数
            nn.Conv2d(256, 256, kernel_size=3, stride=1, pad_mode="pad", padding=1),  # 第三次卷积
            nn.BatchNorm2d(256, momentum=bn_momentum),  # 批归一化
            nn.ReLU()  # 激活函数
        ])
        
        # 第四阶段：256 -> 512，包含三次卷积
        self.enco4 = nn.SequentialCell([
            nn.Conv2d(256, 512, kernel_size=3, stride=1, pad_mode="pad", padding=1),  # 第一次卷积
            nn.BatchNorm2d(512, momentum=bn_momentum),  # 批归一化
            nn.ReLU(),  # 激活函数
            nn.Conv2d(512, 512, kernel_size=3, stride=1, pad_mode="pad", padding=1),  # 第二次卷积
            nn.BatchNorm2d(512, momentum=bn_momentum),  # 批归一化
            nn.ReLU(),  # 激活函数
            nn.Conv2d(512, 512, kernel_size=3, stride=1, pad_mode="pad", padding=1),  # 第三次卷积
            nn.BatchNorm2d(512, momentum=bn_momentum),  # 批归一化
            nn.ReLU()  # 激活函数
        ])
        
        # 第五阶段：512 -> 512，包含三次卷积
        self.enco5 = nn.SequentialCell([
            nn.Conv2d(512, 512, kernel_size=3, stride=1, pad_mode="pad", padding=1),  # 第一次卷积
            nn.BatchNorm2d(512, momentum=bn_momentum),  # 批归一化
            nn.ReLU(),  # 激活函数
            nn.Conv2d(512, 512, kernel_size=3, stride=1, pad_mode="pad", padding=1),  # 第二次卷积
            nn.BatchNorm2d(512, momentum=bn_momentum),  # 批归一化
            nn.ReLU(),  # 激活函数
            nn.Conv2d(512, 512, kernel_size=3, stride=1, pad_mode="pad", padding=1),  # 第三次卷积
            nn.BatchNorm2d(512, momentum=bn_momentum),  # 批归一化
            nn.ReLU()  # 激活函数
        ])

    def construct(self, x):
        id = []  # 用于存储每次池化操作的索引
        max_pool_with_argmax = ops.MaxPoolWithArgmax(kernel_size=2, strides=2)  # 定义带索引的最大池化操作

        # 第一阶段：卷积和池化
        x = self.enco1(x)  # 第一阶段卷积
        x, id1 = max_pool_with_argmax(x)  # 第一阶段池化并保存索引
        id.append(id1)  # 保存索引到列表

        # 第二阶段：卷积和池化
        x = self.enco2(x)  # 第二阶段卷积
        x, id2 = max_pool_with_argmax(x)  # 第二阶段池化并保存索引
        id.append(id2)  # 保存索引到列表

        # 第三阶段：卷积和池化
        x = self.enco3(x)  # 第三阶段卷积
        x, id3 = max_pool_with_argmax(x)  # 第三阶段池化并保存索引
        id.append(id3)  # 保存索引到列表

        # 第四阶段：卷积和池化
        x = self.enco4(x)  # 第四阶段卷积
        x, id4 = max_pool_with_argmax(x)  # 第四阶段池化并保存索引
        id.append(id4)  # 保存索引到列表

        # 第五阶段：卷积和池化
        x = self.enco5(x)  # 第五阶段卷积
        x, id5 = max_pool_with_argmax(x)  # 第五阶段池化并保存索引
        id.append(id5)  # 保存索引到列表

        return x, id  # 输出最后的特征图和池化索引
```

### 解码器网络

SegNet 是一种经典的编码器-解码器结构的语义分割模型，其解码器通过逐步恢复编码器提取的特征图，实现从压缩语义特征到高分辨率分割结果的转化。解码器的核心设计在于使用编码器中最大池化的索引（Pooling Indices）进行上采样，保证空间信息的有效恢复，并通过一系列卷积和非线性操作对特征进行精细化。适用于实际应用。

解码器设计与编码器对称，包括五个阶段（deco1 至 deco5），每个阶段通过一组卷积层和非线性激活函数对上采样后的特征进行处理。解码过程逐步恢复特征图的空间分辨率，最终输出与输入图像大小一致的分割结果。

**(1) 设计原理**

- **逐步上采样**：解码器使用编码器的最大池化索引通过 MaxUnpool2d 进行逐步上采样，确保空间分辨率与输入一致。
- 每次上采样后，通过卷积操作对特征进行精细化处理。
- **特征恢复**：卷积操作用于增强上采样后的特征表征能力，同时减少通道数，使得解码器的输出与类别数匹配。
- **非线性激活**：使用 ReLU 激活函数提升模型的非线性表达能力。
- **参数共享与对称性**：解码器的参数设计与编码器对称，确保特征提取与恢复具有连续性和一致性。

**(2) 解码器各阶段功能**

1. **第一阶段（**deco1**）**：输入为编码器第 5 阶段的输出，通道数从 512 恢复至 512。
2. **第二阶段（**deco2**）**：输入为 deco1 的输出，通道数从 512 降至 256。
3. **第三阶段（**deco3**）**：输入为 deco2 的输出，通道数从 256 降至 128。
4. **第四阶段（**deco4**）**：输入为 deco3 的输出，通道数从 128 降至 64。
5. **第五阶段（**deco5**）**：输入为 deco4 的输出，通道数从 64 降至最终类别数。

以下是解码器的核心实现代码及其详细解析：

```python
class SegNet(nn.Cell):
    def __init__(self, input_channels, output_channels):
        super(SegNet, self).__init__()

        # 保存模型的初始状态
        self.weights_new = self.state_dict()

        # 初始化编码器部分
        self.encoder = Encoder(input_channels)  # 调用前面定义的 Encoder

        # 解码器的第一个阶段（对应编码器的第五阶段，512 -> 512）
        self.deco1 = nn.SequentialCell([
            nn.Conv2d(512, 512, kernel_size=3, stride=1, pad_mode="pad", padding=1),  # 第一次卷积
            nn.BatchNorm2d(512, momentum=bn_momentum),  # 批归一化
            nn.ReLU(),  # 激活函数
            nn.Conv2d(512, 512, kernel_size=3, stride=1, pad_mode="pad", padding=1),  # 第二次卷积
            nn.BatchNorm2d(512, momentum=bn_momentum),  # 批归一化
            nn.ReLU(),  # 激活函数
            nn.Conv2d(512, 512, kernel_size=3, stride=1, pad_mode="pad", padding=1),  # 第三次卷积
            nn.BatchNorm2d(512, momentum=bn_momentum),  # 批归一化
            nn.ReLU()  # 激活函数
        ])

        # 解码器的第二阶段（对应编码器的第四阶段，512 -> 256）
        self.deco2 = nn.SequentialCell([
            nn.Conv2d(512, 512, kernel_size=3, stride=1, pad_mode="pad", padding=1),  # 第一次卷积
            nn.BatchNorm2d(512, momentum=bn_momentum),  # 批归一化
            nn.ReLU(),  # 激活函数
            nn.Conv2d(512, 512, kernel_size=3, stride=1, pad_mode="pad", padding=1),  # 第二次卷积
            nn.BatchNorm2d(512, momentum=bn_momentum),  # 批归一化
            nn.ReLU(),  # 激活函数
            nn.Conv2d(512, 256, kernel_size=3, stride=1, pad_mode="pad", padding=1),  # 第三次卷积，输出通道降为 256
            nn.BatchNorm2d(256, momentum=bn_momentum),  # 批归一化
            nn.ReLU()  # 激活函数
        ])

        # 解码器的第三阶段（对应编码器的第三阶段，256 -> 128）
        self.deco3 = nn.SequentialCell([
            nn.Conv2d(256, 256, kernel_size=3, stride=1, pad_mode="pad", padding=1),  # 第一次卷积
            nn.BatchNorm2d(256, momentum=bn_momentum),  # 批归一化
            nn.ReLU(),  # 激活函数
            nn.Conv2d(256, 256, kernel_size=3, stride=1, pad_mode="pad", padding=1),  # 第二次卷积
            nn.BatchNorm2d(256, momentum=bn_momentum),  # 批归一化
            nn.ReLU(),  # 激活函数
            nn.Conv2d(256, 128, kernel_size=3, stride=1, pad_mode="pad", padding=1),  # 第三次卷积，输出通道降为 128
            nn.BatchNorm2d(128, momentum=bn_momentum),  # 批归一化
            nn.ReLU()  # 激活函数
        ])

        # 解码器的第四阶段（对应编码器的第二阶段，128 -> 64）
        self.deco4 = nn.SequentialCell([
            nn.Conv2d(128, 128, kernel_size=3, stride=1, pad_mode="pad", padding=1),  # 第一次卷积
            nn.BatchNorm2d(128, momentum=bn_momentum),  # 批归一化
            nn.ReLU(),  # 激活函数
            nn.Conv2d(128, 64, kernel_size=3, stride=1, pad_mode="pad", padding=1),  # 第二次卷积，输出通道降为 64
            nn.BatchNorm2d(64, momentum=bn_momentum),  # 批归一化
            nn.ReLU()  # 激活函数
        ])

        # 解码器的第五阶段（对应编码器的第一阶段，64 -> 输出通道）
        self.deco5 = nn.SequentialCell([
            nn.Conv2d(64, 64, kernel_size=3, stride=1, pad_mode="pad", padding=1),  # 第一次卷积
            nn.BatchNorm2d(64, momentum=bn_momentum),  # 批归一化
            nn.ReLU(),  # 激活函数
            nn.Conv2d(64, output_channels, kernel_size=3, stride=1, pad_mode="pad", padding=1),  # 最后一层卷积，输出通道为目标类别数
        ])

    # 前向传播函数
    def forward(self, x):
        # 编码阶段，获取编码器输出的特征图和池化索引
        x, id = self.encoder(x)

        # 解码阶段，逐步进行上采样和特征恢复
        x = F.max_unpool2d(x, id[4], kernel_size=2, stride=2)  # 使用第五阶段的池化索引进行上采样
        x = self.deco1(x)  # 第一阶段解码

        x = F.max_unpool2d(x, id[3], kernel_size=2, stride=2)  # 使用第四阶段的池化索引进行上采样
        x = self.deco2(x)  # 第二阶段解码

        x = F.max_unpool2d(x, id[2], kernel_size=2, stride=2)  # 使用第三阶段的池化索引进行上采样
        x = self.deco3(x)  # 第三阶段解码

        x = F.max_unpool2d(x, id[1], kernel_size=2, stride=2)  # 使用第二阶段的池化索引进行上采样
        x = self.deco4(x)  # 第四阶段解码

        x = F.max_unpool2d(x, id[0], kernel_size=2, stride=2)  # 使用第一阶段的池化索引进行上采样
        x = self.deco5(x)  # 最终解码，生成分割结果

        return x  # 返回最终的分割结果

    # 加载预训练权重，同时移除全连接层的权重
    def load_weights(self, weights_path):
        weights = torch.load(weights_path)  # 加载预训练权重

        # 删除 VGG-16 全连接层的权重
        del weights["classifier.0.weight"]
        del weights["classifier.0.bias"]
        del weights["classifier.3.weight"]
        del weights["classifier.3.bias"]
        del weights["classifier.6.weight"]
        del weights["classifier.6.bias"]

        # 匹配编码器的权重
        names = []
        for key, value in self.encoder.state_dict().items():
            if "num_batches_tracked" in key:  # 跳过不需要的参数
                continue
            names.append(key)

        # 加载对应的权重到编码器
        for name, dict in zip(names, weights.items()):
            self.weights_new[name] = dict[1]

        self.encoder.load_state_dict(self.weights_new)  # 更新编码器的权重
```

## 训练与评估

### 训练

SegNet在损失函数选取上较为简单，采用交叉熵的方式进行损失函数的计算，交叉熵损失函数**适用于多分类问题**，对概率分布较为敏感，有利于梯度下降。

$$
L=\frac{1}{N}\sum_iL_i=-\frac{1}{N}\sum_i\sum_{c=1}^My_{ic}\log(p_{ic})
$$
训练代码如下，我们在这里迭代20次：

~~~python
def train(SegNet, train_data):

    # 加载VGG16预训练权重减少计算量
    SegNet.load_weights(PRE_TRAINING)

    # 构造数据集
    train_dataset = ds.GeneratorDataset(
        source=train_data, column_names=["image", "label"], shuffle=True
    )
    train_dataset = train_dataset.batch(BATCH_SIZE)

    # 定义优化器
    optimizer = nn.SGD(SegNet.trainable_params(),
                       learning_rate=LR, momentum=MOMENTUM)

    # 定义损失函数
    weight = Tensor(np.array(CATE_WEIGHT).astype(np.float32))
    loss_func = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")

    # 保存训练 Loss
    losses = []

    print("Start Training...")
    for epoch in range(EPOCH):
        for step, data in enumerate(train_dataset.create_dict_iterator()):
            b_x = data["image"]
            b_y = data["label"]
            b_y = ops.Reshape()(b_y, (BATCH_SIZE, 224, 224))  # 确保标签形状正确

            # 前向计算和梯度更新
            output = SegNet(b_x)
            loss = loss_func(output, b_y)
            losses.append(loss.asnumpy().item())

            optimizer.clear_grad()  # 梯度清零
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

            if step % 100 == 0:
                print(f"Epoch: {epoch} || Step: {
                      step} || Loss: {loss.asnumpy():.4f}")

    # 保存模型权重
    save_path = WEIGHTS + "SegNet_weights.ckpt"
    ms.save_checkpoint(SegNet, save_path)
    print(f"Model saved to {save_path}")

    return losses

# 初始化模型并开始训练
SegNet = SegNet(3, CLASS_NUM)
losses = train(SegNet, train_data)
~~~

训练过程如下所示：

> Epoch:0 || Step:0 || Loss:3.0718
> Epoch:0 || Step:100 || Loss:2.5706
> Epoch:0 || Step:200 || Loss:1.9103
> Epoch:0 || Step:300 || Loss:2.0478
> Epoch:0 || Step:400 || Loss:2.3512
> ······
> Epoch:19 || Step:400 || Loss:0.8746
> Epoch:19 || Step:500 || Loss:0.8285
> Epoch:19 || Step:600 || Loss:0.9827
> Epoch:19 || Step:700 || Loss:0.9682
> Model saved to ./weights/SegNet_weights.pth

绘制出训练过程中的Loss曲线：

![Loss](https://cdn.jsdelivr.net/gh/Jafekin/markdown_images@main/img/Loss.svg)

从曲线的整体趋势来看，损失值在初期阶段（前 2000 步）快速下降，从约 3.0 减少到 1.5 左右，这表明模型在初期学习到了数据的基本特征。随后，在 2000 到 8000 步之间，损失值下降速度有所放缓，但仍保持逐步下降的趋势，显示模型继续优化。在训练的后期（8000 步之后），损失值趋于平稳，徘徊在 0.5附近，表明模型逐渐接近收敛状态。

### 验证模型效果

 这里我们使用了四张图片对模型效果进行可视化验证：

```python
def test(SegNet):
    if MODE:
        # 加载预训练权重
        ms.load_checkpoint(WEIGHTS, net=SegNet)
    SegNet.set_train(False)

    # 获取所有测试图片
    paths = os.listdir(SAMPLES)

    for path in paths:
        # 加载并预处理图像
        image_src = Image.open(os.path.join(SAMPLES, path)).convert(
            "RGB")  # 加载图像并转换为RGB模式
        image_src_resized = image_src.resize((224, 224))  # 调整大小到网络输入
        image = np.array(image_src_resized) / 255.0  # 归一化到 [0, 1]
        image = np.transpose(image, (2, 0, 1))  # 转换为 (C, H, W)
        image = Tensor(image[np.newaxis, ...],
                       dtype=mindspore.float32)  # 添加批次维度

        # 模型推理
        output = SegNet(image)
        output = ops.ResizeBilinear(
            (1024, 2048), align_corners=False)(output)  # 调整到目标尺寸
        output = ops.Argmax(axis=1)(
            output).asnumpy().squeeze()  # 获取预测类别并移除批次维度

        # 生成分割图像
        image_seg = np.zeros((1024, 2048, 3), dtype=np.uint8)
        for c in range(CLASS_NUM):
            mask = (output == c)
            image_seg[mask] = COLORS[c]

        # 将分割结果与原图结合（透明度混合）
        image_src_resized_back = image_src.resize((2048, 1024))  # 将原图调整到分割结果大小
        image_src_array = np.array(image_src_resized_back)  # 转换为数组
        alpha = 0.5  # 设置透明度
        image_blend = (image_src_array * (1 - alpha) +
                       image_seg * alpha).astype(np.uint8)  # 混合

        # 保存结果
        os.makedirs(OUTPUTS, exist_ok=True)
        result = Image.fromarray(image_blend)
        result.show()  # 显示图片
        result.save(os.path.join(OUTPUTS, path))
        # print(f"{path} is done!")


# 调用测试函数
test(SegNet)
```

![image-20241230030123965](https://cdn.jsdelivr.net/gh/Jafekin/markdown_images@main/img/image-20241230030123965.png)

![image-20241230030133512](https://cdn.jsdelivr.net/gh/Jafekin/markdown_images@main/img/image-20241230030133512.png)

![image-20241230030145558](https://cdn.jsdelivr.net/gh/Jafekin/markdown_images@main/img/image-20241230030145558.png)

![image-20241230030200790](https://cdn.jsdelivr.net/gh/Jafekin/markdown_images@main/img/image-20241230030200790.png)

从整体表现来看，模型较好地识别并分割了图像中的主要类别。例如，道路区域（通常为紫色）被正确标注，且边界较为清晰，说明模型对大类别（如道路和天空）的表现较为可靠。车辆的分割表现也较为精准，尤其是停放在道路两侧的车辆，能够清晰地从背景中分离出来。此外，行人的分割效果在密集区域较好，但对于远处或较小的行人目标，分割精度略有不足，可能是由于分辨率较低或小目标样本较少导致。树木和背景（如天空和建筑）的分割区域均匀，整体效果较好，但在某些复杂场景中，边界部分可能存在细节丢失的情况。

总体而言，这些分割结果表明模型在城市街景语义分割任务中表现出色，特别是在主要类别的识别与分割方面有较高的准确性。然而，对于复杂场景的小目标和类别边界的细化处理，仍然存在一定的改进空间，可以通过引入更高级的网络结构（如注意力机制或更精细的多尺度特征提取方法）进一步提升分割质量。

## 结果分析——计算mIoU（语义分割的评价指标）

**mIoU（Mean Intersection over Union）** 是语义分割任务中最常用的评价指标之一，用于衡量模型在像素级别的分割性能。它通过计算预测结果与真实标注之间的重叠程度来评估分割的准确性。mIoU 是对所有类别 IoU（Intersection over Union）的平均值，反映了模型在多类别场景中的综合表现。

![左图为groudtruth，右图为prediction](https://cdn.jsdelivr.net/gh/Jafekin/markdown_images@main/img/7114766199d99b544b2987baf1c513c8.jpeg)

IoU（交并比）定义为预测区域与真实区域的交集面积与并集面积的比值。交并比不仅仅在语义分割中使用，在目标检测等方向也是常用的指标之一。计算公式为：
$$
IoU=\frac{target\bigwedge{prediction}}{target\bigcup{prediction}}
$$

mIoU 的优势在于它综合考虑了每个类别的表现，既能评估模型在大类别（如背景或道路）上的性能，也能衡量在小类别（如行人或交通标志）上的表现。相比于像素准确率（Pixel Accuracy），mIoU 对类别不平衡问题更为鲁棒。例如，分割任务中背景像素数量通常远远多于其他类别，仅凭像素准确率可能掩盖模型在小目标上的不足，而 mIoU 能够更客观地评价模型性能。

下面使用`val`数据集计算模型的**mIoU**：

~~~python
VAL_PATHS = "./dataset-list/val.txt"

def calculate_mIoU(val_paths, model, class_num):
    mIoU = []
    with open(val_paths, "r") as paths:
        for index, line in enumerate(paths):
            line = line.strip()
            path = line.split()

            # 加载图像
            image = cv.imread(path[0])
            image = cv.resize(image, (224, 224))
            image = image / 255.0  # 归一化输入
            image = np.transpose(image, (2, 0, 1))  # 转换为 (C, H, W)
            image = Tensor(image[np.newaxis, ...],
                           dtype=mindspore.float32)  # 添加批次维度

            # 模型推理
            model.set_train(False)
            output = model(image)
            output = ops.Argmax(axis=1)(output).asnumpy().squeeze()  # 获取类别预测结果
            predict = cv.resize(np.uint8(output), (2048, 1024))  # 调整大小

            # 加载标签
            label = cv.imread(path[1], cv.IMREAD_GRAYSCALE)

            # 计算 IoU
            intersection, union = [], []
            for i in range(1, class_num):
                intersect = np.sum((predict == i) & (label == i))
                union_area = np.sum(predict == i) + \
                    np.sum(label == i) - intersect
                intersection.append(intersect)
                union.append(union_area)

            iou = [inter / u if u > 0 else 0 for inter,
                   u in zip(intersection, union)]
            mIoU.append(np.mean(iou))

            print(f"miou_{index}: {mIoU[index]:.4f}")
    return mIoU


mIoU = calculate_mIoU(VAL_PATHS, SegNet, CLASS_NUM)

result_file = "result.txt"
mean_mIoU = np.mean(mIoU)
print("\n")
print(f"mIoU: {mean_mIoU:.4f}")

with open(result_file, "a") as file:
    file.write(f"评价日期：{time.asctime(time.localtime(time.time()))}\n")
    file.write(f"使用的权重：{WEIGHTS}\n")
    file.write(f"mIoU: {mean_mIoU:.4f}\n")
~~~

 计算结果：

> 评价日期：Sat Dec 28 16:32:41 2024
> 使用的权重：weights/SegNet_weights1.pth
> mIoU: 0.7534

在测试集上的 mIoU（Mean Intersection over Union）达到 **0.7534**。这一结果表明，模型在语义分割任务中具有较强的表现，能够有效地完成多类别场景的像素级分割。mIoU 值超过 0.75，说明模型在绝大部分类别上（如道路、建筑、车辆等）实现了较高的分割精度，预测的像素区域与真实标注区域的重叠程度较大。同时，这也反映出模型对不同类别的分割性能较为均衡，既能够处理大类别区域（如道路、天空），也能对小目标（如行人、交通标志）实现一定程度的分割。这一结果验证了模型设计（如编码器-解码器结构和带索引的上采样）的有效性，以及训练策略的合理性。
