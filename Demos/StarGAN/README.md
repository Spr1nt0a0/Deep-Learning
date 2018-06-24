<p align="center"><img width="40%" src="./assests/logo.jpg" /></p>

--------------------------------------------------------------------------------
## Requirements
* Tensorflow 1.8
* Python 3.6

## Usage
### 下载数据集
```python
> python download.py celebA
```

```
├── dataset
   └── celebA
       ├── train
           ├── 000001.jpg 
           ├── 000002.jpg
           └── ...
       ├── test (It is not celebA)
           ├── a.jpg (The test image that you wanted)
           ├── b.png
           └── ...
       ├── list_attr_celeba.txt (For attribute information) 
```

### 训练
* python main.py --phase train

### 测试
* python main.py --phase test 
* 同时运行 celebA 测试图像和您想要的图像


### 预训练模型
* Download [checkpoint for 128x128](https://drive.google.com/open?id=1ezwtU1O_rxgNXgJaHcAynVX8KjMt0Ua-)



