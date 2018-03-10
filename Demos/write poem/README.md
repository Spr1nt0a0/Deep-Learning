## 环境配置
- 安装[PyTorch](http://pytorch.org)

```Python
pip install -r requirements.txt
```
- 启动visodm
```Bash
 python -m visdom.server
```

## 训练
训练的命令如下：

```Bash
python main.py train --plot-every=150\
		     --batch-size=128\
                     --pickle-path='tang.npz'\
                     --lr=1e-3 \
                     --env='poetry3' \
                     --epoch=50
```

命令行选项：
```Python
    data_path = 'data/' # 诗歌的文本文件存放路径
    pickle_path= 'tang.npz' # 预处理好的二进制文件 
    author = None # 只学习某位作者的诗歌
    constrain = None # 长度限制
    category = 'poet.tang' # 类别，唐诗还是宋诗歌(poet.song)
    lr = 1e-3 
    weight_decay = 1e-4
    use_gpu = True
    epoch = 20  
    batch_size = 128
    maxlen = 125 # 超过这个长度的之后字被丢弃，小于这个长度的在前面补空格
    plot_every = 20 # 每20个batch 可视化一次
    # use_env = True # 是否使用visodm
    env='poetry' # visdom env
    max_gen_len = 200 # 生成诗歌最长长度
    debug_file='/tmp/debugp'
    model_path=None # 预训练模型路径
    prefix_words = '细雨鱼儿出,微风燕子斜。' # 不是诗歌的组成部分，用来控制生成诗歌的意境
    start_words='闲云潭影日悠悠' # 诗歌开始
    acrostic = False # 是否是藏头诗
    model_prefix = 'checkpoints/tang' # 模型保存路径

```
## 生成诗歌

```Bash
python  main.py gen  --model-path='checkpoints/tang_199.pth' \
       --pickle-path='tang.npz' \
       --start-words='深度学习' \
       --prefix-words='江流天地外，山色有无中。' \
       --acrostic=True\
       --nouse-gpu  # 或者 --use-gpu=False
深居不可见，浩荡心亦同。度年一何远，宛转三千雄。学立万里外，诸夫四十功。习习非吾仕，所贵在其功。
```
