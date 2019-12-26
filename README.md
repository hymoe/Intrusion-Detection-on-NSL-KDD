# Intrusion-Detection-on-NSL-KDD

本项目复现论文《An Intrusion Detection System Using a Deep Neural Network with Gated Recurrent Units》（DOI：10.1109/ACCESS.2018.DOI）

**注意本人不是论文原作者！**

**Note that I am not the original author of the paper!**

代码基于Keras编写。

### 基于Docker的配置（非必须）

使用Docker：

https://hub.docker.com/r/gw000/keras

对应tag：:2.1.4-py3-tf-gpu

转换为本地docker：keras-py3-tf-gpu:2.1.4

CPU：

`$ docker run -it --rm -v $(pwd):/srv gw000/keras:2.1.4-py3-tf-gpu /srv/run.py`

GPU：(数据集较小，不需要）

`$ docker run -it --rm $(ls /dev/nvidia* | xargs -I{} echo '--device={}') $(ls /usr/lib/*-linux-gnu/{libcuda,libnvidia}* | xargs -I{} echo '-v {}:{}:ro') -v $(pwd):/srv gw000/keras:2.1.4-py3-tf-gpu /srv/run.py`

### 数据集：

NSL_KDD:（见NSL-KDD目录）

https://www.unb.ca/cic/datasets/nsl.html

可以参考这篇介绍文章：

https://towardsdatascience.com/a-deeper-dive-into-the-nsl-kdd-data-set-15c753364657

### 使用方法（Usage）：

`python3 run.py`

请注意确保tensorflow、sklearn、keras、numpy等依赖均已安装。另外，运行`test_keras.py`可以测试Keras工作是否正常，运行`check_tf_version.py`可以测试tensorflow版本，当前运行版本为1.5.0。*utils.ipynb是jupyter文档，用于开发过程中的实验环境。*

### 实验结果：

20个Epoch情况下，Accuracy为98%+，使用Dropout后略低(96%)。

**如有任何问题，可以邮件联系：heyu#bupt.edu.cn。论文中的问题请联系论文原作者。**
