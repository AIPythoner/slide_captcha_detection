# -
识别网易腾讯等滑块型验证码的坐标


##### 环境及版本依赖
依赖     | 版本
-------- | -----
系统  | win10
cuda/cudnn  | 10.0/7.6.0
python | 3.6
tensorflow-gpu | 1.9
object_detection|https://github.com/tensorflow/models/tree/v1.13.0/research/object_detection



##### 主要流程
>1 安装环境以及下载对应的项目代码
>2 将\models\research\slim 加入PYTHONPATH,或者cd research/slim , python setup.py build , python setup.py install
>3 编译Protobuf以及安装该API，powershell可以使用.*通配符
>4 执行setup将object_detection安装到site-pachages中去
>5 判断该API是否安装成功以及测试官方demo
>6 标注训练集，xml转csv
>7 csv训练集转换为TF Recode格式
>8 下载预训练的模型的配置文件，然后进行相关配置，如果class不需要的话只填写就好了，label.ckpt中写一个item就好了
>9 训练，step最少20000，训练集数量最好>=400
>10 转换动态模型为冻结模型pb
>11 测试模型的训练效果

##### 特别注意
GPU训练请执行以下cmd命令，最新版本训练入口迁移到model_main.py文件了而不是train，但是此脚本无法跑gpu，
```cmd
python legacy/train.py --train_dir=training/ --pipeline_config_path=training/faster_rcnn_inception_v2_pets.config --alsologtostderr
```

##### 具体的配置请参考一下文章即可
整套流程：https://blog.csdn.net/csdn_6105/article/details/82933628

问题汇总：https://blog.csdn.net/dy_guox/article/details/80139981

项目地址：https://github.com/tensorflow/models/tree/v1.13.0/research/object_detection

视频教程：https://www.bilibili.com/video/av21539370/?p=2

其他参考：https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10


欢迎加入小白交流群1135165504，一起学习共同进步，其他群吹水太多，有资源大家一起分享
