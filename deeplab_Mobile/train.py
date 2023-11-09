import time
import keras
import numpy as np
from keras import backend as K
from keras.callbacks import (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,TensorBoard)
from keras.optimizers import Adam
from keras.utils.data_utils import get_file
from PIL import Image
from nets.deeplab import Deeplabv3
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

#-------------------------------------------------------------#
#   定义了一个生成器，用于读取datasets2文件夹里面的图片与标签
#-------------------------------------------------------------#
#-------------------------------------------------------------#
#   改
#-------------------------------------------------------------#
city_list = ["aachen", "bochum", "bremen", "cologne", "darmstadt", "dusseldorf", "erfurt", "hamburg", "hanover",
             "jena", "krefeld", "monchengladbach", "strasbourg", "stuttgart", "tubingen", "ulm", "weimar", "zurich"]
# txt_number = random.randint(1,18)
txt_number = 0


def generate_arrays_from_file(lines, batch_size):
    # 获取总长度
    n = len(lines)
    i = 0
    while 1:
        X_train = []
        Y_train = []
        # 获取一个batch_size大小的数据
        for _ in range(batch_size):
            if i == 0:
                np.random.shuffle(lines)
            name = lines[i].split(';')[0]
            # 从文件中读取图像
            # img = Image.open(r".\dataset2\jpg" + '/' + name)
            img = Image.open(r"D:\Deep Learning\make dataset_2\leftImg8bit\train01" + '/' + name)
            img = img.resize((WIDTH, HEIGHT))
            img = np.array(img)
            img = img / 255
            X_train.append(img)

            name = (lines[i].split(';')[1]).replace("\n", "")
            # 从文件中读取图像
            # img = Image.open(r".\dataset2\png" + '/' + name)
            img = Image.open(r"D:\Deep Learning\make dataset_2\gtFine\train01" + '/' + name)

            # 一种方法：尝试从“L”转为“RGB”
            # img = img.convert("RGB")

            img = img.resize((int(WIDTH), int(HEIGHT)))
            img = np.array(img)

            # 二种方法：尝试升维   失败
            # img = img[:, :, None]

            seg_labels = np.zeros((int(HEIGHT), int(WIDTH), NCLASSES))
            for c in range(NCLASSES):
                # seg_labels[: , : , c ] = (img[:,:,0] == c ).astype(int)
                seg_labels[:, :, c] = (img[:] == c).astype(int)
            seg_labels = np.reshape(seg_labels, (-1, NCLASSES))
            Y_train.append(seg_labels)

            # 读完一个周期后重新开始
            i = (i + 1) % n
        yield (np.array(X_train), np.array(Y_train))



def loss(y_true, y_pred):
    crossloss = K.binary_crossentropy(y_true,y_pred)
    loss = K.sum(crossloss)/HEIGHT/WIDTH
    return loss


if __name__ == "__main__":
    # ---------------------------------------------#
    #   定义输入图片的高和宽，以及种类数量
    # ---------------------------------------------#
    HEIGHT = 416
    WIDTH = 416
    # ---------------------------------------------#
    #   背景 + 斑马线 = 2
    # ---------------------------------------------#
    NCLASSES = 2

    log_dir = "logs/"
    model = Deeplabv3(classes=NCLASSES, input_shape=(HEIGHT, WIDTH, 3))
    # ---------------------------------------------------------------------#
    #   这一步是获得主干特征提取网络的权重、使用的是迁移学习的思想
    #   如果下载过慢，可以复制连接到迅雷进行下载。
    #   之后将权值复制到目录下，根据路径进行载入。
    #   如：
    #   weights_path = "xxxxx.h5"
    #   model.load_weights(weights_path,by_name=True,skip_mismatch=True)
    # ---------------------------------------------------------------------#
    # BASE_WEIGHT_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.6/'
    # model_name = 'mobilenet_%s_%d_tf_no_top.h5' % ('1_0', 224)
    # weight_path = BASE_WEIGHT_PATH + model_name
    # weights_path = keras.utils.get_file(model_name, weight_path)
    #weights_path = (r"C:\Users\11356\.keras\models\deeplabv3_mobilenetv2_tf_dim_ordering_tf_kernels.h5")
    weights_path = (r"D:\Deep Learning\semantic-segmentation-master\deeplab_Mobile\logs\ep013-loss0.162-val_loss0.195.h5")
    model.load_weights(weights_path, by_name=True, skip_mismatch=True)

    # 打开数据集的txt
    with open(r"D:\Deep Learning\make dataset_2\read_data\train_data.txt") as f:
        lines = f.readlines()

    # ---------------------------------------------#
    #   打乱的数据更有利于训练
    #   90%用于训练，10%用于估计。
    # ---------------------------------------------#
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)

    # 90%用于训练，10%用于估计。
    num_val = int(len(lines) * 0.1)
    num_train = len(lines) - num_val

    # 保存的方式，1世代保存一次
    checkpoint_period = ModelCheckpoint(
        log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='val_loss',
        save_weights_only=True,
        save_best_only=True,
        period=1
    )
    # 学习率下降的方式，val_loss三次不下降就下降学习率继续训练
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.1,
        patience=3,
        verbose=1
    )
    # 是否需要早停，当val_loss一直不下降的时候意味着模型基本训练完毕，可以停止
    early_stopping = EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=10,
        verbose=1
    )

    # 交叉熵
    model.compile(loss=loss,
                  optimizer=Adam(lr=1e-3),
                  metrics=['accuracy'])
    batch_size = 2
    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))

    # 开始训练
    model.fit_generator(generate_arrays_from_file(lines[:num_train], batch_size),
                        steps_per_epoch=max(1, num_train // batch_size),
                        validation_data=generate_arrays_from_file(lines[num_train:], batch_size),
                        validation_steps=max(1, num_val // batch_size),
                        epochs=50,
                        initial_epoch=0,
                        callbacks=[checkpoint_period, reduce_lr,early_stopping])

    model.save_weights(log_dir + 'last1.h5')
