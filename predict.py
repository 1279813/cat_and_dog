# coding:UTF-8
import argparse
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg19 import preprocess_input


def predict(model):
    if opt.videocap == True:
        cap = cv2.VideoCapture(0)
        while True:
            open, frame = cap.read()
            img2 = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img = img2.resize((224, 224))
            img = img_to_array(img)
            img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
            img = preprocess_input(img)
            output = model.predict(img)
            res = np.argmax(output, axis=1)
            print(class_dict[res[0]])
            cv2.imshow("img", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # Esc键退出
                break
    else:
        model.load_weights(opt.model)
        img = load_img(opt.img, target_size=(224, 224))
        img = img_to_array(img)
        img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
        img = preprocess_input(img)
        output = model.predict(img)
        res = np.argmax(output, axis=1)
        print(class_dict[res[0]])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="model/cat-and-dog.h5", help="模型保存位置", type=str)
    parser.add_argument('--img', default="cats_and_dogs_dataset/train/cats/cat.0.jpg", help="待预测图片",
                        type=str)
    parser.add_argument('--videocap', default=False, help="是否需要调用摄像头检测")
    opt = parser.parse_args()

    class_dict = {0: "cat",
                  1: "dog"}

    model = tf.keras.models.load_model(opt.model)

    predict(model=model)
