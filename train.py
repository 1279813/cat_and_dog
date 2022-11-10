# coding:UTF-8
import argparse
import os
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def dataset(train_dir, val_dir):
    train_generator = ImageDataGenerator(rescale=1.0 / 255.0)
    train_gen = train_generator.flow_from_directory(train_dir,
                                                    target_size=(224, 224),
                                                    batch_size=32,
                                                    class_mode="binary",
                                                    shuffle=True)

    val_generator = ImageDataGenerator(rescale=1.0 / 255.0)
    val_gen = val_generator.flow_from_directory(val_dir,
                                                target_size=(224, 224),
                                                batch_size=32,
                                                class_mode="binary",
                                                shuffle=True)

    return train_gen, val_gen


def model():
    base_model = tf.keras.applications.VGG19(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
    x = base_model.outputs[0]
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(1024, activation="relu")(x)
    output = tf.keras.layers.Dense(2, activation=tf.keras.activations.softmax)(x)
    model = tf.keras.models.Model(inputs=base_model.inputs, outputs=output)

    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])

    return model


def model_fit(model, train_gen, val_gen):
    weight_save_path = opt.model_save_path
    if not os.path.exists(weight_save_path):
        os.makedirs(weight_save_path)

    save_model = tf.keras.callbacks.ModelCheckpoint(weight_save_path + "best.h5",
                                                    monitor="val_accuracy",
                                                    save_weights_only=True,
                                                    save_best_only=True,
                                                    mode="auto",
                                                    period=1)

    if opt.ckpt == True:
        ckpt_save_path = opt.ckpt_save_path
        if os.path.exists(ckpt_save_path + ".index"):
            print("=" * 40 + "load the ckpt" + "=" * 30)
            model.load_weights(ckpt_save_path)

        ckpt = tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_save_path,
                                                  save_weights_only=True,
                                                  save_best_only=True)

    history = model.fit_generator(train_gen, epochs=opt.epochs, validation_data=val_gen, callbacks=[ckpt, save_model])

    return history


def plt_parameter(history):
    if not os.path.exists("plt"):
        os.makedirs("plt")
    acc = history.history["accuracy"]
    loss = history.history["loss"]
    val_acc = history.history["val_accuracy"]
    val_loss = history.history["val_loss"]

    plt.plot(acc, label="acc")
    plt.plot(val_acc, label="val_acc")
    plt.title("acc")
    plt.legend()
    plt.savefig("plt/acc.png")
    plt.close()

    plt.plot(loss, label="loss")
    plt.plot(val_loss, label="val_loss")
    plt.title("loss")
    plt.legend()
    plt.savefig("plt/loss.png")
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--ckpt', default=True, help="是否需要断点续训")
    parser.add_argument('--ckpt_save_path', default="weight/ckpt/", help="断点续训文件保存位置", type=str)
    parser.add_argument('--model_save_path', default="weight/weight/", help="模型保存位置", type=str)
    parser.add_argument('--train_dir', default="cats_and_dogs_dataset/train", help="训练集位置", type=str)
    parser.add_argument('--val_dir', default="cats_and_dogs_dataset/val", help="验证集位置", type=str)
    opt = parser.parse_args()

    train_dir = opt.train_dir
    val_dir = opt.val_dir
    train, val = dataset(train_dir, val_dir)
    model = model()
    history = model_fit(model, train, val)
    plt_parameter(history)
