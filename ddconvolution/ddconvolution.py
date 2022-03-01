#%%

from typing import Tuple
import tensorflow as tf
import tensorflow.keras.layers as L
import matplotlib.pyplot as plt


class DDConvolution(L.Layer):
    def __init__(
        self,
        filter_sizes: Tuple[int],
        filters_per_size: int,
        activation: str = None,
        trainable: bool = True,
        name:str = None,
        **kwargs
    ):
        """
        Dance Dance Convolution, an "augmented" Conv2D layer that mixes many filter sizes into one.

        """
        super(DDConvolution, self).__init__(name=name, **kwargs)
        self.filter_sizes = filter_sizes
        self.filters_per_size = filters_per_size

        self.conv_layers = []
        for fsize in filter_sizes:
            self.conv_layers.append(L.Conv2D(filters_per_size, fsize, padding="same", activation=None, trainable=trainable))
        self.activation = L.Activation(activation, name=f"{name}_activation")

    def call(self, inputs, training):
        conv_output = tf.concat([conv_layer(inputs) for conv_layer in self.conv_layers], axis=0)
        return self.activation(conv_output)

#%%

if __name__ == "__main__":

    ds_name = "fashion_mnist"

    if ds_name == "mnist":
        ds = tf.keras.datasets.mnist
    elif ds_name == "fashion_mnist":
        ds = tf.keras.datasets.fashion_mnist
    else:
        raise NotImplementedError(f"Bad dataset {ds_name}")

    (x_train, y_train), (x_test, y_test) = ds.load_data()
    x_train = tf.expand_dims(x_train / 255.0, -1)
    x_test = tf.expand_dims(x_test / 255.0, -1)

    activation = "elu"

    histories = []
    for force_std in [True, False]:

        kernel_sizes = [(1,1),(2,2),(3,3),(4,4),(5,5),(6,6),(7,7),(8,8)]
        filters_per_layer = [256, 256, 128, 128, 64, 64]

        if force_std:
            layers = [
                L.Conv2D(f, (3,3), activation=activation, name=f"dense_{i}")
                for i,f in enumerate(filters_per_layer)
            ]
        else:
            layers = [
                DDConvolution(f//len(kernel_sizes), s, activation=activation, name=f"ddconv_{i}")
                for i,(f,s) in enumerate(zip(filters_per_layer, kernel_sizes))
            ]

        model = tf.keras.Sequential(
            [
                L.InputLayer((28, 28, 1)),
            ] + layers + [
                L.Flatten(),
                L.Dense(10, activation="softmax"),
            ]
        )

        model.summary()

        model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer="adam",
            metrics=["accuracy"],
        )

        model.fit(
            x_train,
            y_train,
            epochs=5,
            validation_data=(x_test, y_test),
            callbacks=[
                tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=10)
            ],
        )
        histories.append(model.history.history)

    plt.figure(figsize=(20, 10), facecolor="white")
    plt.plot(
        histories[0]["accuracy"],
        "--",
        c="#1f77b4",
        label=f"Std Dense, train acc max {max(histories[0]['accuracy'])*100:.02f}%",
    )
    plt.plot(
        histories[0]["val_accuracy"],
        "-",
        c="#1f77b4",
        label=f"Std Dense, val acc max {max(histories[0]['val_accuracy'])*100:.02f}%",
    )
    plt.plot(
        histories[1]["accuracy"],
        "--",
        c="#ff7f0e",
        label=f"Forgetron, train acc max {max(histories[1]['accuracy'])*100:.02f}%",
    )
    plt.plot(
        histories[1]["val_accuracy"],
        "-",
        c="#ff7f0e",
        label=f"Forgetron, val acc max {max(histories[1]['val_accuracy'])*100:.02f}%",
    )
    plt.title(f"{ds_name}, {activation}")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"ddconvolution/results/{ds_name}_{activation}.png")

# %%
