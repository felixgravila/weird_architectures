#%%

import tensorflow as tf
import tensorflow.keras.layers as L
import matplotlib.pyplot as plt


class Forgetron(L.Layer):
    def __init__(
        self,
        units: int,
        activation: str = None,
        trainable: bool = True,
        name:str = None,
        **kwargs
    ):
        """
        Forgetron, an "augmented" Dense layer that can decide to forget some outputs.
        * Computes the dense output `o` (without activation) from input `i`
        * Computes a sigmoid `forget` vector `f` from both `o` and `i`
        * Uses `f` to choose how much of `o` to keep
        * Applies activation

        args:
        units: number of neurons in the dense layer
        activation: activation for the output
        """
        super(Forgetron, self).__init__(name=name, **kwargs)
        self.dense = L.Dense(units, activation=None, trainable=trainable, name=f"{name}_dense")
        self.activation = L.Activation(activation, name=f"{name}_activation")
        self.forget = L.Dense(units, activation="sigmoid", trainable=trainable, name=f"{name}_forget")

    def call(self, inputs, training):
        dense_out = self.dense(inputs)
        decider = self.forget(tf.concat([inputs, dense_out], axis=-1))

        # selected_out = tf.where(decider >= 0, dense_out, dense_out * self.neg_ratio)
        selected_out = dense_out * decider
        activated_out = self.activation(selected_out)
        return activated_out

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

    activation = "sigmoid"

    histories = []
    for force_std in [True, False]:

        layer_sizes = [256, 256, 128, 128, 64, 64]

        if force_std:
            layers = [
                L.Dense(s, activation=activation, name=f"dense_{i}")
                for i,s in enumerate(layer_sizes)
            ]
        else:
            layers = [
                Forgetron(s, activation=activation, name=f"forgetron_{i}")
                for i,s in enumerate(layer_sizes)
            ]

        model = tf.keras.Sequential(
            [
                L.InputLayer((28, 28, 1)),
                L.Flatten(),
            ] + layers + [
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
            epochs=30,
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
    plt.savefig(f"forgetron/results/{ds_name}_{activation}.png")

# %%
