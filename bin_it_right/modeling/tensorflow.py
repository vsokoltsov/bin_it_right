import tensorflow as tf


@tf.keras.saving.register_keras_serializable(package="garbage")
class GarbageClassificationCNN(tf.keras.Model):
    def __init__(self, num_classes: int = 6) -> None:
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(16, 3, padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.pool = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)

        self.conv2 = tf.keras.layers.Conv2D(32, 3, padding="same")
        self.bn2 = tf.keras.layers.BatchNormalization()

        self.conv3 = tf.keras.layers.Conv2D(64, 3, padding="same")
        self.bn3 = tf.keras.layers.BatchNormalization()

        self.gap = tf.keras.layers.GlobalAveragePooling2D()
        self.fc = tf.keras.layers.Dense(num_classes)  # logits

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        x = self.pool(tf.nn.relu(self.bn1(self.conv1(x), training=training)))
        x = self.pool(tf.nn.relu(self.bn2(self.conv2(x), training=training)))
        x = self.pool(tf.nn.relu(self.bn3(self.conv3(x), training=training)))
        x = self.gap(x)
        x = self.fc(x)
        return x
