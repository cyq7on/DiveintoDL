import tensorflow as tf
from tensorflow import data as tfdata
from tensorflow import optimizers
from tensorflow import losses
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow import initializers as init

print(tf.__version__)

num_feature = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = tf.random.normal((num_examples, num_feature), stddev=1)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += tf.random.normal(labels.shape, stddev=0.01)
print(features[0], labels[0])

batch_size = 10
# Dataset API:https://zhuanlan.zhihu.com/p/30751039
# 将训练数据的特征和标签组合
dataset = tfdata.Dataset.from_tensor_slices((features, labels))
# 随机读取小批量
dataset = dataset.shuffle(buffer_size=num_examples)
dataset = dataset.batch(batch_size)
# data_iter = iter(dataset)

model = keras.Sequential()
# 第一个参数是units，输出的维度大小
model.add(layers.Dense(1, kernel_initializer=init.RandomNormal(stddev=0.01)))

loss = losses.MeanSquaredError()

trainer = optimizers.SGD(learning_rate=0.03)

num_epochs = 3
for epoch in range(num_epochs):
    for (batch, (X, y)) in enumerate(dataset):
        with tf.GradientTape() as tape:
            l = loss(model(X, training=True), y)
        grads = tape.gradient(l, model.trainable_variables)
        trainer.apply_gradients(zip(grads, model.trainable_variables))
    l = loss(model(features), labels)
    print('epoch %d, loss: %f' % (epoch, l))
