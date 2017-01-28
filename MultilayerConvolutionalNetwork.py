import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time


# 重み(Weight)の初期化
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# バイアスの初期化
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# 畳み込み処理
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# プーリング処理
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

# MNISTデータのダウンロード
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# 外部処理を行うためのInteractiveSessionを作成
sess = tf.InteractiveSession()
# 入力 28*28画像
x = tf.placeholder(tf.float32, shape=[None, 784])
# 出力 10種類
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# 第一畳み込み層
# 5*5のパッチ、1つの入力チャンネル、32の出力層
W_conv1 = weight_variable([5, 5, 1, 32])
# 書く出力チャンネルのためのバイアス
b_conv1 = bias_variable([32])
# 第2，3引数は画像サイズ、第4引数はカラーチャンネル
x_image = tf.reshape(x, [-1, 28, 28, 1])
# x_imageの重みを畳み込み、バイアス加算して、ReLU関数適用
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
# プール
h_pool1 = max_pool_2x2(h_conv1)
# 第二畳み込み層 出力は64
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
# よくわからんけど、一回畳み込むごとに、画像サイズが1/2になっていってるっぽい？
# poolが2*2だからかな やっぱり本格的にディープラーニング勉強しないと

# 1024のニューロン全結合層
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
# プーリング層→バッチにReshape
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
# 加重・バイアス・ReLU!
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# 過学習対策のドロップアウト
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 読み出し用のsoftmax
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# モデルの評価を、最急勾配降下→ADAM
cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())
for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x: batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g" % (i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g" % accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

