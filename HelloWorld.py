import tensorflow as tf
import time
from tensorflow.examples.tutorials.mnist import input_data

# MNISTデータのダウンロード
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# 外部処理を行うためのInteractiveSessionを作成
sess = tf.InteractiveSession()
# 入力 28*28画像
x = tf.placeholder(tf.float32, shape=[None, 784])
# 出力 10種類
y_ = tf.placeholder(tf.float32, shape=[None, 10])
# 784*10行列
W = tf.Variable(tf.zeros([784, 10]))
# 10次元ベクトル
b = tf.Variable(tf.zeros([10]))
# 全変数の初期化@外部セッション
sess.run(tf.global_variables_initializer())
# 入力画像xを重みWで乗算して、バイアスbを足してsoftmaxを取る。
y = tf.nn.softmax(tf.matmul(x, W) + b)
# コスト関数
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
# コスト関数を、0.01ずつ変えて最適化
# 実験に使ったステップ長は、0.5,0.1,0.05,0.001,0.0005
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)
start = time.time()
# 実験に使った試行回数は、500,1000,5000,10000
for i in range(1000):
    # 50の訓練サンプルをロード
    batch = mnist.train.next_batch(50)
    # feed_dictを用いて、placeholderをサンプルに置き換え
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})
train_end = time.time()
# 予測が真かどうか確認
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
# 正確性の計算
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# 出力
print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
print("Train:" + (train_end - start).__str__())
