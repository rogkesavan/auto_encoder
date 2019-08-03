import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig
import cv2
np.set_printoptions(threshold=np.inf)

def autoencoder(inputs): 

    net = tf.layers.conv2d(inputs, 128, 2, activation = tf.nn.relu)
    net = tf.layers.max_pooling2d(net, 2, 2, padding = 'same')
    net = tf.image.resize_nearest_neighbor(net, tf.constant([129, 129]))
    net = tf.layers.conv2d(net, 1, 2, activation = None, name = 'outputOfAuto')
    return net
ae_inputs = tf.placeholder(tf.float32, (None, 128, 128, 3), name = 'inputToAuto')
ae_target = tf.placeholder(tf.float32, (None, 128, 128, 1))

ae_outputs = autoencoder(ae_inputs)
lr = 0.001

loss = tf.reduce_mean(tf.square(ae_outputs - ae_target)) #loss_function 
train_op = tf.train.AdamOptimizer(learning_rate = lr).minimize(loss) #adam_optimizer 

init = tf.global_variables_initializer()
saver = tf.train.Saver()

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

saver.restore(sess, 'model/colortogray.ckpt')
import glob as gl 

filenames = gl.glob('test.jpg')
test_data = []
for file in filenames[0:100]:
    test_data.append(np.array(cv2.imread(file)))

test_dataset = np.asarray(test_data)
batch_imgs = test_dataset
gray_imgs = sess.run(ae_outputs, feed_dict = {ae_inputs: batch_imgs})

for i in range(gray_imgs.shape[0]):
    cv2.imwrite('output' +str(i) +'.jpeg', gray_imgs[i])
    print('output_save_successfully')

imgFile = cv2.imread('output0.jpeg')

cv2.imshow('output', imgFile)
cv2.waitKey(0)
cv2.destroyAllWindows()