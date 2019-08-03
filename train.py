import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig
import cv2
np.set_printoptions(threshold=np.inf)

dataset = []

for i in range(1, 812+1):#size_of_dataset '812'
    img = cv2.imread("color/co" +str(i) +".jpg" ) 
    dataset.append(np.array(img))
dataset_source = np.asarray(dataset)

dataset_tar = []

for i in range(1, 812+1):#size_of_dataset '812'
    img = cv2.imread("gray/gr" +str(i) +".jpg", 0)    
    dataset_tar.append(np.array(img))
dataset_target = np.asarray(dataset_tar)

dataset_target = dataset_target[:, :, :, np.newaxis]

def colortogray(inputs): 
    
    net = tf.layers.conv2d(inputs, 128, 2, activation = tf.nn.relu) # Encoder
    print(net.shape)
    net = tf.layers.max_pooling2d(net, 2, 2, padding = 'same')
    print(net.shape)

    net = tf.image.resize_nearest_neighbor(net, tf.constant([129, 129])) # Decoder 
    net = tf.layers.conv2d(net, 1, 2, activation = None, name = 'outputOfAuto')
    return net

ae_inputs = tf.placeholder(tf.float32, (None, 128, 128, 3), name = 'inputToAuto')
ae_target = tf.placeholder(tf.float32, (None, 128, 128, 1))

ae_outputs = colortogray(ae_inputs)
lr = 0.001

loss = tf.reduce_mean(tf.square(ae_outputs - ae_target))
train_op = tf.train.AdamOptimizer(learning_rate = lr).minimize(loss)
# Intialize the network 
init = tf.global_variables_initializer()
print('Network Start training.......')
batch_size = 32
epoch_num = 12

saving_path = 'model/colortogray.ckpt'

saver_ = tf.train.Saver(max_to_keep = 3)

batch_img = dataset_source[0:batch_size]
batch_out = dataset_target[0:batch_size]

num_batches = 812//batch_size #size_of_dataset '812'

sess = tf.Session()
sess.run(init)

for ep in range(epoch_num):
    batch_size = 0
    for batch_n in range(num_batches): 

        _, c = sess.run([train_op, loss], feed_dict = {ae_inputs: batch_img, ae_target: batch_out})
        print("Epoch: {} - cost = {:.5f}" .format((ep+1), c))
            
        batch_img = dataset_source[batch_size: batch_size+32]
        batch_out = dataset_target[batch_size: batch_size+32]
            
        batch_size += 32
    
    saver_.save(sess, saving_path)
recon_img = sess.run([ae_outputs], feed_dict = {ae_inputs: batch_img})
saver = tf.train.Saver()
sess.close()

print('traing Sucessfully complete')
