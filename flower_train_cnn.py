#coding=utf-8
import tensorflow as tf # tensorflow module
import numpy as np # numpy module
import os # path join


DATA_DIR = "./data/"
TRAINING_SET_SIZE = 3670
BATCH_SIZE = 64
IMAGE_SIZE = 224 #要进行分类,224*224是一个标准大小,经典的网络


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# 一张图像必要的信息封装成一个类
class _image_object:
    def __init__(self):
        self.image = tf.Variable([], dtype = tf.string) #图像
        self.height = tf.Variable([], dtype = tf.int64) #高度
        self.width = tf.Variable([], dtype = tf.int64) #宽度
        self.filename = tf.Variable([], dtype = tf.string) #路径
        self.label = tf.Variable([], dtype = tf.int32) #数字标签

def read_and_decode(filename_queue):
    reader = tf.TFRecordReader() #创建一个阅读tfrecord文件的阅读器
    '''
    https://www.jianshu.com/p/d063804fb272
    import tensorflow as tf
    
    # 同时打开多个文件，显示创建Queue，同时隐含了QueueRunner的创建
    filename_queue = tf.train.string_input_producer(["data1.csv","data2.csv"])
    reader = tf.TextLineReader(skip_header_lines=1)
    # Tensorflow的Reader对象可以直接接受一个Queue作为输入
    key, value = reader.read(filename_queue)
    
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        # 启动计算图中所有的队列线程
        threads = tf.train.start_queue_runners(coord=coord)
        # 主线程，消费100个数据
        for _ in range(100):
            features, labels = sess.run([data_batch, label_batch])
        # 主线程计算完成，停止所有采集数据的进程
        coord.request_stop()
        coord.join(threads)
    '''
    _, serialized_example = reader.read(filename_queue) #读取一个字符串队列,依次读取tfrecord文件
    #将生成tfrecord文件时的特征读取出来
    #一些像text,colorspace,channel都没读出来,只读取了有用的特征
    features = tf.parse_single_example(serialized_example, features = {
        "image/encoded": tf.FixedLenFeature([], tf.string),
        "image/height": tf.FixedLenFeature([], tf.int64),
        "image/width": tf.FixedLenFeature([], tf.int64),
        "image/filename": tf.FixedLenFeature([], tf.string),
        "image/class/label": tf.FixedLenFeature([], tf.int64),})
    image_encoded = features["image/encoded"] #取出编码好的jpg文件
    image_raw = tf.image.decode_jpeg(image_encoded, channels=3) #解码成RGB jpg
    image_object = _image_object() #创建一个图像对象
    #对图像对象的五个必不可少的属性进行赋值
    #将图像像素点image_raw,resize成IMAGE_SIZE*IMAGE_SIZE大小的图像
    image_object.image = tf.image.resize_image_with_crop_or_pad(image_raw, IMAGE_SIZE, IMAGE_SIZE)
    image_object.height = features["image/height"]
    image_object.width = features["image/width"]
    image_object.filename = features["image/filename"]
    image_object.label = tf.cast(features["image/class/label"], tf.int64)
    return image_object #返回单独一张图片(但是features是一个队列,详情见tensorflow队列机制)

def flower_input(if_random = True, if_training = True):
    if(if_training): #如果在训练,就将两个tfrecord文件加到filenames列表中
        filenames = [os.path.join(DATA_DIR, "train-0000%d-of-00002.tfrecord" % i) for i in range(0, 2)]
    else:
        filenames = [os.path.join(DATA_DIR, "eval-0000%d-of-00002.tfrecord" % i) for i in range(0, 2)]

    for f in filenames:
        if not tf.gfile.Exists(f): #如果文件路径不存在,直接提交一个error
            raise ValueError("Failed to find file: " + f)
    #输出字符串到一个输入管道队列。
    filename_queue = tf.train.string_input_producer(filenames)
    image_object = read_and_decode(filename_queue)
    #图像预处理操作可以在制作tfrecord时进行,也可以在读取时进行
    #这里使用tensorflow中自带的函数对原生的image像素值进行标准化
    image = tf.image.per_image_standardization(image_object.image)
    #底下这两行是另一种标准化的方式
#    image = image_object.image
#    image = tf.image.adjust_gamma(tf.cast(image_object.image, tf.float32), gamma=1, gain=1) # Scale image to (0, 1)
    label = image_object.label
    filename = image_object.filename

    if(if_random): #函数传参时决定做不做数据随机
        min_fraction_of_examples_in_queue = 0.4
        min_queue_examples = int(TRAINING_SET_SIZE * min_fraction_of_examples_in_queue)
        print("Filling queue with %d images before starting to train. " "This will take a few minutes." % min_queue_examples)
        num_preprocess_threads = 1
        image_batch, label_batch, filename_batch = tf.train.shuffle_batch(
            [image, label, filename],
            batch_size = BATCH_SIZE,
            num_threads = num_preprocess_threads,
            capacity = min_queue_examples + 3 * BATCH_SIZE,
            min_after_dequeue = min_queue_examples)
        return image_batch, label_batch, filename_batch
    else: #这里没有使用随机
        #将image_object中的三个属性制作成64个一批次
        image_batch, label_batch, filename_batch = tf.train.batch(
            [image, label, filename], #虽然这里放的是单个样本,但是队列中会有一个全局计数器,来读取一个批次64张图像
            batch_size = BATCH_SIZE,
            num_threads = 1)
        return image_batch, label_batch, filename_batch #一个批次的相关信息(64, 224, 224, 3)


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.05)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.02, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

#网络模型,速度原因做了简化,但是还是训练的很慢
#但是看CPU和GPU利用率的话,感觉应该是CPU拖后腿了,毕竟有很多IO操作,涉及到tfrecord之类的
def flower_inference(image_batch):
    W_conv1 = weight_variable([5, 5, 3, 32])
    b_conv1 = bias_variable([32])

    x_image = tf.reshape(image_batch, [-1, IMAGE_SIZE, IMAGE_SIZE, 3])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1) # 112

    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2) # 56

    W_conv3 = weight_variable([5, 5, 64, 128])
    b_conv3 = bias_variable([128])

    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    h_pool3 = max_pool_2x2(h_conv3) # 28

    W_conv4 = weight_variable([5, 5, 128, 256])
    b_conv4 = bias_variable([256])

    h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
    h_pool4 = max_pool_2x2(h_conv4) # 14

    W_conv5 = weight_variable([5, 5, 256, 256])
    b_conv5 = bias_variable([256])

    h_conv5 = tf.nn.relu(conv2d(h_pool4, W_conv5) + b_conv5)
    h_pool5 = max_pool_2x2(h_conv5) # 7

    W_fc1 = weight_variable([7*7*256, 1024])
    b_fc1 = bias_variable([1024])

    h_pool5_flat = tf.reshape(h_pool5, [-1, 7*7*256])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool5_flat, W_fc1) + b_fc1)

    h_fc1_drop = tf.nn.dropout(h_fc1, 1.0)

    #模型太复杂了,我的GPU训练起来太慢,所以进行了一系列的简化
    W_fc2 = weight_variable([1024, 64])
    b_fc2 = bias_variable([64])

    h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    # W_fc3 = weight_variable([256, 64])
    # b_fc3 = bias_variable([64])
    #
    # h_fc3 = tf.nn.relu(tf.matmul(h_fc2, W_fc3) + b_fc3)

    W_fc4 = weight_variable([64, 5])
    b_fc4 = bias_variable([5])

    # y_conv = tf.nn.softmax(tf.matmul(h_fc3, W_fc4) + b_fc4)
    y_conv = tf.nn.softmax(tf.matmul(h_fc2, W_fc4) + b_fc4)
#    y_conv = tf.matmul(h_fc3, W_fc4) + b_fc4

    return y_conv


def flower_train():
    #取出一个批次的数据:
    image_batch_out, label_batch_out, filename_batch = flower_input(if_random = False, if_training = True)

    # 接下来把原始数据和placeholder准备好:
    #这是图像批次数据的placeholder
    image_batch_placeholder = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 224, 224, 3])
    #将数据reshape一下,别出现问题
    image_batch = tf.reshape(image_batch_out, (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3))

    #这是图像标签的placeholder,几分类就是(BATCH_SIZE,几)
    label_batch_placeholder = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 5])
    #原来的label是1,2,3,4,5
    #这里添加一个-1,变成从0开始的,0,1,2,3,4
    label_offset = -tf.ones([BATCH_SIZE], dtype=tf.int64, name="label_batch_offset")
    #one_hot格式的标签
    label_batch_one_hot = tf.one_hot(tf.add(label_batch_out, label_offset), depth=5, on_value=1.0, off_value=0.0)

    #建立神经网络模型,logits_out是神经网络输出
    logits_out = flower_inference(image_batch_placeholder)
    #loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=label_batch_one_hot, logits=logits_out))
    loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(label_batch_out,depth=5), logits=logits_out))
    #loss = tf.losses.mean_squared_error(labels=label_batch_placeholder, predictions=logits_out)

    # train_step = tf.train.GradientDescentOptimizer(0.0001).minimize(loss)
    train_step = tf.train.AdamOptimizer(0.001).minimize(loss)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #saver.save(sess, "")
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess = sess)

        # for i in range(TRAINING_SET_SIZE * 100):
        for i in range(int(TRAINING_SET_SIZE / BATCH_SIZE * 100)):
            image_out, label_out, label_batch_one_hot_out, filename_out = sess.run([image_batch, label_batch_out, label_batch_one_hot, filename_batch])

            _, infer_out, loss_out = sess.run([train_step, logits_out, loss], feed_dict={image_batch_placeholder: image_out, label_batch_placeholder: label_batch_one_hot_out})
            """
            print(i)
            print(image_out.shape)
            print("label_out: ")
            print(filename_out)
            print(label_out)
            print(label_batch_one_hot_out)
            print("infer_out: ")
            print(infer_out)
            print("loss: ")
            print(loss_out)
            """
            if(i % 100 == 0):
                print(i)
                print(image_out.shape)
                print("label_out: ")
                print(filename_out)
                print(label_out)
                print(label_batch_one_hot_out)
                print("infer_out: ")
                print(infer_out)
                print("loss: ")
                print(loss_out)
                

        coord.request_stop()
        coord.join(threads)
        sess.close()



def flower_eval():
    image_batch_out, label_batch_out, filename_batch = flower_input(if_random = True, if_training = False)

    image_batch_placeholder = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 224, 224, 3])
    image_batch = tf.reshape(image_batch_out, (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3))

    label_tensor_placeholder = tf.placeholder(tf.int64, shape=[BATCH_SIZE])
    label_offset = -tf.ones([BATCH_SIZE], dtype=tf.int64, name="label_batch_offset")
    label_batch = tf.add(label_batch_out, label_offset)

    logits_out = tf.reshape(flower_inference(image_batch_placeholder), [BATCH_SIZE, 5])
    logits_batch = tf.to_int64(tf.arg_max(logits_out, dimension = 1))

    correct_prediction = tf.equal(logits_batch, label_tensor_placeholder)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #saver.restore(sess, "���·��")
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess = sess)

        accuracy_accu = 0

        for i in range(100):
            image_out, label_out, filename_out = sess.run([image_batch, label_batch, filename_batch])

            accuracy_out, logits_batch_out = sess.run([accuracy, logits_batch], feed_dict={image_batch_placeholder: image_out, label_tensor_placeholder: label_out})
            #accuracy_accu += accuracy_out

            #print(i)
            #print(image_out.shape)
            #print("label_out: ")
            #print(filename_out)
            #print(label_out)
            #print(logits_batch_out)

            print("Accuracy: ")
        #print(accuracy_accu / 29)

        coord.request_stop()
        coord.join(threads)
        sess.close()

flower_train()