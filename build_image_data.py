# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
'''
从python2.1开始以后, 当一个新的语言特性首次出现在发行版中时候, 
如果该新特性与以前旧版本python不兼容, 则该特性将会被默认禁用.
 如果想启用这个新特性, 则必须使用 "from __future__ import *" 语句进行导入.
'''

from datetime import datetime
import os
import random
import sys
import threading
#多线程制作数据集，一会将数据预处理和制作数据源写在一起，多线程来进行
#数据量非常大的时候这样比较快（10000以上的数据量）
#IO密集型任务

import numpy as np
import tensorflow as tf

'''
这里的几个文件夹必须之前就有,才能自己制作数据集
flower_photos这里偷懒了没有制作验证集
可以在src根目录下手动创建一个文件夹,从flower_photos中选取一些图片作为验证集
'''
'''
通过tf.app来将一会需要用的参数指定出来
第一个参数是参数名字
'''
#训练集
tf.app.flags.DEFINE_string('train_directory', './flower_photos/',
                           'Training data directory')
#验证集
tf.app.flags.DEFINE_string('validation_directory', './flower_photos/',
                           'Validation data directory')
#输出数据源
tf.app.flags.DEFINE_string('output_directory', './data/',
                           'Output data directory')
#这里的2是指定data中生成几个tfrecord文件,本例中生成两个,如果图片再多一点的话就可以生成10个8个都可以
tf.app.flags.DEFINE_integer('train_shards', 2,
                            'Number of shards in training TFRecord files.')
#针对于测试集,要生成几个tfrecord
#这里写0是因为我们根本不会生成验证集
tf.app.flags.DEFINE_integer('validation_shards', 0,
                            'Number of shards in validation TFRecord files.')
#使用几个线程来做数据
tf.app.flags.DEFINE_integer('num_threads', 2,
                            'Number of threads to preprocess the images.')

# The labels file contains a list of valid labels are held in this file.
# Assumes that the file contains entries as such:
#   dog
#   cat
#   flower
# where each line corresponds to a label. We map each label contained in
# the file to an integer corresponding to the line number starting from 0.
#在这个flower_label.txt文件中放标签
#本例是五分类问题,这个文本文件中每一行都是flower_photos文件夹中的一个文件夹名
tf.app.flags.DEFINE_string('labels_file', './flower_label.txt', 'Labels file')

#使用这行代码将参数全拿过来
FLAGS = tf.app.flags.FLAGS


def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list): #如果不是一个list实例(只是一个数字),将之变成一个list
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value])) #返回一个feature


def _convert_to_example(filename, image_buffer, label, text, height, width):
    """Build an Example proto for an example.

  Args:
    filename: string, path to an image file, e.g., '/path/to/example.JPG'
    image_buffer: string, JPEG encoding of RGB image
    label: integer, identifier for the ground truth for the network
    text: string, unique human-readable, e.g. 'dog'
    height: integer, image height in pixels
    width: integer, image width in pixels
  Returns:
    Example proto
  """

    colorspace = 'RGB'
    channels = 3
    image_format = 'JPEG'
    #这些feature都是官网扒下来的,就不改了,都是一张张图片的特征信息
    '''
    制作数据集时,每一条数据的特征都可以由自己指定
    feature是一个字典,键是文本解释,值是特征
    '''
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': _int64_feature(height),
        'image/width': _int64_feature(width),
        'image/colorspace': _bytes_feature(tf.compat.as_bytes(colorspace)),
        'image/channels': _int64_feature(channels),
        'image/class/label': _int64_feature(label),
        'image/class/text': _bytes_feature(tf.compat.as_bytes(text)),
        'image/format': _bytes_feature(tf.compat.as_bytes(image_format)),
        #os.path.basename(): 返回文件名和后缀 例如1.jpg
        #tf.compat.as_bytes(): 变成bytes数组
        'image/filename': _bytes_feature(tf.compat.as_bytes(os.path.basename(filename))),
        'image/encoded': _bytes_feature(tf.compat.as_bytes(image_buffer))}))
    return example


class ImageCoder(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self):
        # 创建一个会话
        self._sess = tf.Session()

        '''
        encode编码指的是将rgb(240,240,3)编码成byte数组
        decode解码指的是将byte数组解码成rgb(240, 240, 3)
        '''
        # 我们希望图片都是rgb格式的(希望是jpg文件,而不是png文件),所以这里对所有图片进行一个rgb的解码
        self._png_data = tf.placeholder(dtype=tf.string)
        image = tf.image.decode_png(self._png_data, channels=3) #png解码
        #quality: An optional int. Defaults to 95. Quality of the compression from 0 to 100
        #jpeg编码(higher is better and slower).
        self._png_to_jpeg = tf.image.encode_jpeg(image, format='rgb', quality=100)

        # 用来解码rgb的JPG图像
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)
    #将png解码成RGB png,再编码成jpg
    def png_to_jpeg(self, image_data):
        return self._sess.run(self._png_to_jpeg,
                              feed_dict={self._png_data: image_data})
    #将jpg文件解码成RGB jpg
    def decode_jpeg(self, image_data):
        image = self._sess.run(self._decode_jpeg,
                               feed_dict={self._decode_jpeg_data: image_data})
        assert len(image.shape) == 3 #对图像是否已经解码成RGB jpg进行断言
        assert image.shape[2] == 3
        return image #返回解码后的RGB jpg


def _is_png(filename):
    """看看图像是否含有.png字符串

  Args:
    filename: string, path of the image file.

  Returns:
    boolean indicating if the image is a PNG.
  """
    return '.png' in filename


def _process_image(filename, coder):
    """Process a single image file.

  Args:
    filename: string, path to an image file e.g., '/path/to/example.JPG'.
    coder: instance of ImageCoder to provide TensorFlow image coding utils.
  Returns:
    image_buffer: string, JPEG encoding of RGB image.
    height: integer, image height in pixels.
    width: integer, image width in pixels.
  """
    # 读取图像文件,rb代表非UTF-8编码
    with tf.gfile.FastGFile(filename, 'rb') as f:
        image_data = f.read() #读取图像文件(字节数组)

    # Convert any PNG to JPEG's for consistency.
    if _is_png(filename): #只要图像是.png文件
        print('Converting PNG to JPEG for %s' % filename)
        image_data = coder.png_to_jpeg(image_data) #使用coder对象中的函数将png转化为jpg

    # 将jpg文件解码成RGB格式
    #image是ndarray类型,shape是类似于(240,240,3)的数据
    image = coder.decode_jpeg(image_data)

    # 对图像是否是RGB jpg进行断言
    assert len(image.shape) == 3
    height = image.shape[0] #获取图像高度
    width = image.shape[1] #获取图像宽度
    assert image.shape[2] == 3

    return image_data, height, width
#每一个线程,使用这个函数生成若干tfrecord文件
#假设要生成8个tfrecord文件,使用2线程,则这个函数中要生成4个tfrecord,且一共被调用2次
def _process_image_files_batch(coder, thread_index, ranges, name, filenames,
                               texts, labels, num_shards):
    """Processes and saves list of images as TFRecord in 1 thread.

  Args:
    coder: instance of ImageCoder to provide TensorFlow image coding utils.
    thread_index: integer, unique batch to run index is within [0, len(ranges)).
    ranges: list of pairs of integers specifying ranges of each batches to
      analyze in parallel.
    name: string, unique identifier specifying the data set
    filenames: list of strings; each string is a path to an image file
    texts: list of strings; each string is human readable, e.g. 'dog'
    labels: list of integer; each integer identifies the ground truth
    num_shards: integer number of shards for this data set.
  """
    # Each thread produces N shards where N = int(num_shards / num_threads).
    # For instance, if num_shards = 128, and the num_threads = 2, then the first
    # thread would produce shards [0, 64).
    num_threads = len(ranges) #两个线程
    assert not num_shards % num_threads #tfrecord可以平分分配给几个线程
    num_shards_per_batch = int(num_shards / num_threads) #每个线程均分多少tfrecord文件

    shard_ranges = np.linspace(ranges[thread_index][0], #此tfrecord文件对应照片的索引
                               ranges[thread_index][1],
                               num_shards_per_batch + 1).astype(int)
    num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0] #一个线程有多少张图片

    counter = 0
    for s in range(num_shards_per_batch): #每个线程负责生成多少个tfrecord文件
        # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
        shard = thread_index * num_shards_per_batch + s #第几个tfrecord文件
        output_filename = '%s-%.5d-of-%.5d.tfrecord' % (name, shard, num_shards) #tfrecord文件名
        output_file = os.path.join(FLAGS.output_directory, output_filename) #tfrecord文件全路径
        writer = tf.python_io.TFRecordWriter(output_file) #生成tfrecord文件的文件写入器

        shard_counter = 0
        files_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int) #生成索引,前含后不含
        for i in files_in_shard:
            filename = filenames[i] #一张张的图片全路径
            label = labels[i] #一张张的照片数字标签
            text = texts[i] #一张张的照片文本标签

            image_buffer, height, width = _process_image(filename, coder)
            #得到图片的信息
            example = _convert_to_example(filename, image_buffer, label,
                                          text, height, width)
            #tf.train.Example()返回是example,使用examplt.SerializeToString()实现写入文件
            writer.write(example.SerializeToString()) #将图片信息写入文件
            shard_counter += 1
            counter += 1

            if not counter % 1000: #如果能被1000整除,print信息告知已经将1000张图片制作成数据集
                print('%s [thread %d]: Processed %d of %d images in thread batch.' %
                      (datetime.now(), thread_index, counter, num_files_in_thread))
                sys.stdout.flush()

        writer.close()
        print('%s [thread %d]: Wrote %d images to %s' %
              (datetime.now(), thread_index, shard_counter, output_file))
        sys.stdout.flush()
        shard_counter = 0
    print('%s [thread %d]: Wrote %d images to %d shards.' %
          (datetime.now(), thread_index, counter, num_files_in_thread))
    sys.stdout.flush()


def _process_image_files(name, filenames, texts, labels, num_shards):
    """Process and save list of images as TFRecord of Example protos.

  Args:
    name: string, unique identifier specifying the data set
    filenames: list of strings; each string is a path to an image file
    texts: list of strings; each string is human readable, e.g. 'dog'
    labels: list of integer; each integer identifies the ground truth
    num_shards: integer number of shards for this data set.
  """
    assert len(filenames) == len(texts) #确定三个数字必须一一对应起来,长度都是3670
    assert len(filenames) == len(labels)

    # 将3670张图像按照线程数切分成几个batches
    spacing = np.linspace(0, len(filenames), FLAGS.num_threads + 1).astype(np.int)
    ranges = [] #两个线程操作照片的索引: [[0, 1835], [1835, 3670]]
    for i in range(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i + 1]])

    # 两个线程分别处理0到1835和1835到3670
    print('Launching %d threads for spacings: %s' % (FLAGS.num_threads, ranges))
    sys.stdout.flush() #手动刷新缓冲区(多线程可以实时看到print出来的信息)

    # 创建一个线程管理器（协调器）对象,用来监视所有线程是否已经结束
    coord = tf.train.Coordinator()

    # Create a generic TensorFlow-based utility for converting all image codings.
    coder = ImageCoder()

    threads = []
    for thread_index in range(len(ranges)):
        args = (coder, thread_index, ranges, name, filenames,
                texts, labels, num_shards) #target中的函数我们用一个线程来做,args就是此函数的参数
        t = threading.Thread(target=_process_image_files_batch, args=args)
        t.start() #运行线程
        threads.append(t) #将线程添加到线程列表中

    # 把threads中的线程加入主线程,等待所有线程都结束
    coord.join(threads)
    print('%s: Finished writing all %d images in data set.' %
          (datetime.now(), len(filenames)))
    sys.stdout.flush()


def _find_image_files(data_dir, labels_file):
    """在数据集中创建一个所有图像文件和标签的列表

  Args:
    data_dir: string, 图像文件夹根目录

      假设图像数据集时JPEG文件格式
      且在下面的文件夹中:

        data_dir/dog/another-image.JPEG
        data_dir/dog/my-image.jpg

      这些图像的标签都是狗

    labels_file: string, 分类标签文件路径

      用来验证的标签写在这个文件中,
      想象文件中存在以下三种分类:
        dog
        cat
        flower
      其中每一行对应一个标签,我们将此标签文件中包含的每个标签和
      一个以0开头的整数相对应,而这个整数对应于
      包含在第一行中的标签.

  Returns:
    filenames: list of strings; 这些字符串是一个图像文件的路径
    texts: list of strings; 这些字符串是分类标签
    labels: list of integer; each integer identifies the ground truth.
  """
    print('目标文件夹位置： %s.' % data_dir)
    unique_labels = [l.strip() for l in tf.gfile.FastGFile(
        labels_file, 'r').readlines()] #读取文件的每一行,去掉空格,重构成列表,unique_labels是5个标签

    labels = []
    filenames = []
    texts = []

    # 五分类问题,索引从1到5,0为空
    label_index = 1

    # Construct the list of JPEG files and labels.
    for text in unique_labels: #text是五种花的类名
        jpeg_file_path = '%s/%s/*' % (data_dir, text)
        try:
            matching_files = tf.gfile.Glob(jpeg_file_path) #获取其中一种花的文件夹下所有文件
        except:
            print(jpeg_file_path)
            continue

        labels.extend([label_index] * len(matching_files)) #将[1]复制633份(这种花有633张图片)
        texts.extend([text] * len(matching_files)) #将[daisy]复制633份(正在处理daisy种类的花)
        filenames.extend(matching_files) #将matching_files中633张图像的相对路径添加进来

        label_index += 1

    # 经过上面的循环,labels代表3670张图片的标签数字索引(1到5),filenames代表3670张图片的相对路径
    # texts代表3670张图片的标签文字(daisy等).但是所有图像的这三个列表都是按顺序的,不利于训练,要打乱顺序
    # saved TFRecord files. Make the randomization repeatable.
    shuffled_index = list(range(len(filenames))) #从0到3669的索引
    random.seed(12345) #如果每次随机不同,可能会对训练效果有影响,所以指定随机种子,使得每次随机结果相同
    random.shuffle(shuffled_index) #使用随机种子对索引0到3669进行洗牌

    filenames = [filenames[i] for i in shuffled_index] #对文件名进行洗牌
    texts = [texts[i] for i in shuffled_index] #对标签文本进行洗牌
    labels = [labels[i] for i in shuffled_index] #对标签数字进行洗牌

    print('Found %d JPEG files across %d labels inside %s.' % #打印相关信息
          (len(filenames), len(unique_labels), data_dir))
    return filenames, texts, labels #将数据集图像相关信息返回


def _process_dataset(name, directory, num_shards, labels_file):
    """Process a complete data set and save it as a TFRecord.

  Args:
    name: string, 现在是在训练还是验证
    directory: string, 数据集根目录
    num_shards: integer 制作数据集使用的线程数
    labels_file: string, 分类标签文件
  """
    filenames, texts, labels = _find_image_files(directory, labels_file) #找到并处理图像文件
    _process_image_files(name, filenames, texts, labels, num_shards) #制作数据源


def main(unused_argv):
    assert not FLAGS.train_shards % FLAGS.num_threads, (
        '在测试集中：线程数量应用建立文件个数相对应')
    #not后面是0,0是假,not 0是真,才不会退出
    assert not FLAGS.validation_shards % FLAGS.num_threads, (
        '在测试集中：线程数量应用建立文件个数相对应')
    print('生成数据文件夹 %s' % FLAGS.output_directory)

    # 训练集文件夹,训练使用进程数,分类标签文件
    _process_dataset('train', FLAGS.train_directory,
                     FLAGS.train_shards, FLAGS.labels_file)


"""
  _process_dataset('validation', FLAGS.validation_directory,
                   FLAGS.validation_shards, FLAGS.labels_file)
"""

if __name__ == '__main__':
    tf.app.run()
