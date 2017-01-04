import os, sys, subprocess, time
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

import tinkerbell.logger as logger
import rl_algs.common.tf_util as U
from rl_algs.common.mpi_adam import MpiAdam
from rl_algs.common.mpi_fork import mpi_fork
from rl_algs.common import set_global_seeds

from mpi4py import MPI
num_cpu = 8

LOG_DIR = "ramdisk/"   # careful, will erase recursively subdirs there (and yes you can mount tmpfs here if you have a lot of memory)
MAX_STEPS = 10000
BATCH = 64 / 8

class Dataset:
    def __init__(self):
        self.cursor = 1e10
        self.epoch = 0
    def next_batch(self, rank, rank_count):
        if self.cursor+BATCH*rank_count > self.x_train.shape[0]:
            sh = np.random.choice( self.x_train.shape[0], size=self.x_train.shape[0], replace=False )
            # sh is list of indexes
            self.shuffled_x_train = self.x_train[sh]
            self.shuffled_y_train = self.y_train[sh]
            self.cursor = 0
            self.epoch += 1
        #cnt    = BATCH / rank_count
        #offset = rank * cnt
        x = self.shuffled_x_train[self.cursor+BATCH*rank : self.cursor+BATCH*(rank+1)]
        y = self.shuffled_y_train[self.cursor+BATCH*rank : self.cursor+BATCH*(rank+1)]
        #print("randomtest[%i]==%s" % (rank, self.shuffled_y_train[0]))
        self.cursor += BATCH*rank_count
        return x, y

def mnist():
    from tensorflow.examples.tutorials.mnist import input_data
    dataset = Dataset()
    dataset.LABELS = 10
    dataset.LABELS_TEXT = "0 1 2 3 4 5 6 7 8 9".split()
    dataset.H = 28
    dataset.W = 28
    dataset.COLORS = 1
    mnist = input_data.read_data_sets("ramdisk/", one_hot=True)
    dataset.x_train = np.zeros( (len(mnist.train.images),dataset.H,dataset.W,1) )
    dataset.x_train[:,0:28,0:28,:] = mnist.train.images.reshape( (-1,28,28,1) )
    dataset.y_train = mnist.train.labels
    dataset.x_test  = np.zeros( (len(mnist.test.images),dataset.H,dataset.W,1) )
    dataset.x_test[:,0:28,0:28,:] = mnist.test.images.reshape( (-1,28,28,1) )
    dataset.y_test  = mnist.test.labels
    return dataset

def discriminator_network(x, learnable):
    h, w, features = x.get_shape().as_list()[-3:]
    print("discriminator input h w f = %i %i %i" % (h,w,features))
    x = tf.nn.relu( U.conv2d(x,  32, "conv1",  [3,3], [1,1], learnable=learnable) )
    x = tf.nn.relu( U.conv2d(x,  64, "conv21", [3,3], [2,2], learnable=learnable) )
    x = tf.nn.relu( U.conv2d(x,  64, "conv22", [3,3], [1,1], learnable=learnable) )
    x = tf.nn.relu( U.conv2d(x, 128, "conv31", [3,3], [2,2], learnable=learnable) )
    x = tf.nn.relu( U.conv2d(x, 128, "conv32", [3,3], [1,1], learnable=learnable) )
    x = tf.nn.relu( U.conv2d(x, 256, "conv4",  [3,3], [2,2], learnable=learnable) )
    h, w, features = x.get_shape().as_list()[-3:]
    print("discriminator final h w f = %i %i %i -> flat %i" % (h,w,features, h*w*features))
    x = tf.reshape(x, [-1,h*w*features])
    x = tf.nn.relu( U.dense(x, 256, "dense1", weight_init=U.normc_initializer(1.0), learnable=learnable) )
    return x

def do_all(dataset):
    rank = MPI.COMM_WORLD.Get_rank()

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    config.intra_op_parallelism_threads = 1
    config.inter_op_parallelism_threads = 1
    sess = tf.InteractiveSession(config=config)

    input = tf.placeholder(tf.float32, [None,dataset.H,dataset.W,1], name='image_input')
    gt    = tf.placeholder(tf.float32, [None,dataset.LABELS], name='ground_truth')

    learnable = []
    disc_code = discriminator_network(input, learnable)
    disc_10way = U.dense(disc_code, dataset.LABELS, "dense10", weight_init=U.normc_initializer(1.0), learnable=learnable)

    with tf.name_scope("classification"):
        disc_loss     = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(disc_10way, gt))
        disc_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(disc_10way, 1), tf.argmax(gt, 1)), tf.float32))
        tf.summary.scalar('disc_loss', disc_loss)
        tf.summary.scalar('disc_accuracy', disc_accuracy)

    merged = tf.summary.merge_all()
    if rank==0:
        train_writer = tf.summary.FileWriter(LOG_TRAIN_DIR, sess.graph)
        test_writer  = tf.summary.FileWriter(LOG_TEST_DIR)
    else:
        train_writer = None
        test_writer = None

    adam = MpiAdam(learnable, 0.00005, beta1=0.5)
    summary_loss_and_grad = U.function([input, gt], [merged, disc_loss, U.flatgrad(disc_loss, learnable)])

    #with tf.name_scope('adam_discriminator'):
    #    adam_discriminator = tf.train.AdamOptimizer(0.00005, beta1=0.5).minimize(
    #        disc_loss,
    #        var_list=learnable)

    tf.global_variables_initializer().run()
    ts1 = 0
    for step in range(MAX_STEPS):
        run_options = None
        run_metadata = None
        summary = None
        if step % 100 == 0 and rank==0:
            #print("testing...", flush=True)
            ts2 = time.time()
            #summary, loss, acc = sess.run( [merged, disc_loss, disc_accuracy], feed_dict = { input: dataset.x_test, gt: dataset.y_test })
            #test_writer.add_summary(summary, step)
            #if ts1:
            #    print('e%02i:%05i %0.2fms loss=%0.2f acc=%0.2f%%' % (dataset.epoch, step, 1000*(ts2-ts1), loss, acc*100))
            ts1 = time.time()
            run_options  = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()

        d_real, d_labels = dataset.next_batch(MPI.COMM_WORLD.Get_rank(), MPI.COMM_WORLD.Get_size())

        summary, loss, grad = summary_loss_and_grad(d_real, d_labels)
        adam.update(grad)
        #print("test[%i]==%0.3f" % (rank, loss), flush=True)

#        summary, loss = sess.run(
#            [merged, adam_discriminator],
#            feed_dict = {
#                input: d_real,
#                gt: d_labels,
#                },
#            options=run_options,
#            run_metadata=run_metadata)

        if run_metadata:
            train_writer.add_run_metadata(run_metadata, 'step%05i' % step)
        if summary and rank==0:
            train_writer.add_summary(summary, step)

    if rank==0:
        train_writer.close()
        test_writer.close()

if __name__ == '__main__':
    import shutil, os
    LOG_TEST_DIR  = LOG_DIR + "/test_%s" % sys.argv[1]
    LOG_TRAIN_DIR = LOG_DIR + "/train_%s" % sys.argv[1]
    if os.getenv("IN_MPI") is None:
        shutil.rmtree(LOG_TEST_DIR,  ignore_errors=True)
        shutil.rmtree(LOG_TRAIN_DIR, ignore_errors=True)
        os.makedirs(LOG_TEST_DIR)
        os.makedirs(LOG_TRAIN_DIR)
    me = mpi_fork(num_cpu)
    if me=="child":
        np.random.seed(0)
        dataset = mnist()
        do_all(dataset)
