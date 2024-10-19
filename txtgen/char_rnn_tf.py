import codecs
import os
import time

import numpy as np
import tensorflow as tf

from lesson9.utils import TextReader, pick_top_n
from lesson9.utils import batch_generator2

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('checkpoint_path', '../content/drive/My Drive/txtgen/model', 'checkpoint path')
tf.flags.DEFINE_string('converter_path', '../content/drive/My Drive/txtgen/converter.pkl', 'converter path')
tf.flags.DEFINE_string('name', 'default', 'the name of the model')
tf.flags.DEFINE_integer('num_seqs', 32, 'number of seqs in batch')
tf.flags.DEFINE_integer('num_seq', 20, 'length of one seq')
tf.flags.DEFINE_integer('lstm_size', 128, 'size of hidden layer')
tf.flags.DEFINE_integer('num_layers', 2, 'number of lstm layers')
tf.flags.DEFINE_integer('embedding_size', 128, 'size of embedding')
tf.flags.DEFINE_float('learning_rate', 0.009, 'learning_rate')
tf.flags.DEFINE_string('input_file', '../content/drive/My Drive/txtgen/test.txt', 'utf-8 encoded input file')
tf.flags.DEFINE_integer('max_steps', 10000, 'max steps of training')
tf.flags.DEFINE_integer('save_model_every', 1000, 'save the model every 1000 steps')
tf.flags.DEFINE_integer('log_every', 50, 'log the summaries every 10 steps')
tf.flags.DEFINE_integer('max_vocab', 3500, 'the maximum of char number')


class CharRNN(object):
    def __init__(self,
                 num_classes,
                 batch_size=64,
                 num_seq=50,
                 lstm_size=128,
                 num_layers=2):

        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_seq = num_seq
        self.rnn_size = lstm_size
        self.num_layers = num_layers
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

    def build_graph(self):
        self._create_inputs()
        self._create_model()
        self._create_loss()
        self._create_optimizer()
        self._create_summary()

    def _create_inputs(self):
        with tf.name_scope('inputs'):
            self.inputs = tf.placeholder(tf.int32, shape=(None, None), name='inputs')
            self.targets = tf.placeholder(tf.int32, shape=(None, None), name='targets')

    def _create_model(self):
        with tf.name_scope('RNN'):
            self.rnn_inputs = tf.one_hot(self.inputs, self.num_classes)
            cell = [tf.nn.rnn_cell.GRUCell(num_units=self.rnn_size) for _ in range(self.num_layers)]
            rnn_cell = tf.nn.rnn_cell.MultiRNNCell(cell)
            self.initial_state = rnn_cell.zero_state(batch_size=tf.shape(self.inputs)[0], dtype=tf.float32)

            self.rnn_outputs, self.final_state = tf.nn.dynamic_rnn(
                cell=rnn_cell, inputs=self.rnn_inputs, initial_state=self.initial_state)

            self.logits = tf.layers.dense(self.rnn_outputs, self.num_classes)

            self.prediction = tf.nn.softmax(logits=self.logits, name='predictions')

    def _create_loss(self):
        with tf.name_scope('loss'):
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.logits, labels=self.targets)
            self.loss = tf.reduce_mean(loss)

    def _create_optimizer(self):
        with tf.name_scope('optimizer'):
            self.optimizer = tf.train.AdamOptimizer().minimize(self.loss, global_step=self.global_step)

    def _create_summary(self):
        with tf.name_scope('summary'):
            tf.summary.scalar(name='loss', tensor=self.loss)
            tf.summary.histogram(name='logit', values=self.logits)
            tf.summary.histogram(name='prediction', values=self.prediction)
            tf.summary.histogram(name='rnn_output', values=self.rnn_outputs)
            self.summary_op = tf.summary.merge_all()

    def train(self, model_path, batch_gen):
        with tf.Session() as sess:

            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())

            if os.path.exists(model_path):
                saver.restore(sess, tf.train.latest_checkpoint(model_path))
                print('model restored!')
            else:
                os.makedirs(model_path)

            writer = tf.summary.FileWriter(model_path + '/tensorboard', sess.graph)

            new_state = sess.run(self.initial_state,
                                 feed_dict={self.inputs: np.zeros([self.batch_size, 128], dtype=np.int32)})

            for x, y in batch_gen:
                start = time.time()
                feed_dict = {
                    self.inputs: x,
                    self.targets: y,
                    self.initial_state: new_state
                }
                _, step, new_state, loss, summary = sess.run(
                    [self.optimizer, self.global_step, self.final_state, self.loss, self.summary_op],
                    feed_dict)
                writer.add_summary(summary, global_step=step)
                end = time.time()
                current_step = tf.train.global_step(sess, self.global_step)
                if step % 50 == 0:
                    print('step: {}/{}... '.format(step, 3500),
                          'loss: {:.4f}... '.format(loss),
                          '{:.4f} sec/batch'.format((end - start)))
                if current_step % 500 == 0:
                    saver.save(sess,
                               os.path.join(model_path, 'model.ckpt'),
                               global_step=current_step)
                if current_step >= 3500:
                    break

                writer.close()

    def inference(self):
        converter = TextReader(filename=FLAGS.converter_path)
        if os.path.isdir(FLAGS.checkpoint_path):
            FLAGS.checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)

        start = converter.text_to_arr('به نام خداوند جان و خرد')
        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, FLAGS.checkpoint_path)
            samples = [c for c in start]
            new_state = sess.run(self.initial_state, feed_dict={self.inputs: np.zeros([1, 1], dtype=np.int32)})
            preds = np.ones((converter.vocab_size,))

            for c in start:
                x = np.zeros((1, 1))
                x[0, 0] = c
                feed_dict = {
                    self.inputs: x,
                    self.initial_state: new_state
                }
                preds, new_state = sess.run(
                    [self.prediction, self.final_state],
                    feed_dict=feed_dict)

            c = pick_top_n(preds, converter.vocab_size)

            samples.append(c)

            for i in range(400):
                x = np.zeros((1, 1))
                x[0, 0] = c
                feed_dict = {
                    self.inputs: x,
                    self.initial_state: new_state
                }
                preds, new_state = sess.run(
                    [self.prediction, self.final_state],
                    feed_dict=feed_dict)
                c = pick_top_n(preds, converter.vocab_size)
                samples.append(c)

            samples = np.array(samples)
            print(converter.arr_to_text(samples))


def main(_):
    if not os.path.exists(FLAGS.name):
        os.makedirs(FLAGS.name)

    model_path = os.path.join(FLAGS.name, 'model')

    with codecs.open(FLAGS.input_file, encoding='utf-8') as f:
        text = f.read()

    print('read file')
    Reader = TextReader(text, FLAGS.max_vocab)
    Reader.save_to_file(os.path.join(FLAGS.name, 'converter.pkl'))

    arr = Reader.text_to_arr(text)
    batch_gen = batch_generator2(arr, FLAGS.num_seqs, FLAGS.num_seq)
    print('build model')

    char_rnn = CharRNN(
        num_classes=Reader.vocab_size,
        batch_size=FLAGS.num_seqs,
        num_seq=FLAGS.num_seq,
        lstm_size=FLAGS.lstm_size,
        num_layers=FLAGS.num_layers)

    char_rnn.build_graph()
    char_rnn.train(model_path, batch_gen)
    char_rnn.inference()


if __name__ == '__main__':
    tf.app.run()
