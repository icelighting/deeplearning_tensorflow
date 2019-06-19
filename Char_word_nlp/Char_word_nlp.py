import numpy as np
import tensorflow as tf
import math
import time
import os

def pick_top_k(preds, vocabulary_size, top_k=5):
    p = np.squeeze(preds)#从数组的形状中删除单维度条目，即把shape中为1的维度去掉，其实对原始数据或者在tf都可以一样处理
    p[np.argsort(p)[:-top_k]] = 0
    p = p / sum(p)
    c = np.random.choice(vocabulary_size, 1, p=p)[0]
    return c

class CharModel(object):
    def __init__(self, num_class,lstm_size = 128, embedding_size=128, num_seq= 64,
                 num_step = 50, num_layers = 2, grad_clip=5, sampling=False, train_keep_prob=0.5, is_English=True, is_Training=True):
        '''
        :param num_class: 词的种类，样本中有多少种不同的单词
        :param batch_size: batch 数目， 一次传入的sequence数目 对应batch的行数
        :param lstm_size: lstm中神经元的数量
        :param embedding_size: #词向量embedding后的维度
        :param num_seq: ## 一个batch中传入的句子的数量
        :param num_step: ## RNN的横向cell数量
        :param num_layers: CELL的纵向堆叠次数
        :param grad_clip: 梯度裁剪参数
        :param sampling: 对应生成样本过程
        :param train_keep_prob: dropout参数， 如果过拟合可以减少数值
        :param is_English: 是否英文，英文不需要embedding， 中文需要
        :param is_Training: ##是否是training的过程，对应是否添加，droupout
        '''
        if sampling:
            num_seq, num_step = 1, 1

        self.num_class = num_class
        #self.batch_size = batch_size
        self.lstm_size = lstm_size
        self.hidden_size = lstm_size
        self.embedding_size = embedding_size
        self.num_seqs = num_seq
        self.num_steps = num_step
        self.num_layers = num_layers
        self.grad_clip = grad_clip
        self.train_keep_prob = train_keep_prob
        self.is_English = is_English
        self.is_Training = is_Training

        tf.reset_default_graph()
        self.add_placeholder()
        self.build_lstm()
        self.build_loss()
        self.build_optimizer()
        self.saver = tf.train.Saver()


    def add_placeholder(self,):
        self.input_placeholder = tf.placeholder(tf.int32, shape=[self.num_seqs, self.num_steps], name='Inputs')
        self.target_placeholder = tf.placeholder(tf.int32, shape=[self.num_seqs, self.num_steps], name='Target')

        #英文的话，不需要embedding, 中文需要
        if not self.is_English:
            with tf.device('/cpu:0'):
                embeddings = tf.Variable(
                    tf.random_uniform([self.num_class, self.embedding_size], -1.0, 1.0))
                #embeddings = tf.get_variable('embedding', [self.num_class, self.embedding_size])
                self.model_inputs = tf.nn.embedding_lookup(embeddings, self.input_placeholder)
        else:
            self.model_inputs = tf.one_hot(self.input_placeholder, self.num_class)


    def build_lstm(self,):
        def lstm_cell():
            lstm = tf.nn.rnn_cell.BasicLSTMCell(
                self.lstm_size, forget_bias = 0, state_is_tuple=True
            )
            #drop = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=self.train_keep_prob)
            return lstm

        attn_cell = lstm_cell
        if self.is_Training:
            def attn_cell():
                # 对于rnn部分不进行dropout，即状态从t-1到t时候进行计算时，中间不进行memory的dropout，
                #仅仅在同一时刻中，多层cell之间传递信息的时候drop out
                drop = tf.nn.rnn_cell.DropoutWrapper(lstm_cell(), output_keep_prob=self.train_keep_prob)
                return drop
        with tf.name_scope('lstm'):
            cell = tf.nn.rnn_cell.MultiRNNCell(
                [attn_cell() for _ in range(self.num_layers)]
            )
            self.initial_state = cell.zero_state(self.num_seqs, tf.float32)

            ## 通过dynamic_rnn 对cell 展开时间维度
            self.lstm_ouputs, self.final_state = tf.nn.dynamic_rnn(cell, self.model_inputs, initial_state= self.initial_state)

            ## 通过lstm得到概率
            seq_outputs = tf.concat(self.lstm_ouputs, 1)
            x = tf.reshape(seq_outputs, [-1, self.lstm_size])

            with tf.variable_scope('softmax'):
                softmax_weights = tf.Variable(tf.truncated_normal([self.lstm_size, self.num_class], stddev=0.1))
                #softmax_weights = tf.Variable(tf.truncated_normal([self.lstm_size, self.num_class], stddev=1.0/math.sqrt(self.embedding_size)))
                softmax_bias = tf.Variable(tf.zeros([self.num_class]))

            self.logits = tf.matmul(x, softmax_weights) + softmax_bias
            self.prob_predictions = tf.nn.softmax(self.logits, name='preditions')

    def build_rnn(self):
        def rnn_cell():
            return tf.nn.rnn_cell.BasicRNNCell(self.hidden_size)

        attn_cell = rnn_cell
        if self.is_Training:
            def attn_cell():
                drop = tf.nn.rnn_cell.DropoutWrapper(rnn_cell(),output_keep_prob=self.train_keep_prob)
                return drop

        with tf.name_scope('rnn'):
            cell = tf.nn.rnn_cell.MultiRNNCell(
                [attn_cell() for _ in range(self.num_layers)]
            )

            self.initial_state = cell.zero_state(self.num_seqs,tf.float32)

            ## 通过dynamic_cell展开维度求解
            self.rnn_out , self.final_state = tf.nn.dynamic_rnn(cell, self.model_inputs, initial_state = self.initial_state)

            #
            seq_out = tf.concat(self.rnn_out, axis=1)
            x = tf.reshape(seq_out, [-1, self.hidden_size])

            with tf.variable_scope('softmax'):
                softmax_w = tf.Variable(tf.truncated_normal([self.hidden_size, self.num_class], stddev= 1.0/ math.sqrt(self.hidden_size + self.num_class)))
                softmax_b = tf.Variable(tf.zeros([self.num_class]))

            self.logits = tf.matmul(x, softmax_w) + softmax_b
            self.prob_predictions = tf.nn.softmax(logits=self.logits, name='predictions')

    def build_loss(self):
        with tf.variable_scope('loss'):
            y_one_hot = tf.one_hot(self.target_placeholder, self.num_class)
            y_reshape = tf.reshape(y_one_hot, self.logits.get_shape())
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=y_reshape))

    def build_optimizer(self):
        ## 使用 clip_gradients 确保梯度不发散
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), self.grad_clip)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.01).apply_gradients(zip(grads, tvars))

    def train(self, batch_generator, max_steps, save_path):
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            print('Intialized','.'*10)
            sess.run(init)
            self.saver = tf.train.Saver()
            step = 0
            new_state = sess.run(self.initial_state)
            for x, y in batch_generator:
                step += 1
                start = time.time()
                feed = {
                    self.input_placeholder:x,
                    self.target_placeholder:y,
                    self.initial_state:new_state
                        }
                batch_loss, new_state, _ = sess.run([self.loss,
                                                     self.final_state,
                                                     self.optimizer],
                                                    feed_dict=feed)

                end = time.time()
                # control the print lines
                if step % 100 == 0:
                    print('step: {}/{}... '.format(step, max_steps),
                          'loss: {:.4f}... '.format(batch_loss),
                          '{:.4f} sec/batch'.format((end - start)))
                if (step % 1000 == 0):
                    self.saver.save(sess, os.path.join(save_path, 'model'), global_step=step)
                if step >= max_steps:
                    break
            self.saver.save(sess, os.path.join(save_path, 'model'), global_step=step)

    def sample(self, n_samples, prime, vocab_size):
        samples = [c for c in prime]
        sess = tf.Session()
        init = tf.global_variables_initializer()
        new_state = sess.run(self.initial_state)
        preds = np.ones((vocab_size, ))  # for prime=[]
        for c in prime:
            x = np.zeros((1, 1))
            # 输入单个字符
            x[0, 0] = c
            feed = {self.input_placeholder: x,
                    #self.keep_prob: 1.,
                    self.initial_state: new_state}
            preds, new_state = sess.run([self.prob_predictions, self.final_state],
                                        feed_dict=feed)

        c = pick_top_k(preds, vocab_size)
        # 添加字符到samples中
        samples.append(c)

        # 不断生成字符，直到达到指定数目
        for i in range(n_samples):
            x = np.zeros((1, 1))
            x[0, 0] = c
            sess.run(init)
            #if not self.is_English:
                #with tf.Session() as sess:
                    #init = tf.global_variables_initializer()
                    #sess.run(init)
            feed = {self.input_placeholder: x,
                    #self.keep_prob: 1.,
                    self.initial_state: new_state}
            preds, new_state = sess.run([self.prob_predictions, self.final_state],
                                        feed_dict=feed)

            c = pick_top_k(preds, vocab_size)
            samples.append(c)

        return np.array(samples)

    def load(self, checkpoint):
        self.session = tf.Session()
        self.saver.restore(self.session, checkpoint)
        print(tf.trainable_variables())
        print('Restored from: {}'.format(checkpoint))


