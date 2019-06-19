from read_utils import batch_generator, TextConverter
import tensorflow as tf
from Char_word_nlp import CharModel
import numpy as np
import codecs
import os


FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('name', 'default', 'name of the model')
tf.flags.DEFINE_integer('num_seq', 100, 'number of seqs in one batch')
tf.flags.DEFINE_integer('num_step', 100, 'length of one seq')
tf.flags.DEFINE_integer('lstm_size', 128, 'size of hidden state of lstm')
tf.flags.DEFINE_integer('num_layers', 2, 'number of lstm layers')
tf.flags.DEFINE_boolean('use_embedding', False, 'whether to use embedding')
tf.flags.DEFINE_integer('embedding_size', 128, 'size of embedding')
tf.flags.DEFINE_float('learning_rate', 0.01, 'learning_rate')
tf.flags.DEFINE_float('train_keep_prob', 0.5, 'dropout rate during training')
tf.flags.DEFINE_string('input_file', 'C:/Users/宁垚/Desktop/tensorflow/Char-RNN-TensorFlow-master/Char-RNN-TensorFlow-master/data/jay.txt', 'utf8 encoded text file')
tf.flags.DEFINE_integer('max_steps', 10000, 'max steps to train')
#tf.flags.DEFINE_integer('save_every_n', 1000, 'save the model every n steps')
#tf.flags.DEFINE_integer('log_every_n', 10, 'log to the screen every n steps')
tf.flags.DEFINE_integer('max_vocab', 3500, 'max char number')

def main(_):
    model_path = os.path.join('model', FLAGS.name)
    #print(model_path)
    if os.path.exists(model_path) is False:
        os.makedirs(model_path)
    with codecs.open(FLAGS.input_file, encoding='utf-8') as f:
        text = f.read()
    converter = TextConverter(text, FLAGS.max_vocab)
    converter.save_to_file(os.path.join(model_path, 'converter.pkl'))

    arr = converter.text_to_arr(text)
    g = batch_generator(arr, FLAGS.num_seq, FLAGS.num_step)
    print(converter.vocab_size)
    model = CharModel(converter.vocab_size,
                    num_seq=FLAGS.num_seq,
                    num_step=FLAGS.num_step,
                    lstm_size=FLAGS.lstm_size,
                    num_layers=FLAGS.num_layers,
                    #learning_rate=FLAGS.learning_rate,
                    train_keep_prob=FLAGS.train_keep_prob,
                    #use_embedding=FLAGS.use_embedding,
                    embedding_size=FLAGS.embedding_size,
                    is_Training=True
                    )
    #model.add_placeholder()
    #model.build_lstm()
    #model.build_loss()
    #model.build_optimizer()
    model.train(g,
                FLAGS.max_steps,
                model_path
                )
if __name__ == '__main__':
    tf.app.run()
