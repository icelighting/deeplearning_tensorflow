import tensorflow as tf
from read_utils import TextConverter
from Char_word_nlp import CharModel
import os
from IPython import embed

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_integer('lstm_size', 128, 'size of hidden state of lstm')
tf.flags.DEFINE_integer('num_layers', 2, 'number of lstm layers')
tf.flags.DEFINE_boolean('use_embedding', False, 'whether to use embedding')
tf.flags.DEFINE_integer('embedding_size', 128, 'size of embedding')
tf.flags.DEFINE_string('converter_path', 'C:/Users/宁垚/Desktop/tensorflow/Char_word_nlp/model/default/converter.pkl', 'model/name/converter.pkl')
tf.flags.DEFINE_string('checkpoint_path', 'C:/Users/宁垚/Desktop/tensorflow/Char_word_nlp/model/default/', 'checkpoint path')
tf.flags.DEFINE_string('start_string', '', 'use this string to start generating')
tf.flags.DEFINE_integer('max_length', 500, 'max length to generate')


def main(_):
    FLAGS.start_string = FLAGS.start_string.encode('utf-8')
    print('filename', FLAGS.converter_path)
    converter = TextConverter(filename=FLAGS.converter_path)
    if os.path.isdir(FLAGS.checkpoint_path):
        FLAGS.checkpoint_path =\
            tf.train.latest_checkpoint(FLAGS.checkpoint_path)

    model = CharModel(converter.vocab_size,
                    lstm_size=FLAGS.lstm_size, num_layers=FLAGS.num_layers,
                    #use_embedding=FLAGS.use_embedding,
                    is_English=True,
                    embedding_size=FLAGS.embedding_size,
                    sampling=True,
                    is_Training=False)

    #model.add_placeholder()
    #model.build_lstm()
    #model.build_loss()
    #model.build_optimizer()
    model.load(FLAGS.checkpoint_path)
    start = converter.text_to_arr(FLAGS.start_string)
    arr = model.sample(FLAGS.max_length, start, converter.vocab_size)
    print(converter.arr_to_text(arr))


if __name__ == '__main__':
    tf.app.run()
