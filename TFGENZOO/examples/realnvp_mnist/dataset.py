import matplotlib.pyplot as plt
import tensorflow as tf
import os

# From tensorflow's BEGINNER TUTORIALS


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.tobytes()]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


@tf.function
def _preprocess(x):
    x = tf.pad(x, paddings=[[0, 0], [2, 2], [2, 2]], mode="CONSTANT")
    x = tf.expand_dims(x, axis=-1)
    return x


def create_mnist():
    (train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()
    print('train_dataset: {} images'.format(train_x.shape[0]))
    print('test_dataset : {} images'.format(test_x.shape[0]))
    if not os.path.exists("mnists"):
        os.mkdir("mnists")

    def _serialize_example_pyfunction(img, label):
        feature = {
            "img": _bytes_feature(img.numpy()),
            "label": _int64_feature(label.numpy()),
        }
        example_proto = tf.train.Example(
            features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()

    @tf.function
    def _gen_tf_serialize_example(img, label):
        tf_string = tf.py_function(
            _serialize_example_pyfunction, (img, label), tf.string
        )
        return tf.reshape(tf_string, ())

    def _save_features_dataset(save_to, feature_dataset):
        serialized_feature_dataset = train_features_dataset.map(
            _gen_tf_serialize_example
        )
        writer = tf.data.experimental.TFRecordWriter(save_to)
        writer.write(serialized_feature_dataset)

    train_x = _preprocess(train_x)
    train_features_dataset = tf.data.Dataset.from_tensor_slices(
        (train_x, train_y))
    _save_features_dataset(os.path.join(
        'mnists', 'train.tfrecord'), train_features_dataset)

    test_x = _preprocess(test_x)
    test_features_dataset = tf.data.Dataset.from_tensor_slices(
        (test_x, test_y))
    _save_features_dataset(os.path.join(
        'mnists', 'test.tfrecord'), test_features_dataset)


def _parse_function(example_proto):
    feature_description = {
        'img': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64, default_value=0)
    }
    feature = tf.io.parse_single_example(example_proto, feature_description)
    img = tf.io.decode_raw(feature['img'], out_type=tf.uint8)
    img = tf.reshape(img, [32, 32, 1])
    img = tf.cast(img, dtype=tf.float32)
    # img = (img / (255.0 / 2)) - 1
    feature['img'] = img
    return feature


def read_mnist_dataset(show_example: bool = True):
    filenames = [os.path.join('mnists', 'train.tfrecord')]
    raw_dataset = tf.data.TFRecordDataset(filenames)
    parsed_dataset = raw_dataset.map(_parse_function)
    if show_example:
        for img in parsed_dataset.take(1):
            plt.imshow(tf.squeeze(img['img'], axis=-1), cmap='gray_r')
            plt.savefig('test.png')
    test_filenames = [os.path.join('mnists', 'test.tfrecord')]
    test_raw_dataset = tf.data.TFRecordDataset(test_filenames)
    test_parsed_dataset = test_raw_dataset.map(_parse_function)
    return parsed_dataset, test_parsed_dataset
