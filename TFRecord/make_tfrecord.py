from glob import glob
import os
import random
import tensorflow as tf


def serialize_example(image, label):
    feature = {
        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
    }

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def get_file_size(file_path):
    size = os.path.getsize(file_path)
    return size


def make_tfrecords(path, record_file):
    print('start')

    file_index_count = 0
    classes = os.listdir(path)
    classes.sort()
    files_list = glob(path + '/*/*')
    random.shuffle(files_list)
    writer = tf.io.TFRecordWriter(record_file.format(file_index_count))
    print(f'Total image files: {len(files_list)}')
    print(f'TFRecord number: {file_index_count}')

    for filename in files_list:
        image_string = open(filename, 'rb').read()
        category = filename.split('/')[-2]
        label = classes.index(category)
        tf_example = serialize_example(image_string, label)
        writer.write(tf_example)
        # print(f'class:{label}__{filename}')

        size = get_file_size(record_file.format(file_index_count))
        mg_size = round(size / (1024 * 1024), 3)
        # print('File size: ' + str(mg_size) + ' Megabytes')

        if mg_size > 100:
            file_index_count += 1
            writer = tf.io.TFRecordWriter(record_file.format(file_index_count))
            print(f'TFRecord number: {file_index_count}, {files_list.index(filename)}')

    print('done')

