from make_tfrecord import make_tfrecords

if __name__ == '__main__':
    # PATHS TO IMAGES
    img_path = '/home/jang/Disk_1TB/Dataset/ImageNet2012/train/'

    # PATHS TO TFRECORD
    record_file = '/home/jang/Disk_1TB/Dataset/ImageNet2012/TFRecords_split/train/train_split_{}.tfrecord'

    make_tfrecords(img_path, record_file)
