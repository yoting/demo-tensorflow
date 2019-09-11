import os
import tensorflow as tf

num_classes = 10
num_examples_per_epoch_for_train = 50000
num_examples_per_epoch_for_eval = 10000


class CIFAR10Record(object):
    pass

    def read_cifar10(self, file_queue):
        result = CIFAR10Record()
        label_bytes = 1
        result.height = 32
        result.width = 32
        result.depth = 3
        image_bytes = result.height * result.width * result.depth

        # FixedLengthRecorder.__init__(self, record_bytes, header_bytes, footer_bytes, name):
        reader = tf.FixedLengthRecordReader(record_bytes=(label_bytes + image_bytes))
        result.key, value = reader.read(file_queue)
        record_bytes = tf.decode_raw(value, tf.uint8)

        result.label = tf.cast(tf.strided_slice(record_bytes, [0], [label_bytes]), dtype=tf.int32)

        depth_major = tf.reshape(tf.strided_slice(record_bytes, [label_bytes], [label_bytes + image_bytes]),
                                 [result.depth, result.height, result.width])
        result.uint8image = tf.transpose(depth_major, [1, 2, 0])  # 将depth,height,width改为height，width，depth

        return result

    def input(self, data_dir, batch_size, distorted):
        filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i) for i in range(1, 6)]
        # 创建文件队列并且读取文件
        file_queue = tf.train.string_input_producer(filenames)
        read_input = self.read_cifar10(file_queue)

        # 将图片数据转换为float32格式
        reshaped_image = tf.cast(read_input.uint8image, tf.float32)
        num_examples_per_epoch = num_examples_per_epoch_for_train

        if distorted != None:  # 对图像进行增强处理
            # 裁剪图片
            cropped_image = tf.random_crop(reshaped_image, [24, 24, 3])
            # 反转图片
            flipped_image = tf.image.random_flip_left_right(cropped_image)
            # 调整亮度
            adjusted_brightness = tf.image.random_brightness(flipped_image, max_delta=0.8)
            # 调整对比度
            adjusted_contrast = tf.image.random_contrast(adjusted_brightness, lower=0.2, upper=1.8)
            # 标准化图片
            float_image = tf.image.per_image_standardization(adjusted_contrast)  # 对每个像素点减去平均值并除以像素方差

            # 设置图下形状和label形状
            float_image.set_shape([24, 24, 3])
            read_input.label.set_shape([1])

            min_queue_examples = int(num_examples_per_epoch_for_eval * 0.4)
            print(
                'Filling queue with %d CIFAR images before staring to train. This will take a few minutes.' % min_queue_examples)
            # 随机产生一个batch的image和label
            image_train, labels_train = tf.train.shuffle_batch([float_image, read_input.label], batch_size=batch_size,
                                                               num_threads=16,
                                                               capacity=min_queue_examples + 3 * batch_size,
                                                               min_after_dequeue=min_queue_examples)
            return image_train, tf.reshape(labels_train, [batch_size])
        else:  # 不对图像进行增强处理
            # 改变形状
            resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image, 24, 24)
            # 直接标准化
            float_image = tf.image.per_image_standardization(resized_image)
            float_image.set_shape([24, 24, 3])
            read_input.label.set_shape([1])
            min_queue_examples = int(num_examples_per_epoch * 0.4)
            image_test, labels_test = tf.train.batch([float_image, read_input.label], batch_size=batch_size,
                                                     num_threads=16, capacity=min_queue_examples + 3 * batch_size)
            return image_test, tf.reshape(labels_test, [batch_size])


if __name__ == '__main__':
    record = CIFAR10Record()
    record.input(data_dir='data/cifar-10', batch_size=100, distorted=None)
    print(record)
