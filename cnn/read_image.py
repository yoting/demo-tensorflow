import numpy as np
import pylab

filename = '../data/cifar-10/test_batch.bin'
bytestream = open(filename, "rb")
buf = bytestream.read(10000 * (1 + 32 * 32 * 3))
bytestream.close()

data = np.frombuffer(buf, dtype=np.uint8)
data = data.reshape(10000, 1 + 32 * 32 * 3)
labels_images = np.hsplit(data, [1])  # 将label和image分开
labels = labels_images[0].reshape(10000)
images = labels_images[1].reshape(10000, 32, 32, 3)

for i in range(1000, 1010):
    img = np.reshape(images[i], (3, 32, 32))  # 导出第一幅图
    img = img.transpose(1, 2, 0)

    print(labels[i] + 1)
    pylab.imshow(img)
    pylab.show()
