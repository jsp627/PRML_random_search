from os.path import join
from struct import unpack
from numpy import fromfile, int8, uint8


def load_mnist(dataset = "training", path = "."):
    """
    Python function for importing the MNIST data set.  It returns an iterator
    of 2-tuples with the first element being the label and the second element
    being a numpy.uint8 2D array of pixel data for the given image.
    """
    # source: https://gist.github.com/akesling/5358964

    if dataset is "training":
        fname_img = join(path, 'train-images.idx3-ubyte')
        fname_lbl = join(path, 'train-labels.idx1-ubyte')
    elif dataset is "testing":
        fname_img = join(path, 't10k-images.idx3-ubyte')
        fname_lbl = join(path, 't10k-labels.idx1-ubyte')
    else:
        raise(ValueError, "dataset must be 'testing' or 'training'")

    # Load everything in some numpy arrays
    with open(fname_lbl, 'rb') as flbl:
        magic, num = unpack(">II", flbl.read(8))
        lbl = fromfile(flbl, dtype=int8)

    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = unpack(">IIII", fimg.read(16))
        img = fromfile(fimg, dtype=uint8).reshape(len(lbl), rows*cols)

    return lbl, img


if __name__ == '__main__':
    mnist_path = 'Data\\MNIST'
    mnist_tr = load_mnist(dataset='training', path = mnist_path)
    import scipy.misc

    for i in range(36):
        scipy.misc.imsave('x{}.png'.format(i), mnist_tr[1][i, :].reshape([28, 28]))
