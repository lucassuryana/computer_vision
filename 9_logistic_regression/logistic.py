import tensorflow as tf
from solution.utils import check_softmax, check_acc, check_model, check_ce

def softmax(logits):
    """
    softmax implementation
    args:
    - logits [tensor]: 1xN logits tensor
    returns:
    - soft_logits [tensor]: softmax of logits
    """
    exp = tf.exp(logits)
    # tf.math.reduce_sum means summing over the second axis
    # the second axis is the classes axis
    # the expected shape of logits is 1xN
    # for example, logits = [[0.5, 1.0, 2.0, 0.3, 4.0]]
    # example exp = [[logits[0][0], logits[0][1], logits[0][2], logits[0][3], logits[0][4]]]
    # example denom = [[8.712956]]
    denom = tf.math.reduce_sum(exp, 1, keepdims=True)
    return exp / denom


def cross_entropy(scaled_logits, one_hot):
    """
    Cross entropy loss implementation
    args:
    - scaled_logits [tensor]: NxC tensor where N batch size / C number of classes
    - one_hot [tensor]: one hot tensor
    returns:
    - loss [tensor]: cross entropy
    """
    # tf.boolean_mask is used to mask the logits
    # the score of scaled_logits is between 0 and 1, total sum is 1
    # example scaled_logits = [[0.1, 0.2, 0.3, 0.4, 0.5]]
    # example one_hot = [[0, 0, 0, 0, 1]]
    # example masked_logits = [[0.5]]
    masked_logits = tf.boolean_mask(scaled_logits, one_hot)
    return -tf.math.log(masked_logits)


def model(X, W, b):
    """
    logistic regression model
    args:
    - X [tensor]: input HxWx3
    - W [tensor]: weights
    - b [tensor]: bias
    returns:
    - output [tensor]
    """
    # reshape X from HxWx3 to 1x(H*W*3)
    # H is the height of the image
    # W is the width of the image
    # 3 is the number of channels, RGB
    # weights W is a matrix of (H*W*3)xC
    # W is the weights of the model
    # W.shape[0] is the number of inputs
    flatten_X = tf.reshape(X, (-1, W.shape[0]))
    return softmax(tf.matmul(flatten_X, W) + b)


def accuracy(y_hat, Y):
    """
    calculate accuracy
    args:
    - y_hat [tensor]: NxC tensor of models predictions
    - y [tensor]: N tensor of ground truth classes
    returns:
    - acc [tensor]: accuracy
    """
    # tf.argmax returns the index of the maximum value
    argmax = tf.cast(tf.argmax(y_hat, axis=1), Y.dtype)

    # calculate acc
    # tf.math.reduce_sum is used to sum the correct predictions
    # tf.cast is used to cast the boolean to int
    # for example, argmax = [4, 1]
    # for example, Y = [4, 1]
    # for example, acc = 2/2 = 1.0
    # tf.cast is used to cast the boolean to int
    # to cast the boolean means to convert True to 1 and False to 0
    acc = tf.math.reduce_sum(tf.cast(argmax == Y, tf.int32)) / Y.shape[0]
    return acc


if __name__ == '__main__':
    # checking the softmax implementation
    check_softmax(softmax)

    # checking the NLL implementation
    check_ce(cross_entropy)

    # check the model implementation
    check_model(model)

    # check the accuracy implementation
    check_acc(accuracy)