import argparse
import tensorflow as tf
import logging

from dataset import get_datasets
from logistic import softmax, cross_entropy, accuracy


def sgd(params, grads, lr, bs):
    """
    Stochastic gradient descent implementation
    Args:
    - params [list[tensor]]: Model parameters
    - grads [list[tensor]]: Parameter gradients
    - lr [float]: Learning rate
    - bs [int]: Batch size
    """
    # param refers to the model parameters [W,b] and grad refers to the gradients.
    # W is the weight matrix and b is the bias vector
    # the parameters can be seen in the following equation:
    # y_hat = softmax(X @ W + b).
    # param.assign_sub is a method that subtracts the value of the tensor passed
    # as an argument from the tensor it is called on.
    # the value of the tensor passed as an argument is the learning rate
    # multiplied by the gradient of the model parameter divided by the batch size
    # for example:
    # W.assign_sub(lr * grad / bs).
    # In SGD equation, W = W - lr * grad / bs
    # this is the update rule for the weight matrix W
    # the same update rule applies to the bias vector b
    for param, grad in zip(params, grads):
        param.assign_sub(lr * grad / bs)


def training_loop(lr):
    """
    Training loop
    Args:
    - lr [float]: Learning rate
    Returns:
    - mean_loss [tensor]: Training loss
    - mean_acc [tensor]: Training accuracy
    """
    accuracies = []
    losses = []
    for X, Y in train_dataset:
        with tf.GradientTape() as tape:
            # Forward pass
            # X is divided by 255 to normalize the pixel values
            X = X / 255.0
            y_hat = model(X)

            # Calculate loss
            # tf.one_hot is used to convert the ground truth labels Y to one-hot encoding
            # for example, if Y = 2, then one_hot = [0, 0, 1, 0, 0, ..., 0]
            one_hot = tf.one_hot(Y, 43)
            # cross_entropy is the loss function used to calculate the difference between
            # the predicted values y_hat and the ground truth labels one_hot
            loss = cross_entropy(y_hat, one_hot)
            losses.append(tf.reduce_mean(loss))

            # Backpropagation
            grads = tape.gradient(loss, [W, b])
            sgd([W, b], grads, lr, X.shape[0])

            # Calculate accuracy
            acc = accuracy(y_hat, Y)
            accuracies.append(acc)

    mean_loss = tf.reduce_mean(losses)
    mean_acc = tf.reduce_mean(tf.stack(accuracies, axis=0))
    return mean_loss, mean_acc


def model(X):
    """
    Logistic regression model
    """
    flatten_X = tf.reshape(X, (-1, W.shape[0]))
    return softmax(tf.matmul(flatten_X, W) + b)


def validation_loop():
    """
    Loop through the validation dataset
    """
    accuracies = []
    for X, Y in val_dataset:
        X = X / 255.0
        y_hat = model(X)
        acc = accuracy(y_hat, Y)
        accuracies.append(acc)

    mean_acc = tf.reduce_mean(tf.stack(accuracies, axis=0))
    return mean_acc


def get_module_logger(mod_name):
    """
    Configures a logger for the module
    """
    logger = tf.get_logger()
    logger.setLevel(logging.INFO)
    return logger


if __name__ == '__main__':
    logger = get_module_logger(__name__)
    parser = argparse.ArgumentParser(description='Train a logistic regression model with TensorFlow')
    parser.add_argument('--imdir', required=True, type=str, help='Data directory')
    parser.add_argument('--epochs', default=10, type=int, help='Number of epochs')
    args = parser.parse_args()

    logger.info(f"Training for {args.epochs} epochs using data from {args.imdir}")

    # Load datasets
    train_dataset, val_dataset = get_datasets(args.imdir)

    # Model parameters
    num_inputs = 1024 * 3  # Assuming input image dimensions
    num_outputs = 43  # Number of classes
    W = tf.Variable(tf.random.normal(shape=(num_inputs, num_outputs), mean=0, stddev=0.01))
    b = tf.Variable(tf.zeros(num_outputs))

    lr = 0.1  # Learning rate

    # Training process
    for epoch in range(args.epochs):
        logger.info(f"Epoch {epoch}")
        loss, acc = training_loop(lr)
        logger.info(f"Mean training loss: {loss:.4f}, Mean training accuracy: {acc:.4f}")
        val_acc = validation_loop()
        logger.info(f"Mean validation accuracy: {val_acc:.4f}")
