import torch
import numpy as np
from torch import nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score


class GRU(nn.Module):
    """
    Layered GRU network where number of layers, input-dimension, output-dimension
    and size of hidden-matrix is defined by user

    :ivar hidden_dim: the size of the hidden-matrix
    :ivar num_layers: the number of gru-layers you want applied
    :ivar gru: the torch.nn.GRU object, with the layers applied
    :ivar fc: torch.nn.Linear activation-function
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, output_dim: int):
        """
        Initialize the GRU and Linear.

        :param input_dim: size of input-tensor
        :param hidden_dim: wanted size of hidden-matrix
        :param num_layers: number of wanted gru-layers.
        :param output_dim: wanted size of output.
        """
        super(GRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: 'torch.tensor'):
        """
        Apply one step of all the layers and return output.

        :param x: input-data tensor
        :return: predicted output.
        """
        x = x.double()
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().double()
        out, (hn) = self.gru(x, (h0.detach()))
        out = self.fc(out[:, -1, :])
        return out

    def accuracy(self, x: 'torch.tensor', y: 'np.array'):
        """
        Predict y and compute R2 score based on real y.

        :param x: input-tensor
        :param y: expected output as np.array
        :return: None
        """
        return r2_score(self(x).detach().numpy(), y)


class CNN(nn.Module):
    """
    Two layered convolutional network that scales in accordance to specified input-features.

    :ivar lookback: the size of the lookback, i.e length of each x.
    :ivar conv: First convolutional layer, splits features into 64 sets and has a kernel-size of 5.
    :ivar pool: First max-pool, kernel-size of 1
    :ivar conv2: Second convolutional layer, 64 in-channels, 128 out with kernel size 5.
    :ivar pool2: Second max-pool with kernel-size 1.
    :ivar dense: fist dense layer
    :ivar dense2: Dense-layer for determining output.
    :ivar criterion: Torch MSELoss object.
    """

    def __init__(self, features: int, points_in_lookback: int):
        """
        Initialize all layers and loss criterion.

        :param features: The number of features, will be set as number of in-channels in first layer.
        :param points_in_lookback: the length of each x-point.
        """
        super(CNN, self).__init__()
        self.lookback = points_in_lookback
        self.conv = nn.Conv2d(features, 64, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=1)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=5, padding=2)
        self.pool3 = nn.MaxPool2d(kernel_size=1)

        self.dense = nn.Linear(256 * self.lookback, 64)
        self.dense2 = nn.Linear(64, 1)
        self.criterion = nn.MSELoss(reduction='mean')

    def logits(self, x: 'torch.tensor') -> 'torch.tensor':
        """
        Apply the convolution-layers on a given tensor

        :param x: the tensor to apply the convolution
        :return: x after convolution.
        """
        x = self.conv(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)

        # print("Pre dense", x.shape)
        # x = self.dense(x.reshape(-1, 128 * self.lookback))
        x = self.dense(x.reshape(-1, 256 * self.lookback))
        # print("First dense", x.shape)
        return self.dense2(x.reshape(-1, 64))

    def f(self, x: 'torch.tensor') -> 'torch.tensor':
        """
        Apply convolution on x and return a predicted values.

        :param x: input-data.
        :return: x-tensor with applied logits
        """
        return self.logits(x)

    def loss(self, x: 'torch.tensor', y: 'torch.tensor') -> 'torch.nn.MSELoss':
        """
        Compute the loss and return the criterion for back-propagation during training.

        :param x: input-data
        :param y: expected output.
        :return: torch.nn.MSELoss object.
        """
        return self.criterion(self.logits(x), y)

    def accuracy(self, y_pred: 'np.array', y: 'np.array') -> float:
        """
        Compute the R2 score for predicted and expected y-values.

        :param y_pred: predicted y-values
        :param y: expected y-values
        :return: the computed R2-score
        """
        return r2_score(y_pred, y)


class RandomForest:

    """
    Wrapper class for sklearn.ensemble.RandomForestRegressor to store state of "trained" classifier.

    :ivar classifier: The RF-regressor initialized with specified number of trees.
    :ivar model: The fitted RF-regressor. Is none before train is called.
    """

    def __init__(self, random_state: int):
        """
        Initialize the classifier.

        :param random_state: Number of trees in your random-forest.
        """
        self.classifier = RandomForestRegressor(n_estimators=random_state)
        self.model = None

    def train(self, x: 'np.array', y: 'np.array'):
        """
        Fit the classifier

        :param x: input-data
        :param y: expected output
        :return: None
        """
        y = y.ravel()
        self.model = self.classifier.fit(x, y)

    def accuracy(self, x: 'np.array', y: 'np.array') -> float:
        """
        Compute and return the accuracy of the model

        :param x: input-data
        :param y: expected output
        :return: the R2-accuracy of the model
        """
        y = y.reshape(-1, 1)
        return self.model.score(x, y)

    def predict(self, x: 'np.array') -> 'np.array':
        """
        Return the predicted data on given input-data.

        :param x: input-data
        :return: predicted values
        """
        return self.model.predict(x).reshape(-1, 1)

    def get_model(self) -> 'RandomForestRegressor':
        return self.model

