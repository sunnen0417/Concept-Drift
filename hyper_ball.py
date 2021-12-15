import numpy as np
import torch.utils.data as Data
import torch
import torch.nn as nn

class HyperBall_Classifier(nn.Module):
    def __init__(self, in_size=2, out_size=2):
        super(HyperBall_Classifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_size, 16 * in_size),
            nn.ReLU(),
            nn.Linear(16 * in_size, out_size)
        )

    def forward(self, x):
        x = self.fc(x)
        return x

class HyperBallData(Data.Dataset):
    """
    Args:
        ranges:
            * range of each dimension
            * size -> 2 (low and high) * dimension
        n_sample:
            * number of sample
        concept:
            * [r, c, K]
            * r -> raidus of each dimension
                * size: time * dimension
            * c -> center of each dimension
                * size: time * dimension
            * K -> value for easier adjustment
                * size: time
        noise:
            * probability that label is flipped
              while generating data
    """
    """
    decision boundary of hyper ball:
        * sum((x-c)**2 /r**2) = K
    """

    def __init__(self, ranges, n_sample, concept, noise):
        super(HyperBallData, self).__init__()
        self.dim = len(ranges[0])
        self.ranges = ranges
        self.n_sample = n_sample
        self.radius = concept[0]
        self.center = concept[1]
        self.K = concept[2]
        self.noise = noise
        self.data = []
        self.target = []

    def __getitem__(self, index):
        return torch.FloatTensor(self.data[index]), self.target[index]

    def __len__(self):
        return len(self.target)

    def set_t(self, t):
        self.t = t
        self.data = np.random.uniform(self.ranges[0], self.ranges[1],
                                 size=(self.n_sample, self.dim))
        radius = np.array(self.radius[t])
        center = np.array(self.center[t])
        K = self.K[t]
        y = np.sum((self.data - center)**2 / radius**2, axis = 1) - K
        y = y * np.random.uniform(-self.noise, 1 - self.noise, self.n_sample)
        y = y >= 0
        y = np.array(y, dtype='int64')
        self.target = y

def get_hyperball_concept_translate(r_range = None, c_range = None, K_range = None, t = 40):
    """
    Args:
        r_range: the range of each dimension of radius
            * size -> 2 * dimension
        c_range: the range of each dimension of center
            * size -> 2 * dimension
        K_range:
            * size -> dimension
        t:
            number of time
    """

    if r_range == None:
        # default generate
        assert c_range == None
        assert K_range == None
        radius = list(np.linspace([2, 1], [1, 2], t))
        center = list(np.linspace([0, -15], [0, 15], t))
        K = [25] * t
    else:
        radius = list(np.linspace(r_range[0], r_range[1], t))
        center = list(np.linspace(c_range[0], c_range[1], t))
        K = list(np.linspace(K_range[0], K_range[1], t))

    return [radius, center, K]

def get_hyperball_concept(radius = None, center = None, K = None):

    if radius == None:
        assert center == None
        assert K == None
        return get_hyperball_concept_translate()

    return [raidus, center, K]
