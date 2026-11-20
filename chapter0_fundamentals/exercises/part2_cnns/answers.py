# %%

import json
import sys
from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path

import einops
import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torchinfo
from IPython.display import display
from jaxtyping import Float, Int
from PIL import Image
from rich import print as rprint
from rich.table import Table
from torch import Tensor
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models
from torchvision.transforms import v2 as transforms
from tqdm.auto import tqdm

# Make sure exercises are in the path
chapter = "chapter0_fundamentals"
section = "part2_cnns"
root_dir = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

MAIN = __name__ == "__main__"

import part2_cnns.tests as tests
import part2_cnns.utils as utils
from fundamentals_utils import CIFAR10
from plotly_utils import line

# On machines with many cores the default of one
# thread per core makes small CPU tensor operations much slower
t.set_num_threads(min(4, t.get_num_threads()))


# %%

class ReLU(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return (x > 0.0) * x

tests.test_relu(ReLU)


# %%

# [0, 1) to [new_min, new_max)
def scale(tensor: Tensor, new_min: float, new_max: float):
    return (tensor * (new_max - new_min) + new_min)

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias=True):
        """
        A simple linear (technically, affine) transformation.

        The fields should be named `weight` and `bias` for compatibility with PyTorch.
        If `bias` is False, set `self.bias` to None.
        """
        super().__init__()
        dx = 1.0 / in_features**0.5
        self.weight = nn.Parameter(
            scale(t.rand((out_features, in_features)), -dx, dx)
        )
        if bias:
            self.bias = nn.Parameter(
                scale(t.rand((out_features,)), -dx, dx)
            )
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        """
        x: shape (*, in_features)
        Return: shape (*, out_features)
        """
        scaled = einops.einsum(
            self.weight,
            x,
            '... out in, ... in -> ... out'
        )
        if self.bias is None:
            return scaled
        else:
            return scaled + self.bias

    def extra_repr(self) -> str:
        out, in_ = self.weight.shape
        has_bias = self.bias is not None
        return f"in_={in_}, out={out}, bias={has_bias}"


tests.test_linear_parameters(Linear, bias=False)
tests.test_linear_parameters(Linear, bias=True)
tests.test_linear_forward(Linear, bias=False)
tests.test_linear_forward(Linear, bias=True)
print(Linear(3, 4))
print(Linear(3, 4, False))


# %%

class Flatten(nn.Module):
    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input: Tensor) -> Tensor:
        """
        Flatten out dimensions from start_dim to end_dim, inclusive of both.
        """
        shape = input.shape

        # Get start & end dims, handling negative indexing for end dim
        start_dim = self.start_dim
        end_dim = self.end_dim if self.end_dim >= 0 else len(shape) + self.end_dim

        # Get the shapes to the left / right of flattened dims, as well as size of flattened middle
        shape_left = shape[:start_dim]
        shape_right = shape[end_dim + 1 :]
        shape_middle = t.prod(t.tensor(shape[start_dim : end_dim + 1])).item()

        return t.reshape(input, shape_left + (shape_middle,) + shape_right)

    def extra_repr(self) -> str:
        return ", ".join([f"{key}={getattr(self, key)}" for key in ["start_dim", "end_dim"]])

# %%

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # Name your two linear layers `self.linear1` and `self.linear2`: the tests copy reference weights in by name
        self.flatten = Flatten() # (28 28) -> 28*28
        self.linear1 = Linear(28*28, 100)
        self.relu = ReLU()
        self.linear2 = Linear(100, 10)

    def forward(self, x1: Tensor) -> Tensor:
        x2 = self.flatten(x1)
        x3 = self.linear1(x2)
        x4 = self.relu(x3)
        x5 = self.linear2(x4)
        return x5


tests.test_mlp_module(SimpleMLP)
tests.test_mlp_forward(SimpleMLP)


# %%

MNIST_TRANSFORM = transforms.Compose(
    [
        transforms.ToImage(),
        transforms.ToDtype(t.float32, scale=True),
        transforms.Normalize([0.1307], [0.3081]),
    ]
)


def get_mnist(trainset_size: int = 10_000, testset_size: int = 1_000) -> tuple[Subset, Subset]:
    """Returns a subset of MNIST training data."""

    # Get original datasets, which are downloaded to "./data" for future use
    mnist_trainset = datasets.MNIST(exercises_dir / "data", train=True, download=True, transform=MNIST_TRANSFORM)
    mnist_testset = datasets.MNIST(exercises_dir / "data", train=False, download=True, transform=MNIST_TRANSFORM)

    # # Return a subset of the original datasets
    mnist_trainset = Subset(mnist_trainset, indices=range(trainset_size))
    mnist_testset = Subset(mnist_testset, indices=range(testset_size))

    return mnist_trainset, mnist_testset


mnist_trainset, mnist_testset = get_mnist()
mnist_trainloader = DataLoader(mnist_trainset, batch_size=64, shuffle=True)
mnist_testloader = DataLoader(mnist_testset, batch_size=64, shuffle=False)

# Get the first batch of test data, by starting to iterate over `mnist_testloader`
for img_batch, label_batch in mnist_testloader:
    print(f"{img_batch.shape=}\n{label_batch.shape=}\n")
    break

# Get the first datapoint in the test set, by starting to iterate over `mnist_testset`
for img, label in mnist_testset:
    print(f"{img.shape=}\n{label=}\n")
    break

t.testing.assert_close(img, img_batch[0])
assert label == label_batch[0].item()

# %%

# Q: Why do we include a data normalization function in torchvision.transforms ?
# A: Since the data is multiplied by the weights, twice-as-large data leads to
#    twice-as-large outputs and twice-as-large gradients, which would lead to a
#    twice-as-large learning rate. Normalizing the data removes that factor from
#    the difficult question of choosing the learning rate. Off-center data, say,
#    centered around 1.0 instead of 0.0 would similarly appear twice-as-large.

# Q: What is the benefit of using shuffle=True when defining our dataloaders?
# A: Avoids any structure from the dataset, e.g. if all the 1's are listed
#    before the 2's.

# Q: What might the problem be if we didn't shuffle?
# A: The weights would learn to always return 1, then to always return 2, etc.,
#    instead of learning features which are universally useful throughout the
#    entire dataset.

# %%

device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")

# If this is CPU, we recommend figuring out how to get cuda access (or MPS if you're on a Mac).
print(device)

# %%

model = SimpleMLP().to(device)

batch_size = 128
epochs = 3

mnist_trainset, _ = get_mnist()
mnist_trainloader = DataLoader(mnist_trainset, batch_size=batch_size, shuffle=True)

optimizer = t.optim.Adam(model.parameters(), lr=1e-3)
loss_list = []

for epoch in range(epochs):
    pbar = tqdm(mnist_trainloader)

    for imgs, labels in pbar:
        # Move data to device, perform forward pass
        imgs, labels = (imgs).to(device), labels.to(device)
        logits = model(imgs)

        # Calculate loss, perform backward pass
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Update logs & progress bar
        loss_list.append(loss.item())
        pbar.set_postfix(epoch=f"{epoch + 1}/{epochs}", loss=f"{loss:.3f}", refresh=False)

line(
    loss_list,
    x_max=epochs * len(mnist_trainset),
    labels={"x": "Examples seen", "y": "Cross entropy loss"},
    title="SimpleMLP training on MNIST",
    width=700,
)

# %%

@dataclass
class SimpleMLPTrainingArgs:
    """
    Defining this class implicitly creates an __init__ method, which sets arguments as below, e.g.
    self.batch_size=64. Any of these fields can also be overridden when you create an instance, e.g.
    SimpleMLPTrainingArgs(batch_size=128).
    """

    batch_size: int = 64
    epochs: int = 3
    learning_rate: float = 1e-3


def train(args: SimpleMLPTrainingArgs) -> tuple[list[float], SimpleMLP]:
    """
    Trains & returns the model, using training parameters from the `args` object. Returns the loss
    list, and the model.
    """
    model = SimpleMLP().to(device)

    mnist_trainset, mnist_testset = get_mnist()
    mnist_trainloader = DataLoader(mnist_trainset, batch_size=args.batch_size, shuffle=True)
    mnist_testloader = DataLoader(mnist_testset, batch_size=args.batch_size, shuffle=False)

    optimizer = t.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_list = []

    for epoch in range(args.epochs):
        # train
        pbar = tqdm(mnist_trainloader)
        for imgs, labels in pbar:
            # Move data to device, perform forward pass
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)

            # Calculate loss, perform backward pass
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Update logs & progress bar
            loss_list.append(loss.item())
            pbar.set_postfix(epoch=f"{epoch + 1}/{args.epochs}", loss=f"{loss:.3f}", refresh=False)

        # calculate validity
        with t.no_grad():
            pbar = tqdm(mnist_testloader)
            correct_count = 0.0
            total_count = 0.0
            for imgs, labels in pbar:
                # Move data to device, perform forward pass
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model(imgs)

                correct_count += (logits.argmax(dim=-1) == labels).sum().item()
                total_count += labels.shape[0]

                # Update progress bar
                pbar.set_postfix(epoch=f"{epoch + 1}/{args.epochs}", correct_count=f"{correct_count}", refresh=False)
        validity = correct_count / total_count * 100
        print(f'validity: {validity:.2f}%')

    return loss_list, model


args = SimpleMLPTrainingArgs()
loss_list, model = train(args)

line(
    loss_list,
    x_max=args.epochs * len(mnist_trainset),
    labels={"x": "Examples seen", "y": "Cross entropy loss"},
    title="SimpleMLP training on MNIST",
    width=700,
)