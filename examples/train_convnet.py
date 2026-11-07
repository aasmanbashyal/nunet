"""Train on left/right stripe images without downloading a dataset."""

import numpy as np

from nunet.loss import CategoricalCrossEntropyLoss
from nunet.nn import ConvNet
from nunet.optim import SGD


def main():
    np.random.seed(7)
    rng = np.random.default_rng(7)
    labels = np.arange(24) % 2
    images = rng.normal(0, 0.1, (24, 1, 8, 8))
    images[labels == 0, :, :, 1:3] += 1.0
    images[labels == 1, :, :, 5:7] += 1.0
    targets = np.eye(2)[labels]

    model = ConvNet((1, 8, 8), num_classes=2, hidden_channels=4)
    loss_fn = CategoricalCrossEntropyLoss(from_logits=True)
    optimizer = SGD(model.param, lr=0.1)
    initial_loss = loss_fn(model(images), targets)
    for _ in range(40):
        optimizer.zero_grad()
        loss_fn(model(images), targets)
        model.backward(loss_fn.backward())
        optimizer.step()

    logits = model(images)
    final_loss = loss_fn(logits, targets)
    accuracy = np.mean(logits.argmax(axis=1) == labels)
    print(f"Training loss: {initial_loss:.4f} -> {final_loss:.4f}")
    print(f"Training accuracy: {accuracy:.1%}")


if __name__ == "__main__":
    main()
