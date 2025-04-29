import numpy as np
from nunet.layer import Linear, Sequential
from nunet.layer.activation import ReLU, Sigmoid


class MLP(Sequential):
    def __init__(self, input_dim, hidden_dims, output_dim, init_method='normal', activation='relu', verbose=True):
        self.verbose = verbose
        modules = []
        dims = [input_dim] + hidden_dims + [output_dim]
        for i in range(len(dims) - 1):
            modules.append(Linear(dims[i], dims[i + 1], init_method=init_method, bias=True))
            if i < len(dims) - 2:
                if activation == 'relu':
                    modules.append(ReLU())
                elif activation == 'sigmoid':
                    modules.append(Sigmoid())

                else:
                    raise ValueError("Invalid activation function")
        super().__init__(*modules)

    def load_param(self, params):
        for i, layer in enumerate(self.layers):
            layer.load_param(*params[i])

    def train(self, X, y, loss_fn, optimizer_fn, batch_size=32):
        epoch_loss = 0
        iters = range(0, X.shape[0], batch_size)
        n_iters = len(iters)
        for i in iters:
            X_batch = X[i:i + batch_size]
            y_batch = y[i:i + batch_size]
            optimizer_fn.zero_grad()
            y_pred = self(X_batch)
            loss_value = loss_fn(y_pred, y_batch)
            epoch_loss += loss_value / n_iters
            gradwrtpred = loss_fn.backward()
            super().backward(gradwrtoutput=gradwrtpred)
            optimizer_fn.step()
        return epoch_loss

    def validate(self, X, y, loss, batch_size=32):
        epoch_loss = 0
        iters = range(0, X.shape[0], batch_size)
        n_iters = len(iters)
        for i in iters:
            X_batch = X[i:i + batch_size]
            y_batch = y[i:i + batch_size]
            y_pred = self(X_batch)
            loss_value = loss(y_pred, y_batch)
            epoch_loss += loss_value / n_iters
        return epoch_loss

    def fit(self, X_train, y_train, loss_fn, optimizer_fn, X_val=None, y_val=None, batch_size=32, epochs=10, patience=None):
        train_loss = []
        val_loss = []
        best_val_loss = np.inf
        best_param = self.param
        best_epoch = 0

        for epoch in range(epochs):
            idx = np.random.permutation(X_train.shape[0])
            X_train = X_train[idx]
            y_train = y_train[idx]

            epoch_loss = self.train(X_train, y_train, loss_fn, optimizer_fn, batch_size)
            train_loss.append(epoch_loss)

            if (X_val is not None) and (y_val is not None):
                epoch_val_loss = self.validate(X_val, y_val, loss_fn, batch_size)
                val_loss.append(epoch_val_loss)

                if epoch_val_loss < best_val_loss:
                    best_val_loss = epoch_val_loss
                    best_param = self.param
                    best_epoch = epoch
                elif (patience is not None) and ((epoch - best_epoch) >= patience):
                    if self.verbose:
                        print("Early stopping")
                    break

                if self.verbose:
                    print(f"Epoch [{epoch + 1}/{epochs}] - train_loss: {train_loss[-1]:.4f} - val_loss: {val_loss[-1]:.4f}")
            else:
                if self.verbose:
                    print(f"Epoch [{epoch + 1}/{epochs}] - train_loss: {train_loss[-1]:.4f}")

        if best_param is not None:
            self.load_param(best_param)
        return train_loss, val_loss

    def predict(self, X):
        return self(X)