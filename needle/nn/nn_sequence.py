"""The module.
"""
from typing import List
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module
import math

class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        denom = ops.add_scalar(ops.exp(-x), 1.0)
        one = ops.mul_scalar(denom, 0)   
        one = ops.add_scalar(one, 1)   
        return ops.divide(one, denom)      

class RNNCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.use_bias = bool(bias)
        assert nonlinearity in ("tanh", "relu")
        self.nonlinearity = nonlinearity

        bound = 1.0 / math.sqrt(self.hidden_size)

        self.W_ih = Parameter(
            init.rand(
                self.input_size,
                self.hidden_size,
                low=-bound,
                high=bound,
                device=device,
                dtype=dtype,
            )
        )
        self.W_hh = Parameter(
            init.rand(
                self.hidden_size,
                self.hidden_size,
                low=-bound,
                high=bound,
                device=device,
                dtype=dtype,
            )
        )

        if self.use_bias:
            self.bias_ih = Parameter(
                init.rand(self.hidden_size, low=-bound, high=bound, device=device, dtype=dtype)
            )
            self.bias_hh = Parameter(
                init.rand(self.hidden_size, low=-bound, high=bound, device=device, dtype=dtype)
            )
        else:
            self.bias_ih = None
            self.bias_hh = None

    def forward(self, X, h=None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor contianing the next hidden state
            for each element in the batch.
        """
        bs = X.shape[0]
        if h is None:
            h = init.zeros(bs, self.hidden_size, device=X.device, dtype=X.dtype)

        y = X @ self.W_ih + h @ self.W_hh
        if self.use_bias:
            y = y + ops.reshape(self.bias_ih, (1, self.hidden_size)).broadcast_to(y.shape)
            y = y + ops.reshape(self.bias_hh, (1, self.hidden_size)).broadcast_to(y.shape)
        act = ops.tanh if self.nonlinearity == "tanh" else ops.relu
        return act(y)


class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies a multi-layer RNN with tanh or ReLU non-linearity to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        nonlinearity - The non-linearity to use. Can be either 'tanh' or 'relu'.
        bias - If False, then the layer does not use bias weights.

        Variables:
        rnn_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, hidden_size) for k=0. Otherwise the shape is
            (hidden_size, hidden_size).
        rnn_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, hidden_size).
        rnn_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (hidden_size,).
        rnn_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (hidden_size,).
        """
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)

        self.rnn_cells = []
        in_sz = self.input_size
        for layer in range(self.num_layers):
            cell = RNNCell(
                in_sz,
                self.hidden_size,
                bias=bias,
                nonlinearity=nonlinearity,
                device=device,
                dtype=dtype,
            )
            setattr(self, f"rnn_cell_{layer}", cell)
            self.rnn_cells.append(cell)
            in_sz = self.hidden_size

    def forward(self, X, h0=None):
        """
        Inputs:
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial
            hidden state for each element in the batch. Defaults to zeros if not provided.

        Outputs
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the RNN, for each t.
        h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
        """
        seq_len = X.shape[0]
        bs = X.shape[1]

        if h0 is None:
            h_list = [
                init.zeros(bs, self.hidden_size, device=X.device, dtype=X.dtype)
                for _ in range(self.num_layers)
            ]
        else:
            h_splits = ops.split(h0, axis=0)
            h_list = list(h_splits)

        outputs = []

        x_ts = ops.split(X, axis=0) 
        for x_t in x_ts:
            h_prev_out = x_t
            new_h_list = []
            for layer, cell in enumerate(self.rnn_cells):
                h_next = cell(h_prev_out, h_list[layer])
                new_h_list.append(h_next)
                h_prev_out = h_next 
            h_list = new_h_list
            outputs.append(h_prev_out)

        output = ops.stack(outputs, axis=0) 
        h_n = ops.stack(h_list, axis=0) 
        return output, h_n


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype="float32"):
        """
        A long short-term memory (LSTM) cell.

        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights

        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (4*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (4*hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        self.input_size  = int(input_size)
        self.hidden_size = int(hidden_size)
        self.use_bias    = bool(bias)

        bound = 1.0 / math.sqrt(self.hidden_size)
        fourH = 4 * self.hidden_size

        self.W_ih = Parameter(
            init.rand(self.input_size, fourH, low=-bound, high=bound, device=device, dtype=dtype)
        )
        self.W_hh = Parameter(
            init.rand(self.hidden_size, fourH, low=-bound, high=bound, device=device, dtype=dtype)
        )

        if self.use_bias:
            self.bias_ih = Parameter(
                init.rand(fourH, low=-bound, high=bound, device=device, dtype=dtype)
            )
            self.bias_hh = Parameter(
                init.rand(fourH, low=-bound, high=bound, device=device, dtype=dtype)
            )
        else:
            self.bias_ih = None
            self.bias_hh = None

        self._sigmoid = Sigmoid()


    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (batch, input_size): Tensor containing input features
        h, tuple of (h0, c0), with
            h0 of shape (bs, hidden_size): Tensor containing the initial hidden state
                for each element in the batch. Defaults to zero if not provided.
            c0 of shape (bs, hidden_size): Tensor containing the initial cell state
                for each element in the batch. Defaults to zero if not provided.

        Outputs: (h', c')
        h' of shape (bs, hidden_size): Tensor containing the next hidden state for each
            element in the batch.
        c' of shape (bs, hidden_size): Tensor containing the next cell state for each
            element in the batch.
        """
        bs = X.shape[0]
        H  = self.hidden_size

        if h is None:
            h_t = init.zeros(bs, H, device=X.device, dtype=X.dtype)
            c_t = init.zeros(bs, H, device=X.device, dtype=X.dtype)
        else:
            h_t, c_t = h

        gates = X @ self.W_ih + h_t @ self.W_hh
        if self.use_bias:
            gates = gates + ops.reshape(self.bias_ih, (1, 4 * H)).broadcast_to(gates.shape)
            gates = gates + ops.reshape(self.bias_hh, (1, 4 * H)).broadcast_to(gates.shape)

        gates4 = ops.reshape(gates, (bs, 4, H))   
        i, f, g, o = ops.split(gates4, axis=1)   

        i = self._sigmoid(i)
        f = self._sigmoid(f)
        g = ops.tanh(g)
        o = self._sigmoid(o)

        c_new = f * c_t + i * g
        h_new = o * ops.tanh(c_new)
        return h_new, c_new


class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.

        Variables:
        lstm_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 4*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 4*hidden_size).
        lstm_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 4*hidden_size).
        lstm_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        lstm_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        """
        self.input_size  = int(input_size)
        self.hidden_size = int(hidden_size)
        self.num_layers  = int(num_layers)

        self.lstm_cells: List[LSTMCell] = []
        in_sz = self.input_size
        for l in range(self.num_layers):
            cell = LSTMCell(in_sz, self.hidden_size, bias=bias, device=device, dtype=dtype)
            setattr(self, f"lstm_cell_{l}", cell)
            self.lstm_cells.append(cell)
            in_sz = self.hidden_size

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h, tuple of (h0, c0) with
            h_0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden state for each element in the batch. Defaults to zeros if not provided.
            c0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden cell state for each element in the batch. Defaults to zeros if not provided.

        Outputs: (output, (h_n, c_n))
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the LSTM, for each t.
        tuple of (h_n, c_n) with
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden cell state for each element in the batch.
        """
        S, bs, _ = X.shape
        H = self.hidden_size
        L = self.num_layers

        if h is None:
            h_list = [init.zeros(bs, H, device=X.device, dtype=X.dtype) for _ in range(L)]
            c_list = [init.zeros(bs, H, device=X.device, dtype=X.dtype) for _ in range(L)]
        else:
            h0, c0 = h
            h_list = list(ops.split(h0, axis=0))
            c_list = list(ops.split(c0, axis=0))

        outputs = []
        x_ts = ops.split(X, axis=0)
        for x_t in x_ts:
            layer_in = x_t
            new_h_list = []
            new_c_list = []
            for l, cell in enumerate(self.lstm_cells):
                h_next, c_next = cell(layer_in, (h_list[l], c_list[l]))
                new_h_list.append(h_next)
                new_c_list.append(c_next)
                layer_in = h_next
            h_list, c_list = new_h_list, new_c_list
            outputs.append(layer_in)
        output = ops.stack(outputs, axis=0) 
        h_n    = ops.stack(h_list, axis=0)
        c_n    = ops.stack(c_list, axis=0)
        return output, (h_n, c_n)

class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector

        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = Parameter(
            init.randn(
                num_embeddings,
                embedding_dim,
                mean=0.0,
                std=1.0,
                device=device,
                dtype=dtype,
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors

        Input:
        x of shape (seq_len, bs)

        Output:
        output of shape (seq_len, bs, embedding_dim)
        """
        seq_len, bs = x.shape
        num_classes = self.num_embeddings

        # Split sequence into list of (1, bs) tensors
        x_slices = ops.split(x, axis=0)

        embeddings_per_timestep = []
        for x_t in x_slices:
            # x_t has shape (1, bs); remove the leading singleton with reshape
            x_t_flat = ops.reshape(x_t, (bs,))
            # one-hot encode tokens for this timestep
            one_hot_t = init.one_hot(
                num_classes, x_t_flat, device=x.device, dtype=x.dtype
            )  # shape: (bs, num_classes)
            # project to embedding space
            emb_t = one_hot_t @ self.weight  # (bs, embed_dim)
            embeddings_per_timestep.append(emb_t)

        # Stack embeddings over time -> (seq_len, bs, embedding_dim)
        return ops.stack(embeddings_per_timestep, axis=0)