<h2 style="text-align: center;"><ins>MLP Design</ins></h2>

This fully connected network is simply a sequence of hidden layers seperated by a consistent activation function.

The model is is instantiated with
1. an input dimension,
2. a list of integers corresponding to the width of each layer, and
3. the name of a specific activation function to use after every hidden layer.

### Training

The network's performance is evaluated using MSE loss and tuned using the Adam optimizer.

The training pipeline uses an early stopping heuristic to escape unnecessary computation when the model has converged on a consistent solution. This early stopping works as follows:

- Every 10 epochs, we evaluate the model's validation loss.
- If the validation loss hasn't improved since the last evaluation, a counter is incremented by 10.
- If this counter exceeds 50 (i.e., the model's loss hasn't improved for 50 epochs), the training process is terminated.

### Architecture

A note on the layer shapes available to Bayesian optimization: despite being able to adjust the dimension of every hidden layer, the initial implementation chooses a single width for all layers. Changing individual layer widths could, in theory, force the model to learn greater abstract representations of the data. However, optimizing this kind of heirarchical thinking is very hard - both in theory and in practice.

Given all of this, the model is currently constructed as follows:

$$
\begin{aligned}
    \underset{n \times d}{H}^{[0]} &= X \in \mathbb{R}^{n \times d} \\
    \underset{n \times w}{H}^{[1]} &= \sigma\left(\underset{n \times d}{H}^{[0]} \hspace{3mm} \left(\underset{w \times d}{W}^{[1]}\right)^T + \underset{1 \times w}{\hat{b}}^{[1]}\right) \\
    \underset{n \times w}{H}^{[l]} &= \sigma\left(\underset{n \times w}{H}^{[l-1]} \left(\underset{w \times w}{W}^{[l]}\right)^T + \underset{1 \times w}{\hat{b}}^{[l]}\right) \quad \mid \quad l = 2, \ldots, L \\
    \underset{n \times 1}{y} &= \underset{n \times w}{H}^{[L]} \left(\underset{1 \times w}{W}^{[L+1]}\right)^T + \underset{1 \times 1}{\hat{b}}^{[L+1]}
\end{aligned}
$$

<div align="center">

| Variable | Definition | Search Space |
|:---:|---|:---:|
| $X$ | batched input |  |
| $y$ | batched output |  |
| $H^{[l]}$ | layer $l$ output | |
| $W^{[l]}$ | layer $l$ weights | |
| $\hat{b}^{[l]}$ | layer $l$ bias | |
| $n$ | input batch size |  |
| $d$ | input dim |  |
| $w$ | network width | {16...256} |
| $L$ | network depth | {1...5} |
| $\sigma(\cdot)$ | activation | {'ReLU', 'SiLU', 'ELU'} |

*Legend*

</div>