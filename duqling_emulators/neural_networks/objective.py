"""Objective function for elementary NAS"""

from mlp import MLP, train_mlp
from skopt.space import Integer, Real, Categorical

search_space = [
    Integer(1, 5, name='depth'),
    Integer(16, 264, name='width'),
    Real(1e-4, 1e-2, prior='log-uniform', name='lr'),
    Categorical(['ReLU', 'ELU', 'SiLU'], name='activation')
]

def objective(params, X_train, y_train, X_val, y_val):
    """Objective function for Bayesian optimization."""

    config = {
        'depth':      params[0],
        'width':      params[1], 
        'lr':         params[2],
        'activation': params[3]
    }

    model = MLP(input_dim=X_train.shape[1],
                hidden_dims=[config['width']] * config['depth'],
                activation=config['activation'])

    return train_mlp(model,
                     X_train, y_train,
                     X_val, y_val,
                     config['lr'])
