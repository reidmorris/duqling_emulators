import numpy as np

from sklearn.cross_decomposition      import PLSRegression
from sklearn.linear_model             import LassoLars, ElasticNet
from sklearn.ensemble                 import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.svm                      import SVR
from sklearn.gaussian_process         import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from sklearn.pipeline                 import Pipeline
from sklearn.preprocessing            import StandardScaler

from skopt.space import Real, Integer, Categorical

def get_models(num_features):

    int_sqrt_n_feat = int(np.sqrt(num_features))
    return dict(

        # linear baselines
        pls=dict(
            estimator=PLSRegression(),
            search_spaces={'n_components': Integer(1, max(2, int_sqrt_n_feat))},
            n_iter=int_sqrt_n_feat,
        ),
        lassolars=dict(
            estimator=LassoLars(random_state=42),
            search_spaces={'alpha': Real(1e-6, 1.0, prior='log-uniform')},
            n_iter=20,
        ),
        elasticnet=dict(
            estimator=ElasticNet(max_iter=10_000, random_state=42),
            search_spaces={
                'alpha':    Real(1e-6, 1.0, prior='log-uniform'),
                'l1_ratio': Real(0.0, 1.0),
            },
            n_iter=20,
        ),

        # tree ensembles
        xt=dict(
            estimator=ExtraTreesRegressor(random_state=42),
            search_spaces={
                'n_estimators':     Integer(200, 600),
                'max_features':     Real(0.1, 1.0),
                'min_samples_leaf': Integer(1, 20),
                'bootstrap':        Categorical([True, False]),
            },
            n_iter=20,
        ),
        gbr=dict(
            estimator=GradientBoostingRegressor(random_state=42),
            search_spaces={
                'n_estimators':  Integer(100, 1000),
                'learning_rate': Real(1e-3, 1e-1, prior='log-uniform'),
                'max_depth':     Integer(2, 6),
                'subsample':     Real(0.5, 1.0),
            },
            n_iter=20,
        ),

        # kernel-based models
        svr=dict(
            estimator=SVR(kernel='rbf'),
            search_spaces={
                'C':       Real(1e-2, 1e3, prior='log-uniform'),
                'gamma':   Real(1e-4, min(1.0, 10/num_features), prior='log-uniform'),
                'epsilon': Real(1e-4, 1.0, prior='log-uniform'),
            },
            n_iter=20,
        ),
        gpr=dict(
            estimator = Pipeline([
                ('scale', StandardScaler()),
                ('gpr',   GaussianProcessRegressor(
                    kernel = ConstantKernel(1.0, (1e-2, 1e2))
                           * Matern()
                           + WhiteKernel(1e-3, (1e-6, 1e1)),
                    n_restarts_optimizer = 3,
                    random_state = 42,
                    normalize_y = True
                ))
            ]),
            search_spaces = {
                'gpr__kernel__k1__k2__nu': Categorical([0.5, 1.5, 2.5, 3.5, 4.5]),
            },
            n_iter = 20
        )
    )
