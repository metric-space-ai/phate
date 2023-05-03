# oiginal author: Daniel Burkhardt <daniel.burkhardt@yale.edu>
# (C) 2017 Krishnaswamy Lab GPLv2

import numpy as np
import graphtools
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from scipy import sparse

from . import utils, vne, mds

class PHATE(BaseEstimator):
    """
    *PHATE: A Dimensionality Reduction Method for Visualizing Trajectory
    Structures in High-Dimensional Biological Data*,
    Moon KR, van Dijk D, Zheng W, *et al.* (2017),
    `BioRxiv <http://biorxiv.org/content/early/2017/03/24/120378>`_.
    """

    def __init__(
        self,
        n_components=2,
        knn=5,
        decay=40,
        n_landmark=2000,
        t="auto",
        gamma=1,
        n_pca=100,
        mds_solver="sgd",
        knn_dist="euclidean",
        knn_max=None,
        mds_dist="euclidean",
        mds="metric",
        n_jobs=1,
        random_state=None,
        verbose=1,
        **kwargs
    ):
        self.n_components = n_components
        self.decay = decay
        self.knn = knn
        self.t = t
        self.n_landmark = n_landmark
        self.mds = mds
        self.n_pca = n_pca
        self.knn_dist = knn_dist
        self.knn_max = knn_max
        self.mds_dist = mds_dist
        self.mds_solver = mds_solver
        self.random_state = random_state
        self.kwargs = kwargs

        self.graph = None
        self._diff_potential = None
        self.embedding = None
        self.X = None
        self.optimal_t = None

        self.n_jobs = n_jobs

        if gamma > 0.99 and gamma < 1:
            gamma = 0.99
        self.gamma = gamma

        if verbose is True:
            verbose = 1
        elif verbose is False:
            verbose = 0
        self.verbose = verbose
        self._check_params()

    @property
    def diff_op(self):
        if self.graph is not None:
            if isinstance(self.graph, graphtools.graphs.LandmarkGraph):
                diff_op = self.graph.landmark_op
            else:
                diff_op = self.graph.diff_op
            if sparse.issparse(diff_op):
                diff_op = diff_op.toarray()
            return diff_op
        else:
            raise NotFittedError(
                "This PHATE instance is not fitted yet. Call "
                "'fit' with appropriate arguments before "
                "using this method."
            )

    @property
    def diff_potential(self):
        diff_potential = self._calculate_potential()
        if isinstance(self.graph, graphtools.graphs.LandmarkGraph):
            diff_potential = self.graph.interpolate(diff_potential)
        return diff_potential

    def _check_params(self):
        utils.check_positive(n_components=self.n_components, knn=self.knn)
        utils.check_int(
            n_components=self.n_components, knn=self.knn, n_jobs=self.n_jobs
        )
        utils.check_between(-1, 1, gamma=self.gamma)
        utils.check_if_not(None, utils.check_positive, decay=self.decay)
        utils.check_if_not(
            None,
            utils.check_positive,
            utils.check_int,
            n_landmark=self.n_landmark,
            n_pca=self.n_pca,
            knn_max=self.knn_max,
        )
        utils.check_if_not("auto", utils.check_positive, utils.check_int, t=self.t)
        if not callable(self.knn_dist):
            utils.check_in(
                [
                    "euclidean",
                    "precomputed",
                    "cosine",
                    "correlation",
                    "cityblock",
                    "l1",
                    "l2",
                    "manhattan",
                    "braycurtis",
                    "canberra",
                    "chebyshev",
                    "dice",
                    "hamming",
                    "jaccard",
                    "kulsinski",
                    "mahalanobis",
                    "matching",
                    "minkowski",
                    "rogerstanimoto",
                    "russellrao",
                    "seuclidean",
                    "sokalmichener",
                    "sokalsneath",
                    "sqeuclidean",
                    "yule",
                    "precomputed_affinity",
                    "precomputed_distance",
                ],
                knn_dist=self.knn_dist,
            )
        if not callable(self.mds_dist):
            utils.check_in(
                [
                    "euclidean",
                    "cosine",
                    "correlation",
                    "braycurtis",
                    "canberra",
                    "chebyshev",
                    "cityblock",
                    "dice",
                    "hamming",
                    "jaccard",
                    "kulsinski",
                    "mahalanobis",
                    "matching",
                    "minkowski",
                    "rogerstanimoto",
                    "russellrao",
                    "seuclidean",
                    "sokalmichener",
                    "sokalsneath",
                    "sqeuclidean",
                    "yule",
                ],
                mds_dist=self.mds_dist,
            )
        utils.check_in(["classic", "metric", "nonmetric"], mds=self.mds)
        utils.check_in(["sgd", "smacof"], mds_solver=self.mds_solver)

    def _set_graph_params(self, **params):
        try:
            self.graph.set_params(**params)
        except AttributeError:
            pass

    def _reset_graph(self):
        self.graph = None
        self._reset_potential()

    def _reset_potential(self):
        self._diff_potential = None
        self._reset_embedding()

    def _reset_embedding(self):
        self.embedding = None

    def set_params(self, **params):
        reset_kernel = False
        reset_potential = False
        reset_embedding = False

        # mds parameters
        if "n_components" in params and params["n_components"] != self.n_components:
            self.n_components = params["n_components"]
            reset_embedding = True
            del params["n_components"]
        if "mds" in params and params["mds"] != self.mds:
            self.mds = params["mds"]
            reset_embedding = True
            del params["mds"]
        if "mds_solver" in params and params["mds_solver"] != self.mds_solver:
            self.mds_solver = params["mds_solver"]
            reset_embedding = True
            del params["mds_solver"]
        if "mds_dist" in params and params["mds_dist"] != self.mds_dist:
            self.mds_dist = params["mds_dist"]
            reset_embedding = True
            del params["mds_dist"]

        # diff potential parameters
        if "t" in params and params["t"] != self.t:
            self.t = params["t"]
            reset_potential = True
            del params["t"]
        if "gamma" in params and params["gamma"] != self.gamma:
            self.gamma = params["gamma"]
            reset_potential = True
            del params["gamma"]

        # kernel parameters
        if "knn" in params and params["knn"] != self.knn:
            self.knn = params["knn"]
            reset_kernel = True
            del params["knn"]
        if "knn_max" in params and params["knn_max"] != self.knn_max:
            self.knn_max = params["knn_max"]
            reset_kernel = True
            del params["knn_max"]
        if "decay" in params and params["decay"] != self.decay:
            self.decay = params["decay"];
            reset_kernel = True
            del params["decay"]
        if "n_pca" in params:
            if self.X is not None and params["n_pca"] >= np.min(self.X.shape):
                params["n_pca"] = None
            if params["n_pca"] != self.n_pca:
                self.n_pca = params["n_pca"]
                reset_kernel = True
                del params["n_pca"]
        if "knn_dist" in params and params["knn_dist"] != self.knn_dist:
            self.knn_dist = params["knn_dist"]
            reset_kernel = True
            del params["knn_dist"]
        if "n_landmark" in params and params["n_landmark"] != self.n_landmark:
            if self.n_landmark is None or params["n_landmark"] is None:
                self._reset_graph()
            else:
                self._set_graph_params(n_landmark=params["n_landmark"])
            self.n_landmark = params["n_landmark"]
            del params["n_landmark"]

        if reset_kernel:
            self._reset_graph()
        if reset_potential:
            self._reset_potential()
        if reset_embedding:
            self._reset_embedding()

        self._set_graph_params(**params)

        self._check_params()
        return self

    def _parse_input(self, X):
        if isinstance(X, graphtools.graphs.LandmarkGraph) or (
            isinstance(X, graphtools.base.BaseGraph) and self.n_landmark is None
        ):
            self.graph = X
            X = X.data
            n_pca = self.graph.n_pca
            update_graph = False
            return X, n_pca, update_graph
        elif isinstance(X, graphtools.base.BaseGraph):
            self.graph = None
            X = X.kernel
            precomputed = "affinity"
            n_pca = None
            update_graph = False
            return X, n_pca, precomputed, update_graph
        else:
            update_graph = True
            if not callable(self.knn_dist) and self.knn_dist.startswith("precomputed"):
                if self.knn_dist == "precomputed":
                    if isinstance(X, sparse.coo_matrix):
                        X = X.tocsr()
                    if X[0, 0] == 0:
                        precomputed = "distance"
                    else:
                        precomputed = "affinity"
                elif self.knn_dist in ["precomputed_affinity", "precomputed_distance"]:
                    precomputed = self.knn_dist.split("_")[1]
                else:
                    raise ValueError(
                        "knn_dist {} not recognized. Did you mean "
                        "'precomputed_distance', "
                        "'precomputed_affinity', or 'precomputed' "
                        "(automatically detects distance or affinity)?"
                    )
                n_pca = None
            else:
                precomputed = None
                if self.n_pca is None or self.n_pca >= np.min(X.shape):
                    n_pca = None
                else:
                    n_pca = self.n_pca
            return X, n_pca, precomputed, update_graph

    def _update_graph(self, X, precomputed, n_pca, n_landmark):
        if self.X is not None and not utils.matrix_is_equivalent(X, self.X):
            self._reset_graph()
        else:
            try:
                self.graph.set_params(
                    decay=self.decay,
                    knn=self.knn,
                    knn_max=self.knn_max,
                    distance=self.knn_dist,
                    precomputed=precomputed,
                    n_jobs=self.n_jobs,
                    verbose=self.verbose,
                    n_pca=n_pca,
                    n_landmark=n_landmark,
                    random_state=self.random_state,
                )
            except ValueError as e:
                self._reset_graph()

    def fit(self, X):
        X, n_pca, precomputed, update_graph = self._parse_input(X)

        if self.n_landmark is None or X.shape[0] <= self.n_landmark:
            n_landmark = None
        else:
            n_landmark = self.n_landmark

        if self.graph is not None and update_graph:
            self._update_graph(X, precomputed, n_pca, n_landmark)

        self.X = X

        if self.graph is None:
            self.graph = graphtools.Graph(
                X,
                n_pca=n_pca,
                n_landmark=n_landmark,
                distance=self.knn_dist,
                precomputed=precomputed,
                knn=self.knn,
                knn_max=self.knn_max,
                decay=self.decay,
                thresh=1e-4,
                n_jobs=self.n_jobs,
                verbose=self.verbose,
                random_state=self.random_state,
                **(self.kwargs)
            )
        self.diff_op
        return self

    def transform(self, X=None, t_max=100):
        if self.graph is None:
            raise NotFittedError(
                "This PHATE instance is not fitted yet. Call "
                "'fit' with appropriate arguments before "
                "using this method."
            )
        elif X is not None and not utils.matrix_is_equivalent(X, self.X):
            if self.embedding is None:
                self.transform()
            transitions = self.graph.extend_to_data(X)
            return self.graph.interpolate(self.embedding, transitions)
        else:
            diff_potential = self._calculate_potential(t_max=t_max)
            if self.embedding is None:
                self.embedding = mds.embed_MDS(
                    diff_potential,
                    ndim=self.n_components,
                    how=self.mds,
                    solver=self.mds_solver,
                    distance_metric=self.mds_dist,
                    n_jobs=self.n_jobs,
                    seed=self.random_state,
                    verbose=max(self.verbose - 1, 0),
                )
            if isinstance(self.graph, graphtools.graphs.LandmarkGraph):
                return self.graph.interpolate(self.embedding)
            else:
                return self.embedding

    def fit_transform(self, X, **kwargs):
        self.fit(X)
        embedding = self.transform(**kwargs)
        return embedding

    def _calculate_potential(self, t=None, t_max=100):
        if t is None:
            t = self.t
        if self._diff_potential is None:
            if t == "auto":
                t = self._find_optimal_t(t_max=t_max)
            else:
                t = self.t
            diff_op_t = np.linalg.matrix_power(self.diff_op, t)
            if self.gamma == 1:
                diff_op_t = diff_op_t + 1e-7
                self._diff_potential = -1 * np.log(diff_op_t)
            elif self.gamma == -1:
                self._diff_potential = diff_op_t
            else:
                c = (1 - self.gamma) / 2
                self._diff_potential = ((diff_op_t) ** c) / c
        else:
            self._find_optimal_t(t_max=t_max)

        return self._diff_potential

    def _von_neumann_entropy(self, t_max=100):
        t = np.arange(t_max)
        return t, vne.compute_von_neumann_entropy(self.diff_op, t_max=t_max)

    def _find_optimal_t(self, t_max=100):
        t, h = self._von_neumann_entropy(t_max=t_max)
        t_opt = vne.find_knee_point(y=h, x=t)

        self.optimal_t = t_opt

        return t_opt