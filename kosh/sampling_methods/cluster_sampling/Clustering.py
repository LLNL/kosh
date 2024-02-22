import numpy as np
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
import pandas as pd
import tqdm
import hdbscan
import sklearn.cluster
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import scipy
from scipy.optimize import brute, basinhopping, minimize
from typing import Union
import copy


class Cluster(object):
    def __init__(self, data, method="DBSCAN",
                 scaling_function="", flatten=False):
        """Cluster object for data reduction

        :param data: A Numpy array or Pandas dataframe of shape
        (n_samples, n_features)
        :type data: array
        :param method: DBSCAN, HDBSCAN, or HAC
        (Hierarchical Agglomerative Clustering)
        :type method: string
        :param scaling_function: function for scaling the features
        :type scaling_function: String or callable
        :param flatten: Flattens data to two dimensions.
        (n_samples, n_features_1*n_features_2* ... *n_features_m)
        :type flatten: bool
        """

        if flatten:
            nsamples = data.shape[0]
            nfeatures = np.prod(data.shape[1:])
            data = data.reshape(nsamples, nfeatures)

        if (data.ndim > 2):
            raise ValueError(
                "data needs to be 2 dimensions. (n_samples, "
                "n_features). Use the 'flatten' option to "
                "convert your data to 2 dimensions.")

        self.data = data

        self.method = method
        self.nFeatures = self.data.shape[1]

        # Output
        self.cluster_labels = None
        self.probabilities = None
        self.loss_estimate = 0

        # Optionally scale the data
        self.scaling_function = scaling_function
        if self.scaling_function:
            self.scaleData()

    def scaleData(self):
        """
        Scale or normalize data, or provide custom scaling function.
        """

        if isinstance(self.scaling_function, type('')):
            if self.scaling_function == "standard":
                self.scaler = StandardScaler()
            elif self.scaling_function == "min_max":
                self.scaler = MinMaxScaler()
            else:
                print("No valid scaling defined")

            self.data = self.scaler.fit_transform(self.data)
        elif callable(self.scaling_function):
            self.data, self.scaler = self.scaling_function(self.data,
                                                           self.nFeatures,
                                                           revert=False,
                                                           scale_vars=[],
                                                           comm=None)

    def revertData(self, r_data):

        if isinstance(self.scaling_function, type('')):
            return self.scaler.inverse_transform(r_data)
        elif callable(self.scaling_function):
            return self.scaling_function(r_data,
                                         self.nFeatures,
                                         revert=True,
                                         scale_vars=self.scaler,
                                         comm=None)

    def makeCluster(self, **kwargs):
        """Clusters data given specified method
        """

        if self.method == "DBSCAN":
            self._makeClusterDBSCAN_(**kwargs)
        elif self.method == "HAC":
            self._makeClusterHAC_(**kwargs)
        elif self.method == "HDBSCAN":
            self._makeClusterHDBSCAN_(**kwargs)
        else:
            print("Error: no valid clustering method given")
            exit()

    def _makeClusterDBSCAN_(self, eps=.05, min_samples=2,
                            n_jobs=1, distance_function="euclidean",
                            Nclusters=-1):
        """Clusters samples with scikit-learn's DBSCAN, and saves data
        and cluster labels in a pandas data frame.

        :param eps: The distance around a sample that defines its
        neighbors.
        :type eps: float
        :param min_samples: The number of samples (or total weight)
        in a neighborhood for a point to be considered a core point.
        This includes the point itself.
        :type min_samples: int
        :param n_jobs: from scikit-learn DBSCAN and Nearest Neighbors:
        The number of parallel jobs to run. -1 means using all processors.
        :type n_jobs: int
        :param distance_function: Method used to compute pairwise distances.
        Must be a valid pairwise distance option from scikit-learn or
        scipy.spatial.distance. Also may be a user defined distance function.
        :type distance_function: string, or callable
        :param Nclusters: How many clusters to find
        :type Nclusters: int
        :return: Pandas dataframe with original data and cluster labels
        as the last column
        :rtype: pandas df
        """

        self.eps = eps
        # Check for custom distance function
        if callable(distance_function):
            dd = distance_function(self.data)
            sq_dist = scipy.spatial.distance.squareform(dd)

            if (Nclusters > 0):
                def dbs_clust_func(distance_threshold):
                    dbs_obj = sklearn.cluster.DBSCAN(
                        eps=distance_threshold,
                        min_samples=min_samples,
                        n_jobs=n_jobs,
                        metric='precomputed').fit(sq_dist)
                    Nclust_1 = dbs_obj.labels_.max()
                    return abs(Nclust_1 - Nclusters)

                # Brute force then basin hopping to find min
                var_scale = self.data.std()**2
                bounds = [(1.e-7 * var_scale, var_scale)]
                distance = brute(dbs_clust_func, bounds, Ns=50)[0]
                eps = minimize(
                    dbs_clust_func,
                    distance,
                    method='Nelder-Mead',
                    bounds=bounds).x[0]

                dbs_clust = sklearn.cluster.DBSCAN(
                    eps=eps,
                    min_samples=min_samples,
                    n_jobs=n_jobs,
                    metric='precomputed').fit(sq_dist)
            else:
                dbs_clust = sklearn.cluster.DBSCAN(
                    eps=eps,
                    min_samples=min_samples,
                    n_jobs=n_jobs,
                    metric='precomputed').fit(sq_dist)
        else:
            if (Nclusters > 0):
                def dbs_clust_func(distance_threshold):
                    if isinstance(distance_threshold, np.ndarray) and distance_threshold.shape == (1,):
                        distance_threshold = float(distance_threshold)
                    dbs_obj = sklearn.cluster.DBSCAN(eps=distance_threshold,
                                                     min_samples=min_samples,
                                                     n_jobs=n_jobs,
                                                     metric=distance_function).fit(self.data)
                    Nclust_1 = dbs_obj.labels_.max()
                    return abs(Nclust_1 - Nclusters)

                # Brute force then basin hopping to find min
                var_scale = self.data.std()**2
                bounds = [(1.e-7 * var_scale, var_scale)]
                distance = brute(dbs_clust_func, bounds, Ns=50)[0]

                # Check if brute force found the solution exactly
                if (dbs_clust_func(distance) == 0.0):
                    eps = distance
                else:
                    eps = minimize(
                        dbs_clust_func,
                        distance,
                        method='Nelder-Mead',
                        bounds=bounds).x[0]

                dbs_clust = sklearn.cluster.DBSCAN(
                    eps=eps,
                    min_samples=min_samples,
                    n_jobs=n_jobs,
                    metric=distance_function).fit(
                    self.data)
            else:
                dbs_clust = sklearn.cluster.DBSCAN(
                    eps=eps,
                    min_samples=min_samples,
                    n_jobs=n_jobs,
                    metric=distance_function).fit(
                    self.data)

        # Save original number of clusters
        self.original_clusters = dbs_clust.labels_.max()

        # Save dbscan object
        self.dbs_object = dbs_clust

        # Save cluster labels
        self.cluster_labels = dbs_clust.labels_

        # Create dataframe with cluster labels
        if isinstance(self.data, pd.DataFrame):
            self.data = self.pd_data
            self.pd_data['clust_labels'] = self.cluster_labels
            self.pd_data.rename(columns=str, inplace=True)
        else:
            self.pd_data = pd.DataFrame(self.data)
            self.pd_data['clust_labels'] = self.cluster_labels
            self.pd_data.rename(columns=str, inplace=True)

        # Give outliers their own cluster number
        max_cluster = self.pd_data['clust_labels'].max()
        outlier_index = self.pd_data[self.pd_data['clust_labels'] == -1].index
        for oi in outlier_index:
            max_cluster += 1
            self.cluster_labels[oi] = max_cluster

        # Update cluster labels in data frame
        self.pd_data['clust_labels'] = self.cluster_labels

    def _makeClusterHDBSCAN_(self, min_cluster_size=2,
                             distance_function='euclidean', min_samples=2):
        """Clusters samples with HDBSCAN library, and saves data
        and cluster labels in a pandas data frame.

        :param min_cluster_size: The min number of clusters
        :type min_cluster_size: int
        :param min_samples: The number of samples (or total weight)
        in a neighborhood for a point to be considered a core point.
        This includes the point itself.
        :type min_samples: int
        :param distance_function: Method used to compute pairwise
        distances. Must be a valid pairwise distance option from
        scikit-learn or scipy.spatial.distance. Also may be a user
        defined distance function.
        :type distance_function: string, or callable

        :return: Pandas dataframe with original data and cluster
        labels as the last column
        :rtype: pandas df
        """

        # print("Fitting model to data.")
        hdbs_clust = hdbscan.HDBSCAN(algorithm='best',
                                     alpha=1.0,
                                     approx_min_span_tree=True,
                                     gen_min_span_tree=False,
                                     leaf_size=40,
                                     metric=distance_function,
                                     min_cluster_size=min_cluster_size,
                                     min_samples=min_samples, p=None).fit(self.data)

        # Save cluster labels into dataframe
        self.cluster_labels = hdbs_clust.labels_
        if isinstance(self.data, pd.DataFrame):
            self.data = self.pd_data
            self.pd_data['clust_labels'] = self.cluster_labels
        else:
            self.pd_data = pd.DataFrame(self.data)
            self.pd_data['clust_labels'] = self.cluster_labels

        # Save probabilities into dataframe
            self.probabilities = hdbs_clust.probabilities_
            self.pd_data['probabilities'] = self.probabilities

    def _makeClusterHAC_(self, distance_function="euclidean",
                         HAC_distance_scaling=1.0, HAC_distance_value=-1,
                         Nclusters=-1):
        """Clusters samples with scipy's hierarchical agglomerative
        clustering and the Ward variance minimizing algorithm. The flat
        clusters are created by a specified distance. Default distance is
        the maximum distance between any two samples in the dataset
        (self.default_distance), and you can adjust the default distance
        with HAC_scaling_distance. Alternatively you can define the distance
        yourself (HAC_distance_value), or define the number of clusters
        (Nclusters). The cluster labels are saved as the last column in a
        dataframe of the original data. This algorithm is not consistent when
        clustering the same data more than once, or clustering in batches.

        :param distance_function: distance metric 'euclidean',
        'seuclidean', 'sqeuclidean', 'beuclidean', or user defined
        function. Defaults to 'euclidean'
        :type distance_function: string or user defined function
        :param HAC_distance_scaling: Scales the default distance
        (self.default_distance), should be greater than zero
        :type HAC_distance_scaling: float
        :param HAC_distance_value: User defines cut-off distance
        for clustering
        :type HAC_distance_value: float
        :param Nclusters: User defines number of clusters
        :type Nclusters: int
        :return: Pandas dataframe with original data and cluster labels
        as the last column
        :rtype: pandas df
        """

        # Create hierarchy
        try:
            dd = self.distance_matrix
            L2 = self.L2
        except BaseException:
            dd = self.computeDistance(self.data, distance_function)
            L2 = sch.linkage(dd, method='ward')
            self.distance_matrix = dd
            self.L2 = L2

        # Default distance to determine number of clusters
        if HAC_distance_value <= 0.0:
            distance = dd.max() * HAC_distance_scaling
        else:
            distance = HAC_distance_value

        # If Nclusters is given, try to find cut-off distance that gives that
        # amount of clusters
        if (Nclusters > 0):

            def clust_func(distance_threshold):
                clust_labels = sch.fcluster(
                    L2, distance_threshold, criterion='distance')
                Nclust_1 = clust_labels.max()
                return abs(Nclust_1 - Nclusters)

            # Brute force then basin hopping to find min
            distance = brute(clust_func, [(0, dd.max()**2)], Ns=50)[0]
            distance = basinhopping(clust_func, distance).x[0]

        # Array of cluster labels for each sample, size Nsamples
        clust_labels = sch.fcluster(L2, distance, criterion='distance')
        self.default_distance = dd.max()
        self.cutoff_distance = distance

        # Save labels to list
        # Subtracting 1 to make it a zero based index
        self.cluster_labels = clust_labels - 1

        # Create pandas dataframe for data and cluster labels
        if isinstance(self.data, pd.DataFrame):
            self.pd_data = self.data
            self.pd_data['clust_labels'] = self.cluster_labels
            self.pd_data.rename(columns=str, inplace=True)
        else:
            self.pd_data = pd.DataFrame(self.data)
            self.pd_data['clust_labels'] = self.cluster_labels
            self.pd_data.rename(columns=str, inplace=True)

    def computeDistance(self, data, distance_function):
        """Compute distances between sample pairs, and stores
        in a condensed 1-D matrix

        :param data: m original observations in n-dimensional space
        :type data: array
        :param distance_function: Either one of the valid
        scipy.cluster.hierarchy distance options or a user defined function.
        :type distance_function: string or callable
        :return: condensed distance matrix
        :rtype: ndarray
        """

        if isinstance(distance_function, type('')):
            # For string option
            if distance_function == 'euclidean':
                # Calculate distance between sample values
                dd = sch.distance.pdist(data, 'euclidean')
            elif distance_function == 'seuclidean':

                dd = sch.distance.pdist(data, 'seuclidean')
            elif distance_function == 'sqeuclidean':
                dd = sch.distance.pdist(data, 'sqeuclidean')
            else:
                print('Error: no valid distance string option given')
                exit()
        elif callable(distance_function):
            dd = distance_function(data)
        else:
            print("Error: no valid distance_function given "
                  "(string or function)")
            exit()

        return dd

    def makeBatchCluster(self, batch_size, convergence_num=2,
                         output='samples', core_sample=True,
                         distance_function="euclidean", verbose=False,
                         Nclusters=-1, eps=.05, min_samples=2, n_jobs=1):
        """Clusters data in batches given specified method

        :param batch_size: Number of samples for each batch. It
        will be adjusted to produce evenly sized batches.
        :type batch_size: int
        :param convergence_num: If int, converged after the data size is the
        same for 'num' iterations. The default is 2. If float, it's converged
        after the change in data size is less than convergence_num*100 percent
        of the original data size.
        :type convergence_num: int >= 2 or float between 0. and 1.
        :param output: Returns the subsamples as pandas dataframe
        ('samples') or as 'indices' in numpy array.
        :type format: string
        :param core_sample: Whether to retain a sample from the
        center of the cluster (core sample), or a randomly chosen sample.
        :type core_sample: bool
        :param distance_function: Either one of the valid
        scipy.cluster.hierarchy distance options or a user defined function.
        :type distance_function: string or callable
        :param verbose: Verbose message
        :type verbose: bool
        :param Nclusters: User defines number of clusters
        :type Nclusters: int
        :param eps: The distance around a sample that defines its neighbors.
        :type eps: float
        :param min_samples: The number of samples (or total weight) in a
        neighborhood for a point to be considered a core point. This
        includes the point itself.
        :type min_samples: int
        :param n_jobs: The number of parallel jobs to run. -1 means
        using all processors.
        :type n_jobs: int
        :returns: subsample of original dataset or indices of subsample
        :rtype: pandas dataframe or numpy array
        """

        batch_data = np.copy(self.data)
        data_pd = pd.DataFrame(batch_data)

        # Create an array to keep track of original indices
        global_indices = np.arange(len(batch_data))
        data_pd['global_ind'] = global_indices
        data_pd.rename(columns=str, inplace=True)

        # Keep track of size of data for each batching iteration
        new_n = batch_data.shape[0]
        is_converged = False

        # Verify convergence_num
        msg = "convergence_num should be an int >= 2, or a float between 0. and 1."

        convergence_int = False
        if (convergence_num > 1.0):
            convergence_int = True
            assert convergence_num >= 2, msg
            assert (np.allclose(int(convergence_num), convergence_num)), msg
            convergence_num = int(convergence_num)
        else:
            assert convergence_num > 0. and convergence_num < 1., msg

        if convergence_int:
            data_size = [new_n] * convergence_num
        else:
            data_size = [new_n]
        # This batching loop will continue until sample size is small enough to
        # cluster all together or sample size has converged.
        total_loss = 0
        while not is_converged:

            # Round up number of batches needed
            nbatches = len(batch_data) // batch_size + 1

            # Saving global indices to shuffle and split into batches
            global_indices = np.array(data_pd['global_ind'])
            np.random.seed(3)
            np.random.shuffle(global_indices)
            indices_groups = np.array_split(global_indices, nbatches)
            n_col = batch_data.shape[1] + 1

            # Cluster each batch, create subsamples, gather all subsamples in
            # clusteredDataArrs
            clusteredDataArrs = np.empty(shape=[0, n_col])

            if verbose:
                verb_bool = False
            else:
                verb_bool = True

            for ii in tqdm.trange(len(indices_groups), disable=verb_bool):

                # Find samples in each batch group
                thisData = data_pd.loc[data_pd['global_ind'].isin(
                    indices_groups[ii])]

                my_cluster = Cluster(
                    np.array(thisData.iloc[:, :self.nFeatures]),
                    method=self.method)

                if self.method == 'DBSCAN':
                    my_cluster.makeCluster(eps=eps,
                                           min_samples=min_samples,
                                           distance_function=distance_function,
                                           Nclusters=Nclusters,
                                           n_jobs=n_jobs)
                else:
                    print("Only DBSCAN is available with batch clustering.")

                # Save the global indices from the current batch
                batch_global_ind = np.array(thisData["global_ind"])

                # batch_global_ind = indices_groups[ii]

                # Save the global indices to the cluster object's pd_data for
                # subsample() to use
                my_cluster.pd_data["global_ind"] = batch_global_ind
                aaa = my_cluster.subsample(distance_function=distance_function,
                                           output='samples',
                                           core_sample=core_sample,
                                           n_jobs=n_jobs)
                # Gather each set of retained samples from the clustered
                # batches
                clusteredDataArrs = np.append(clusteredDataArrs, aaa, axis=0)
                total_loss += my_cluster.loss_estimate

            if verbose:
                print("After batching: " +
                      str(len(clusteredDataArrs)) +
                      " clusters.")

            # Update the data with the retained samples
            batch_data = clusteredDataArrs[:, :self.nFeatures]
            # Update the pandas dataframe with global indices
            data_pd = pd.DataFrame(batch_data)
            data_pd['global_ind'] = clusteredDataArrs[:, -1].astype('int')
            # data_pd.rename(columns=str)

            # Check convergence
            data_size.append(len(clusteredDataArrs))
            if convergence_int:
                last_num = list(data_size[-convergence_num:])
                is_converged = len(set(last_num)) == 1
            else:
                is_converged = abs(data_size[-1]-data_size[-2]) < data_size[0] * convergence_num

        final_result = data_pd
        self.loss_estimate = total_loss

        # Return either samples df or indices
        if output == 'samples':
            r_data = final_result.drop(columns='global_ind')
            if self. scaling_function:

                r_data = self.revertData(r_data)

            return r_data
        else:
            return np.array(final_result['global_ind']).astype(int)

    def subsample(self, distance_function='euclidean', output='samples',
                  core_sample=True, n_jobs=1):
        """Takes a sample from each cluster to form subsample of
        the entire dataset

        :param distance_function: Either one of the valid
        scipy.cluster.hierarchy distance options or a user defined function.
        :type distance_function: string or callable
        :param output: Returns the subsamples as pandas dataframe
        ('samples') or as 'indices' in numpy array.
        :type output: string
        :param core_sample: Whether to retain a sample from the center
        of the cluster (core sample), or a randomly chosen sample.
        :type cores_sample: bool
        :param eps: The distance around a sample that defines its neighbors.
        :type eps: float
        :param n_jobs: The number of parallel jobs to run. -1 means using
        all processors.
        :type n_jobs: int
        :returns: subsample of original dataset or indices of subsample
        :rtype: pandas dataframe or numpy array
        """

        if self.method in ['DBSCAN', 'HAC']:

            if self.method == 'DBSCAN':
                eps = self.eps
            if self.method == 'HAC':
                eps = self.cutoff_distance

            # Counts for each cluster
            values, counts = np.unique(self.cluster_labels, return_counts=True)
            new_samples = np.array([])
            max_neighbor_indices = np.array([])
            new_data = self.pd_data.drop(columns=['clust_labels'])
            np.random.seed(3)
            tot_dist = []
            for ci in values:

                clust_data = self.pd_data[self.pd_data['clust_labels'] == ci]
                if "global_ind" in clust_data.columns:
                    clust_data = clust_data.drop(columns=["global_ind"])

                if clust_data.shape[0] == 1:
                    sample_indices = clust_data.index
                else:
                    np.random.seed(3)
                    if core_sample:
                        eps_c = eps / 2.0
                        neighbors_model = NearestNeighbors(
                            radius=eps_c, algorithm='auto')
                        neighbors_model.fit(clust_data)
                        # for each point, get an array of the points within eps_c
                        neighborhoods = neighbors_model.radius_neighbors(
                            clust_data, return_distance=False)
                        n_neighbors = np.array(
                            [len(neighbors) for neighbors in neighborhoods])
                        most_neighbors = n_neighbors.max()
                        # the indices of all points with a neighborhood of size <most_neighbors>
                        core_sample_indices = np.where(
                            n_neighbors == most_neighbors)[0]
                        # get the dataframe indices for the selected points
                        cs_global_index = clust_data.index[core_sample_indices]
                        max_neighbor_indices = np.append(
                            max_neighbor_indices, cs_global_index)
                        n_core_samples = len(cs_global_index)

                        if n_core_samples >= 1:
                            sample_indices = np.random.choice(
                                cs_global_index, size=1)
                        else:
                            sample_indices = np.random.choice(
                                clust_data.index, size=1)
                    else:
                        sample_indices = np.random.choice(
                            clust_data.index, size=1)

                new_samples = np.append(new_samples, sample_indices)
                # compute loss between sample_indices and removed samples
                if clust_data.shape[0] == 2:
                    dist = self.computeDistance(clust_data, distance_function)
                    tot_dist.append(float(dist))
                elif clust_data.shape[0] > 2:
                    dist = self.computeDistance(clust_data, distance_function)
                    sq_dist = scipy.spatial.distance.squareform(dist)
                    local_index = np.where(
                        clust_data.index == sample_indices[0])[0][0]
                    tot_dist.append(float(sum(sq_dist[local_index])))
                else:
                    tot_dist.append(0.0)

            self.loss_estimate = np.array(tot_dist, dtype=object).sum()
            self.max_neighbor_indices = max_neighbor_indices.astype(int)

        elif self.method == 'HDBSCAN':

            # Take -1 out of labels (outliers)
            labels = np.unique(self.cluster_labels)
            no_outlier_labels = labels[1:]

            # Get representatives from each cluster, save indices
            subsample_indices = []
            for label in no_outlier_labels:
                cluster = self.pd_data[self.pd_data['clust_labels'] == label]
                # In cluster, take sample with max probability as rep.
                cluster_rep = cluster['probabilities'].idxmax()
                subsample_indices = np.append(subsample_indices, cluster_rep)

            # Add indices from all the outliers
            outlier_indices = self.pd_data[self.pd_data['clust_labels'] == -1].index
            subsample_indices = np.append(subsample_indices, outlier_indices)

            new_samples = subsample_indices
            new_data = self.pd_data.drop(
                columns=['clust_labels', 'probabilities'])

        # Check output type
        if output == 'indices':
            return new_samples.astype(int)
        elif output == 'samples':

            r_data = new_data.iloc[new_samples]

            if self.scaling_function:
                r_data = self.revertData(r_data)

            return r_data
        else:
            print("Choose a valid return output: 'indices' or 'samples'.")

    @staticmethod
    def compute_hopkins_statistic(
            data_frame: Union[np.ndarray, pd.DataFrame],
            sampling_size: int) -> float:
        """Assess the clusterability of a dataset. A score between
        0 and 1, a score around 0.5 express no clusterability and a
        score tending to 0 express a high cluster tendency.
        Examples
        --------
                >>> from sklearn import datasets
                >>> from pyclustertend import hopkins
                >>> X = datasets.load_iris().data
                >>> hopkins(X,150)
                0.16

        Copyright (c) 2019, Ismaël Lachheb
        All rights reserved.
        THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
        "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
        LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
        FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
        COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
        INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
        BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
        OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
        AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
        OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF
        THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
        DAMAGE.
        """
        from sklearn.neighbors import BallTree

        def get_nearest_sample(df: pd.DataFrame,
                               uniformly_selected_observations: pd.DataFrame):
            tree = BallTree(df, leaf_size=2)
            dist, _ = tree.query(uniformly_selected_observations, k=1)
            uniformly_df_distances_to_nearest_neighbors = dist
            return uniformly_df_distances_to_nearest_neighbors

        def simulate_df_with_same_variation(
                df: pd.DataFrame, sampling_size: int
        ) -> pd.DataFrame:
            max_data_frame = df.max()
            min_data_frame = df.min()
            uniformly_selected_values_0 = np.random.uniform(
                min_data_frame[0], max_data_frame[0], sampling_size
            )
            uniformly_selected_values_1 = np.random.uniform(
                min_data_frame[1], max_data_frame[1], sampling_size
            )
            uniformly_selected_observations = np.column_stack(
                (uniformly_selected_values_0, uniformly_selected_values_1)
            )
            if len(max_data_frame) >= 2:
                for i in range(2, len(max_data_frame)):
                    uniformly_selected_values_i = np.random.uniform(
                        min_data_frame[i], max_data_frame[i], sampling_size
                    )
                    to_stack = (
                        uniformly_selected_observations,
                        uniformly_selected_values_i)
                    uniformly_selected_observations = np.column_stack(to_stack)
            uniformly_selected_observations_df = pd.DataFrame(
                uniformly_selected_observations)
            return uniformly_selected_observations_df

        def dist_samp_to_nn(
                df: pd.DataFrame, data_frame_sample):
            tree = BallTree(df, leaf_size=2)
            dist, _ = tree.query(data_frame_sample, k=2)
            data_frame_sample_distances_to_nearest_neighbors = dist[:, 1]
            return data_frame_sample_distances_to_nearest_neighbors

        def sample_observation_from_dataset(df, sampling_size: int):
            if sampling_size > df.shape[0]:
                raise Exception(
                    "The number of sample of sample is bigger than "
                    "the shape of D")
            data_frame_sample = df.sample(n=sampling_size)
            return data_frame_sample

        if isinstance(data_frame, np.ndarray):
            data_frame = pd.DataFrame(data_frame)

        data_frame_sample = sample_observation_from_dataset(
            data_frame, sampling_size)

        sample_distances_to_nearest_neighbors = dist_samp_to_nn(
            data_frame, data_frame_sample)

        uniformly_selected_observations_df = simulate_df_with_same_variation(
            data_frame, sampling_size
        )

        df_distances_to_nearest_neighbors = get_nearest_sample(
            data_frame, uniformly_selected_observations_df
        )

        x = sum(sample_distances_to_nearest_neighbors)
        y = sum(df_distances_to_nearest_neighbors)

        if x + y == 0:
            raise Exception(
                "The denominator of the hopkins statistics is null")

        return x / (x + y)[0]

    def hopkins(self, sample_ratio=.1):
        """Calculates the Hopkins statistic or cluster tendency of the data
        from a sample of the dataset. A value close to 0 means uniformly
        distributed, .5 means randomly distributed, and a value close to 1
        means highly clustered.

        :param sample_ratio: Proportion of data for sample
        :type sample_ratio: float, between zero and one
        :return: Hopkins statistic
        :rtype: float
        """

        X = (self.data)

        if ((sample_ratio <= 0) or (sample_ratio > 1)):
            print("Error: cluster.hopkins - sample_ratio must be "
                  "between zero and 1.")
            exit()

        # d = X.shape[1]  # columns
        n = len(X)  # rows
        if n < 300:
            m = n
        else:
            m = int(sample_ratio * n)

        stat = self.compute_hopkins_statistic(X, m)

        return 1 - stat

    def lossPlot(self, val_range=np.linspace(1e-4, 1.5, 10),
                 val_type="raw", distance_function='euclidean',
                 draw_plot=False, min_samples=2, n_jobs=1):
        """Calculates sample size and estimated information loss
        for a range of distance values.

        :param val_range: Range of distance values to use for
        clustering/subsampling
        :type val_range: array
        :param val_type: Choose the type of value range for clustering:
        raw distance ('raw'), scaled distance ('scaled'), or number of
        clusters ('Nclusters').
        :type val_type: string
        :param distance_function: A valid pairwise distance option
        from scipy.spatial.distance, or a user defined distance function.
        :type distance_function: string, or callable
        :param draw_plot: Whether to plot the plt object. Otherwise it
        returns a list of three arrays: the distance value range, loss
        estimate, and sample size. You can pass a matplotlib Axes instance
        if desired.
        :type draw_plot: bool or matplotlib.pyplot.Axes object
        :param min_samples: The number of samples (or total weight) in
        a neighborhood for a point to be considered a core point. This
        includes the point itself.
        :type min_samples: int
        :return: plt object showing loss/sample size information or a
        list [val_range, loss estimate, sample size]
        :rtype: plt object or list of 3 arrays
        """
        val_range = list(val_range)

        sample_size = []
        total_dist = []

        for ival in tqdm.tqdm(val_range):

            if self.method == "DBSCAN":

                if val_type == "raw":
                    self.makeCluster(
                        eps=ival,
                        min_samples=min_samples,
                        n_jobs=n_jobs,
                        distance_function=distance_function)
                elif val_type == "Nclusters":
                    self.makeCluster(
                        Nclusters=ival,
                        min_samples=min_samples,
                        n_jobs=n_jobs,
                        distance_function=distance_function)
                else:
                    print("val_types for DBSCAN are: 'raw', 'Nclusters'")

            elif self.method == 'HAC':

                if val_type == "raw":
                    self.makeCluster(
                        HAC_distance_value=ival,
                        distance_function=distance_function)
                elif val_type == "Nclusters":
                    self.makeCluster(
                        Nclusters=ival, distance_function=distance_function)
                elif val_type == "scaled":
                    self.makeCluster(
                        HAC_distance_scaling=ival,
                        distance_function=distance_function)
                else:
                    print("Choose a valid val_type: "
                          "'scaled', 'raw', 'Nclusters'")
                    return
            else:
                return

            # Number of clusters
            k = len(np.unique(self.cluster_labels))
            # nFeatures = self.data.shape[1]
            sample_size.append(k)

            # Estimating the information loss with distance between samples
            tot_dist = []
            for ci in range(0, k):
                cluster_i = self.pd_data[self.pd_data['clust_labels'] == ci]
                dataN = np.array(cluster_i)[:, :self.nFeatures]

                if dataN.shape[0] == 2:
                    dist = self.computeDistance(dataN, distance_function)
                    tot_dist.append(dist.astype(float))
                elif dataN.shape[0] > 2:
                    dist = self.computeDistance(dataN, distance_function)
                    sq_dist = scipy.spatial.distance.squareform(dist)
                    np.random.seed(3)
                    eps_c = ival / 2.0
                    neighbors_model = NearestNeighbors(
                        radius=eps_c, algorithm='auto')
                    neighbors_model.fit(dataN)
                    neighborhoods = neighbors_model.radius_neighbors(
                        dataN, return_distance=False)
                    n_neighbors = np.array([len(neighbors)
                                            for neighbors in neighborhoods])
                    core_sample = n_neighbors.max()
                    core_sample_indices = np.where(
                        n_neighbors == core_sample)[0]
                    rep = np.random.choice(core_sample_indices)
                    tot_dist.append(sum(sq_dist[rep]))
                else:
                    tot_dist.append(0.0)

            total_dist.append(float(sum(tot_dist)))

        if draw_plot:
            if isinstance(draw_plot, plt.Axes):
                # user sent us where to plot
                ax1 = draw_plot
                fig = ax1.get_figure()
            else:
                fig, ax1 = plt.subplots()

            color = 'tab:red'
            ax1.set_xlabel('Clustering Parameter')
            ax1.set_ylabel('Loss', color=color)
            ax1.plot(val_range, total_dist, color=color)
            ax1.tick_params(axis='y', labelcolor=color)

            # instantiate a second axes that shares the same x-axis
            ax2 = ax1.twinx()

            color = 'tab:blue'

            # we already handled the x-label with ax1
            ax2.set_ylabel('Sample Size', color=color)
            ax2.plot(val_range, sample_size, color=color)
            ax2.tick_params(axis='y', labelcolor=color)

            # otherwise the right y-label is slightly clipped
            fig.tight_layout()

            return fig

        else:
            return [val_range, total_dist, sample_size]


def makeBatchClusterParallel(data, comm,  global_ind, flatten=False,
                             batch_size=3000, convergence_num=2,
                             distance_function="euclidean",
                             scaling_function='', core_sample=True,
                             gather_to=0, output='samples',
                             verbose=False, eps=.05, min_samples=2):
    """
    Clusters data with DBSCAN and returns a list containing:
    1. The reduced dataset or indices of the reduced data
    2. The information loss estimate or the epsilon value found if the
        auto eps algorithm was triggered because eps=-1

    :param data: A Numpy array or Pandas dataframe of shape
    (n_samples, n_features)
    :type data: array
    :param comm:
    :param scaling_function: function for scaling the features
    :type scaling_function: String or callable
    :param flatten: Flattens data to two dimensions.
    (n_samples, n_features_1*n_features_2* ... *n_features_m)
    :type flatten: bool
    """

    from mpi4py import MPI
    rank = comm.Get_rank()
    nprocs = comm.Get_size()

    pverbose = (verbose) and (rank == gather_to)

    if flatten:
        nsamples = data.shape[0]
        nfeatures = np.prod(data.shape[1:])
        data = data.reshape(nsamples, nfeatures)

    if (data.ndim > 2):
        raise ValueError("data needs to be 2 dimensions. "
                         "(n_samples, n_features. Use the "
                         "'flatten' option to convert your data "
                         "to 2 dimensions.")

    nfeatures = data.shape[1]

    # Add global indices to data
    data = np.concatenate((data, global_ind.reshape(-1, 1)), axis=1)

    # Get the sizes of each kosh dataset
    total_data_size = comm.allreduce(data.shape[0], MPI.SUM)

    # Data scaling options
    if scaling_function == '':
        pass
    elif isinstance(scaling_function, type('')):
        data, scale_vars = scaleDataParallel(data,
                                             comm,
                                             scaling_function,
                                             nfeatures,
                                             verbose=verbose,
                                             gather_to=gather_to)
    elif callable(scaling_function):
        data, scale_vars = scaling_function(data,
                                            nfeatures,
                                            revert=False,
                                            scale_vars=[],
                                            comm=comm,
                                            verbose=verbose,
                                            gather_to=gather_to)
    else:
        print("Choose a valid scaling function: 'min_max' or "
              "'standard' or define your own scaling function.")

    is_converged = False
    convergence_int = False

    # Check convergence_num type
    if convergence_num >= 1.0:
        convergence_int = True
        convergence_num = int(convergence_num)

    if convergence_int:
        data_size = [total_data_size] * convergence_num
    else:
        data_size = [total_data_size]
    total_loss = 0
    while not is_converged:

        if pverbose:
            print("Clustering data")

        # ranks cluster data and output smaller data/indices
        rank_cluster = Cluster(data[:, :nfeatures], method='DBSCAN')

        subset_indices = rank_cluster.makeBatchCluster(batch_size=batch_size,
                                                       convergence_num=convergence_num,
                                                       verbose=pverbose,
                                                       core_sample=core_sample,
                                                       eps=eps,
                                                       min_samples=min_samples,
                                                       distance_function=distance_function,
                                                       output='indices')

        # everyone sends # of data to master, wait
        n_subsamples = subset_indices.shape[0]
        total_subsamples = comm.allreduce(n_subsamples, op=MPI.SUM)
        data_size.append(total_subsamples)
        total_loss += rank_cluster.loss_estimate

        if pverbose:
            print("Data size: %s" % total_subsamples)

        # Check convergence
        if convergence_int:
            last_n = data_size[-convergence_num:]
            is_converged = len(set(last_n)) == 1
        else:
            is_converged = abs(data_size[-1]-data_size[-2]) < (data_size[0] * convergence_num)

        if (is_converged):
            retained = data[np.array(subset_indices), :]
            break
        elif (total_subsamples < batch_size):
            if pverbose:
                print(f"Total data size ({total_subsamples}) < batch size ({batch_size})."
                      " Moving all the data to rank {gather_to}")
            # Send all data to primary rank
            #  lowercase "gather" supports GatherV like behavior

            last_data = comm.gather(data[np.array(subset_indices), :], root=gather_to)

            retained = np.array([])
            if (rank == gather_to):
                last_data = np.vstack(last_data)

                # Batch solve
                last_cluster = Cluster(last_data[:, :nfeatures], method='DBSCAN')

                data_sub = last_cluster.makeBatchCluster(batch_size=batch_size,
                                                         convergence_num=convergence_num,
                                                         verbose=pverbose,
                                                         core_sample=core_sample,
                                                         eps=eps,
                                                         min_samples=min_samples,
                                                         distance_function=distance_function,
                                                         output='indices')

                retained = last_data[np.array(data_sub), :]
                total_loss += last_cluster.loss_estimate

            break
        else:

            # everyone sends fraction of data to others
            retained = data[np.array(subset_indices), :]
            np.random.seed(3)
            np.random.shuffle(retained)
            groups = np.array_split(retained, nprocs)
            new_groups = copy.copy(groups)

            for ic in range(nprocs):
                new_groups[ic] = comm.scatter(groups, root=ic)

            data = np.concatenate(new_groups)

    # Print final data size
    final_size = comm.allreduce(retained.shape[0], op=MPI.SUM)
    global_loss = comm.allreduce(total_loss, op=MPI.SUM)
    if pverbose:
        print("Final data size: %s" % final_size)

    if retained.shape[0] > 0:

        # Returning samples
        if output == 'samples':
            if scaling_function == '':
                out = retained[:, :nfeatures]
            elif isinstance(scaling_function, type('')):
                retained = scaleDataParallel(
                    retained,
                    comm,
                    scaling_function,
                    nfeatures,
                    revert=True,
                    scale_vars=scale_vars,
                    verbose=False,
                    gather_to=gather_to)
                out = retained[:, :nfeatures]
            elif callable(scaling_function):
                retained = scaling_function(
                    retained,
                    nfeatures,
                    revert=True,
                    scale_vars=scale_vars,
                    comm=comm,
                    verbose=False,
                    gather_to=gather_to)
                out = retained[:, :nfeatures]

        # Returning indices
        else:
            out = retained[:, nfeatures:].astype(int).flatten()

    # If data size < batch size, all data was sent to primary
    # rank and other ranks return None.
    else:
        out = None

    # No labels since parallel is always batched
    return [out, global_loss]


def numpyParallelReader(inputs, input_sizes, comm):

    rank = comm.Get_rank()
    nprocs = comm.Get_size()

    # Get the sizes of each kosh dataset
    total_data_size = sum(input_sizes)

    # Divide all data as evenly as possible between ranks
    # Some will have this much data
    size_div = total_data_size // nprocs
    # Others will need to +1
    procs_to_add_one = total_data_size % nprocs

    # Create a list of all the data sizes needed
    data_to_procs = np.repeat(size_div, nprocs - procs_to_add_one)
    size_div_p_1 = np.repeat(size_div + 1, procs_to_add_one)
    data_to_procs = np.append(data_to_procs, size_div_p_1)

    # Process for ranks to claim data assigned to them
    counter = 0
    data = []

    # Get global indices
    start = sum(data_to_procs[0:rank])
    end = start + data_to_procs[rank]
    global_ind = np.arange(start, end)
    global_ind = global_ind.reshape(-1, 1)

    for i in range(len(input_sizes)):

        readData = True

        for d in range(input_sizes[i]):
            global_index = counter
            iStart = sum(data_to_procs[0:rank])
            iEnd = iStart + data_to_procs[rank]
            counter += 1
            if (global_index >= iStart) and (
                    global_index <= iEnd) and readData:

                start_local = d
                try:
                    ndata = len(np.concatenate(data))
                except BaseException:
                    ndata = 0

                end_local = min(input_sizes[i], iEnd - iStart + d - ndata)

                file_data = np.load(inputs[i])
                data.append(file_data[start_local: end_local])

                readData = False

    data = np.concatenate(data)
    return data, global_ind


def scaleDataParallel(data, comm, scaling_function,
                      nfeatures, scale_vars=[], revert=False,
                      verbose=False, gather_to=0):

    from mpi4py import MPI
    rank = comm.Get_rank()
    # nprocs = comm.Get_size()

    pverbose = (verbose) and (rank == gather_to)

    if scaling_function == 'min_max':

        if revert is False:
            if pverbose:
                print("Normalizing data")
            local_min = data.min(axis=0)
            local_max = data.max(axis=0)

            global_min = local_min * 0.0
            comm.Allreduce([local_min, MPI.DOUBLE],
                           [global_min, MPI.DOUBLE],
                           op=MPI.MIN)

            global_max = local_max * 0.0
            comm.Allreduce([local_max, MPI.DOUBLE],
                           [global_max, MPI.DOUBLE],
                           op=MPI.MAX)

            for f in range(nfeatures):
                data[:, f] = (data[:, f] - global_min[f]) / \
                    (global_max[f] - global_min[f])

            scale_vars.append(global_min)
            scale_vars.append(global_max)

        else:

            global_min = scale_vars[0]
            global_max = scale_vars[1]

            # Scale back to original units
            for f in range(nfeatures):
                data[:, f] = data[:, f] * \
                    (global_max[f] - global_min[f]) + global_min[f]

    elif scaling_function == 'standard':

        if revert is False:
            if pverbose:
                print("Standardizing data")

            # Get the sizes of each kosh dataset
            total_data_size = comm.allreduce(data.shape[0], op=MPI.SUM)

            local_sum = data.sum(axis=0)
            global_sum = comm.allreduce(local_sum, op=MPI.SUM)
            global_mean = global_sum / total_data_size

            sq_dev = np.zeros((data.shape[0], nfeatures))
            sum_sq_dev = []
            for f in range(nfeatures):
                sq_dev[:, f] = (data[:, f] - global_mean[f])**2
                sum_sq_dev.append(np.sum(sq_dev[:, f], axis=0))

            global_sum_sq_dev = []
            for f in range(nfeatures):
                global_sum_sq_dev.append(
                    comm.allreduce(
                        sum_sq_dev[f], op=MPI.SUM))

            var = np.array(global_sum_sq_dev) / total_data_size
            global_std = np.sqrt(var)

            for f in range(nfeatures):
                data[:, f] = (data[:, f] - global_mean[f]) / global_std[f]

            scale_vars.append(global_mean)
            scale_vars.append(global_std)

        else:

            global_mean = scale_vars[0]
            global_std = scale_vars[1]

            # Scale back to original units
            for f in range(nfeatures):
                data[:, f] = (data[:, f] * global_std[f]) + global_mean[f]

    if revert:
        return data
    else:
        return data, scale_vars


def SubsampleWithLoss(data, target_loss, options, parallel=False, comm=None, indices=None):

    from kosh.sampling_methods.cluster_sampling import Cluster
    from mpi4py import MPI

    rank = comm.Get_rank()
    primary = options.get("gather_to", 0)
    verbose = options.get("verbose", False)
    pverbose = rank == primary and verbose
    scaling_function = options.get("scaling_function", '')
    eps_0 = options.get("eps_0", None)

    def DO_CLUSTER(eps):

        options['eps'] = eps

        method = options.get("method", "DBSCAN")
        if method == "HAC":
            print("Error: HAC not supported for SubsampleWithLoss")
            exit()

        # This needs to do the following
        # - Make a cluster:
        #    - serial, batch, parallel-batch
        # - Subsample
        if not parallel:
            [local_data, labels, loss] = SerialClustering(data, options)
        else:
            [local_data, loss] = ParallelClustering(data, comm, indices, options)

        # - Get/return loss
        return [local_data, loss]

    # Make temporary cluster object of subset of data

    # Get subset of data
    sub_idx = np.random.choice(data.shape[0], size=min([data.shape[0], 250]))
    sub_idx.sort()
    sub_data = data[sub_idx, :]

    distance_function = options.get("distance_function", "euclidean")

    temp_cluster_object = Cluster(
        sub_data,
        scaling_function=scaling_function)

    distances = temp_cluster_object.computeDistance(sub_data,
                                                    distance_function=distance_function)

    # 2) Compute max loss @ epsMax
    epsMax = np.max(distances)
    if parallel:
        epsMax = comm.allreduce(epsMax, MPI.MAX)
    [tmpdata, maxLoss] = DO_CLUSTER(epsMax)

    # Use ave distance of subsample if no eps guess is given
    if eps_0 is None:
        epsGuess = np.mean(distances)
        if parallel:
            epsGuess = comm.allreduce(epsGuess, MPI.SUM)/comm.Get_size()
    else:
        epsGuess = eps_0

    # 3) Optimize to find optimal eps, given targetLoss = epsLoss(eps) / maxLoss(epsMax)

    bounds = [1e-15, epsMax]

    if pverbose:
        print("epsGuess: " + str(epsGuess))

    [Cdata, epsLoss] = DO_CLUSTER(epsGuess)
    non_dim_loss = epsLoss / maxLoss
    if pverbose:
        print("Loss proportion: " + str(non_dim_loss))
    if epsLoss > target_loss*maxLoss:
        guess_above = True
    else:
        guess_above = False

    step_size = epsMax / 20

    reduce_step_size = False

    guesses = []
    losses = []
    while (abs((non_dim_loss - target_loss)/target_loss) > .05):

        if epsLoss > target_loss*maxLoss:
            bounds[1] = epsGuess
            if guess_above is False:
                reduce_step_size = True
                guess_above = True
            epsGuess -= step_size
            if epsGuess <= bounds[0]:
                epsGuess = bounds[1] - ((bounds[1]-bounds[0])/2)
                step_size = (bounds[1] - bounds[0])/4
                reduce_step_size = False

        else:
            bounds[0] = epsGuess
            if guess_above is True:
                reduce_step_size = True
                guess_above = False
            epsGuess += step_size
            if epsGuess >= bounds[1]:
                epsGuess = bounds[0] + ((bounds[1] - bounds[0])/2)
                step_size = (bounds[1] - bounds[0])/4
                reduce_step_size = False

        [Cdata, epsLoss] = DO_CLUSTER(epsGuess)
        non_dim_loss = epsLoss / maxLoss
        losses.append(non_dim_loss)
        guesses.append(epsGuess)

        if reduce_step_size:
            step_size /= 2
            reduce_step_size = False

        if pverbose:
            print("epsGuess: " + str(epsGuess))
            print("Loss proportion: " + str(non_dim_loss))

        if len(guesses) == 14:
            min_index = np.argmin(np.abs(np.array(losses)-target_loss))
            epsGuess = guesses[min_index]
            non_dim_loss = losses[min_index]
            break
    if pverbose:
        print("epsFinal: " + str(epsGuess))
        print("Final Loss proportion: " + str(non_dim_loss))

    # 4) Return results
    return [Cdata, epsGuess]


def SerialClustering(data, options):

    method = options.get("method", "DBSCAN")
    scaling_function = options.get("scaling_function", "")
    flatten = options.get("flatten", False)

    distance_function = options.get("distance_function", "euclidean")
    core_sample = options.get("core_sample", True)
    Nclusters = options.get("Nclusters", -1)
    n_jobs = options.get("n_jobs", 1)
    output = options.get("output", "samples")
    batch = options.get("batch", False)
    batch_size = options.get("batch_size", 10000)
    convergence_num = options.get("convergence_num", 2)

    labels = []

    my_cluster = Cluster(
        data,
        method=method,
        scaling_function=scaling_function,
        flatten=flatten)

    if method == 'DBSCAN':

        eps = options.get('eps', .05)
        min_samples = options.get("min_samples", 2)

        if batch:
            out = my_cluster.makeBatchCluster(batch_size=batch_size,
                                              convergence_num=convergence_num,
                                              output=output,
                                              core_sample=core_sample,
                                              Nclusters=Nclusters,
                                              n_jobs=n_jobs,
                                              eps=eps,
                                              min_samples=min_samples,
                                              distance_function=distance_function)
        else:

            my_cluster.makeCluster(eps=eps, min_samples=min_samples,
                                   distance_function=distance_function,
                                   Nclusters=Nclusters, n_jobs=n_jobs)

            labels = my_cluster.pd_data['clust_labels']

            out = my_cluster.subsample(output=output,
                                       core_sample=core_sample,
                                       distance_function=distance_function,
                                       n_jobs=n_jobs)

    elif method == 'HDBSCAN':

        min_cluster_size = options.get("min_cluster_size", 2)
        min_samples = options.get("min_samples", 2)

        my_cluster.makeCluster(min_cluster_size=min_cluster_size,
                               distance_function=distance_function,
                               min_samples=min_samples)

        labels = my_cluster.pd_data[["clust_labels", "probabilities"]]

        out = my_cluster.subsample(output=output,
                                   core_sample=core_sample,
                                   distance_function=distance_function)

    elif method == 'HAC':

        HAC_distance_scaling = options.get('HAC_distance_scaling', 1.0)
        HAC_distance_value = options.get("HAC_distance_value", -1)

        if batch:
            out = my_cluster.makeBatchCluster(batch_size,
                                              convergence_num=convergence_num,
                                              output=output,
                                              core_sample=core_sample,
                                              distance_function=distance_function,
                                              HAC_distance_scaling=HAC_distance_scaling,
                                              Nclusters=Nclusters,
                                              HAC_distance_value=HAC_distance_value,
                                              n_jobs=n_jobs)
        else:
            my_cluster.makeCluster(distance_function=distance_function,
                                   HAC_distance_scaling=HAC_distance_scaling,
                                   HAC_distance_value=HAC_distance_value,
                                   Nclusters=Nclusters)

            labels = my_cluster.pd_data['clust_labels']

            out = my_cluster.subsample(output=output,
                                       core_sample=core_sample,
                                       distance_function=distance_function,
                                       n_jobs=n_jobs)

    else:
        print("Error: no valid clustering method given")
        exit()

    loss = my_cluster.loss_estimate
    return [out, labels, loss]


def ParallelClustering(data, comm, global_ind, options):

    # Parse the input arguments
    flatten = options.get("flatten", False)
    batch_size = options.get("batch_size", 10000)
    convergence_num = options.get("convergence_num", 2)
    distance_function = options.get("distance_function", "euclidean")
    scaling_function = options.get("scaling_function", "")
    core_sample = options.get("core_sample", True)
    output = options.get("output", "samples")
    eps = options.get('eps', .05)
    min_samples = options.get("min_samples", 2)
    gather_to = options.get("gather_to", 0)
    verbose = options.get("verbose", False)
    # format = options.get("format", "numpy")

    [local_data, loss] = makeBatchClusterParallel(data,
                                                  comm,
                                                  global_ind,
                                                  flatten=flatten,
                                                  batch_size=batch_size,
                                                  convergence_num=convergence_num,
                                                  distance_function=distance_function,
                                                  verbose=verbose,
                                                  gather_to=gather_to,
                                                  scaling_function=scaling_function,
                                                  core_sample=core_sample,
                                                  output=output,
                                                  eps=eps,
                                                  min_samples=min_samples)

    return [local_data, loss]
