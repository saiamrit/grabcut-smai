import numpy as np
# from sklearn.cluster import KMeans
import warnings
import datetime
from sklearn.mixture import GaussianMixture
import tqdm
warnings.filterwarnings("ignore")

gmm_components = 5
gamma = 30
neighbours = 8 
color_space = 'RGB'
n_iters = 10


class GMM:
    """
    An extension to the GaussianMixture class of scikit-learn.
    This class is a wrapper above the library package to provide
    a few extra functions.
    """
    def __init__(self, X_data, n_components):
        self.X_data = X_data
        self.n_components = n_components
        self.components = None
        
        # Initialize the actual the GMM
        self.model = GaussianMixture(n_components=self.n_components).fit(self.X_data)
        
        # Update model parameters
        self.update_params(self.model.predict(self.X_data))
    
    def get_components(self, x_data):
        preds = self.model.predict(x_data)
        return preds
    
    # def fit(self, x_data, labels):
    #     pass
    
    def update_params(self, labels):
        """
        Function to update the parameters of this wrapper
        for easier access to useful  properties.
        """
        # Update the components labeling and the label counts
        self.components = labels
        self.counts = np.unique(labels, return_counts=True)[1]
        self.weights = self.counts / len(labels)
        
        # Save means and covariances from the model
        self.means = self.model.means_
        self.cov = self.model.covariances_ + 1e-08*np.eye(self.means.shape[1])
    
    def compute_D(self, data=None):
        if data is None:
            data = self.X_data
            components = self.components
        else:
            components = self.model.predict(data)
        
        diffs = data.reshape((-1, 1, 3)) - self.means.reshape((1, -1, 3))
        
        matrix_term = 0.5 * np.einsum('ijk,kji->ij', np.einsum('ijk,jkl->ijl', diffs, np.linalg.inv(self.cov)), diffs.T)
        det_term = 0.5 * np.log(np.linalg.det(self.cov))
        
        D_val = -np.log(self.weights) + det_term + matrix_term
        
        return D_val[range(len(components)), components]




# class GaussianMixture:
#     def __init__(self, X, gmm_components):
#         self.n_components = gmm_components
#         self.n_features = X.shape[1]
#         self.n_samples = np.zeros(self.n_components)

#         self.coefs = np.zeros(self.n_components)
#         self.means = np.zeros((self.n_components, self.n_features))
#         self.covariances = np.zeros(
#             (self.n_components, self.n_features, self.n_features))

#         self.init_with_kmeans(X)

#     def init_with_kmeans(self, X):
#         label = KMeans(n_clusters=self.n_components, n_init=1).fit(X).labels_
#         self.fit(X, label)


#     def score_formula(self, mult, mat):
#         score = np.exp(-.5 * mult) / np.sqrt(2 * np.pi)/np.sqrt(np.linalg.det(mat))
#         return score

#     def calc_score(self, X, ci):
#         score = np.zeros(X.shape[0])
#         if self.coefs[ci] > 0:
#             diff = X - self.means[ci]
#             Tdiff = diff.T
#             inv_cov = np.linalg.inv(self.covariances[ci])
#             dot = np.dot(inv_cov, Tdiff)
#             Tdot = dot.T
#             mult = np.einsum('ij,ij->i', diff, Tdot)
#             score = self.score_formula(mult,self.covariances[ci])
#         return score

#     def calc_prob(self, X):
#         prob = []
#         for ci in range(self.n_components):
#             score = np.zeros(X.shape[0])
#             if self.coefs[ci] > 0:
#                 diff = X - self.means[ci]
#                 Tdiff = diff.T
#                 inv_cov = np.linalg.inv(self.covariances[ci])
#                 dot = np.dot(inv_cov, Tdiff)
#                 Tdot = dot.T
#                 mult = np.einsum('ij,ij->i', diff, Tdot)
#                 score = self.score_formula(mult,self.covariances[ci])
#             prob.append(score)
#         ans = np.dot(self.coefs, prob)
#         return ans

#     def which_component(self, X):
#         prob = []
#         for ci in range(self.n_components):
#             score = self.calc_score(X,ci)
#             prob.append(score)
#         prob = np.array(prob).T
#         return np.argmax(prob, axis=1)

#     def fit(self, X, labels):
#         assert self.n_features == X.shape[1]
#         self.n_samples[:] = 0
#         self.coefs[:] = 0
#         uni_labels, count = np.unique(labels, return_counts=True)
#         self.n_samples[uni_labels] = count
#         variance = 0.01
#         for ci in uni_labels:
#             n = self.n_samples[ci]
#             sum = np.sum(self.n_samples)
#             self.coefs[ci] = n / sum
#             self.means[ci] = np.mean(X[ci == labels], axis=0)
#             if self.n_samples[ci] <= 1:
#                 self.covariances[ci] = 0
#             else:
#                 self.covariances[ci] =  np.cov(X[ci == labels].T)
#             det = np.linalg.det(self.covariances[ci])
#             if det <= 0:
#                 self.covariances[ci] += np.eye(self.n_features) * variance
#                 det = np.linalg.det(self.covariances[ci])