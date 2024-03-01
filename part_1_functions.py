# part_1_functions.py
# Created: Tal Daniel (August 2019)
# Updates: Ron Amit (January 2021)

import numpy as np
from scipy.sparse import csr_matrix
from helper_functions import email_pipeline

######################################################################################################


def train_test_split(X, y, test_size):
    """
    :param X: features [n_samples x n_features]
    :param y: labels [n_samples]
    :param test_size: test set size in percentage (0 < test_size < 1)
    :return X_train, X_test, y_train, y_test
    """
    n_samples = X.shape[0]  # total number of samples
    n_train = int((1 - test_size) * n_samples)  # number of training samples
    idxs = np.arange(n_samples)
    new_idx = np.random.permutation(n_samples)
    #creating the training set
    X_train = X[new_idx[0:n_train]]
    X_test = X[new_idx[n_train:n_samples]]
    y_train = y[new_idx[0:n_train]]
    y_test = y[new_idx[n_train:n_samples]]
   
    #X_train = X_test = y_train = y_test = None  # delete this line

    '''    '''
    return X_train, X_test, y_train, y_test
######################################################################################################

"""
The following function calculates the likelihood parameters of each distribution. 
Please take some time to understand how it works.
"""

def estimate_likelihood_params(X, y, dist_type="gaussian", c=0.5, n_classes=2):
    """
    Calculate the likelihood P(X|y,theta)
    :param X: features
    :param y: labels
    :param dist_type: type of distribution: "gaussian", "bernoulli", "multinomial", "multinomial_smooth"
    :param c: smoothing parameter for "multinomial_smooth"
    :param n_classes: number of classes
    :return likelihood_params
    """
    if isinstance(X, csr_matrix):
        X = X.todense()
    n_samples = X.shape[0]
    n_feat = X.shape[1]
    params = {'type': dist_type}
    if dist_type == 'gaussian':
        mu_s = np.zeros((n_classes, n_feat))
        sigmaSqr_s = np.zeros((n_classes, n_feat))
        for i_class in range(n_classes):
            mu_s[i_class] = X[y == i_class].mean(axis=0)
            sigmaSqr_s[i_class] = np.square(X[y == i_class] - mu_s[i_class]).mean(axis=0)
        params['mu'] = mu_s
        params['sigmaSqr'] = sigmaSqr_s

    elif dist_type == 'bernoulli':
        p_s = np.zeros((n_classes, n_feat))
        for i_class in range(n_classes):
            x_i = X[y == i_class]
            # change to 0-1 (binary features)
            x_i[x_i > 0] = 1
            p_s[i_class] = x_i.mean(axis=0)
        params['p'] = p_s

    elif dist_type == 'multinomial':
        p_s = np.zeros((n_classes, n_feat))
        for i_class in range(n_classes):
            x_i = X[y == i_class]
            T = np.sum(x_i)
            p_s[i_class] = np.sum(x_i, axis=0) / T
        params['p'] = p_s
    elif dist_type == 'multinomial_smooth':
        p_s = np.zeros((n_classes, n_feat))
        for i_class in range(n_classes):
            x_i = X[y == i_class]
            T = np.sum(x_i[:]) + c * n_feat
            p_s[i_class] = (c + np.sum(x_i, axis=0)) / T
        params['p'] = p_s
    else:
        print("unknown distribution!")
        return
    return params

######################################################################################################


"""
Implement the Naive Bayes classifier. Complete the code where you are asked to.
"""
class MlabNaiveBayes():
    "This class implement a Naive Bayes Classifier."

    def __init__(self, dist_type="gaussian", n_classes=2, use_log_prob=False):
        """
        Initialize the classifier.
        :param dist_type: type of distribution: "gaussian", "bernoulli", "multinomial", "multinomial_smooth"
        :param n_classes: number of classes
        :param use_log_prob: whether or not to use the log probability instead of the regular probility
        """
        self.dist_type = dist_type
        self.n_classes = n_classes
        self.priors = None  # no priors
        self.likelihood_params = None
        self.use_log_prob = use_log_prob
        self.last_scores = None
    ######################################################################################################

    def fit(self, X, y):
        """
        Train the classifier.
        :param X: features
        :param y: labels
        """
        self.likelihood_params = estimate_likelihood_params(X, y, dist_type=self.dist_type)
        n_samples = y.shape[0]
        priors = np.zeros(self.n_classes)
        for i_class in range(self.n_classes):
          if(i_class == 1):
            priors[i_class] = np.mean(y)
          if(i_class == 0):
            priors[i_class] = 1-np.mean(y)
            '''    '''
        self.priors = priors
    ######################################################################################################

    def predict(self, X):
        """
        Predict labels for features
        :param X: features
        :return y_pred: predictions
        """
        n_samples = X.shape[0]
        if self.priors is None or self.likelihood_params is None:
            print("can't call 'predict' before 'fit'")
            return
        if isinstance(X, csr_matrix):
            X = X.todense()
        self.last_scores = np.zeros((n_samples, self.n_classes))
        y_pred = np.zeros(n_samples)
        for i_sample in range(n_samples):
            sample = X[i_sample, :]
            prior = self.priors
            if not self.use_log_prob:
                # in case use_log_prob == False
                likelihood = self.eval_sample_likelihood(sample)
                if likelihood is None:
                    print("Error: eval_sample_likelihood failed!!!")
                    return y_pred
               
                ## calculate the un-normalized posterior:
                posterior = likelihood*prior
                ## implement the prediction rule:
                y_pred[i_sample] = np.argmax(posterior)
                self.last_scores[i_sample] = posterior
                '''    '''

            else:   # in case use_log_prob == True
                log_likelihood = self.eval_sample_log_likelihood(sample)
                ## calculate the log-posterior (up to additive constant):
                log_posterior=log_likelihood+np.log(prior)
                ## implement the prediction rule:
                y_pred[i_sample] = np.argmax(log_posterior)
                self.last_scores[i_sample] = log_posterior
                '''    '''
        return y_pred
    ######################################################################################################

    def eval_sample_likelihood(self, sample):
        dist_type, likelihood_params, n_classes = self.dist_type, self.likelihood_params, self.n_classes
        likelihood = np.zeros(n_classes)
        if dist_type == 'gaussian':
            mu_s = likelihood_params['mu']
            sigmaSqr_s = likelihood_params['sigmaSqr']
            for i_class in range(n_classes):
                mu = mu_s[i_class]
                sigmaSqr = sigmaSqr_s[i_class]
                if np.any(sigmaSqr == 0):
                    print("Error: cannot predict with Gaussian distribution, SigmaSqr has zeros !!!")
                    return None
                
                norm_factor = 1/(np.sqrt(2*np.pi)*sigmaSqr)
                exp_argument = -0.5*((np.square(sample-mu))/sigmaSqr)
                norm_pdf = norm_factor * np.exp(exp_argument.T)
                likelihood[i_class] = np.prod(norm_pdf)
                """
                Use the vectors mu  [n_features x 1] and sigmaSqr [n_features x 1]
                  (estimated mean and variance vectors,  given the class i_class)
                  and the feature vector sample [n_features x 1]
                """
                # norm_factor =
                # exp_argument =
                # norm_pdf = norm_factor * np.exp(exp_argument)
                # likelihood[i_class] =

                '''    '''
            # end for
        elif dist_type == 'multinomial' or dist_type == 'multinomial_smooth':
            p_s = likelihood_params['p']
            for i_class in range(n_classes):
                p = p_s[i_class]
                likelihood[i_class] = np.prod(np.power(p,sample))
                """
                  Use the vector p [n_features x 1]  (estimated theta probabilities given the class i_class) 
                    and the feature vector sample [n_features x 1]
                """
                # likelihood[i_class] =
                '''    '''
        else:
            raise ValueError("unknown distribution!")
        # end if
        return likelihood
    ######################################################################################################

    def eval_sample_log_likelihood(self, sample):
        dist_type, likelihood_params, n_classes = self.dist_type, self.likelihood_params, self.n_classes
        log_likelihood = np.zeros(n_classes)
        if dist_type == 'multinomial' or dist_type == 'multinomial_smooth':
            p_s = likelihood_params['p']
            for i_class in range(n_classes):
                p = p_s[i_class]
                """

                  Use the vector p [n_features x 1]  (estimated theta probabilites given the class i_class) 
                    and the feature vector sample [n_features x 1]
                """
                log_likelihood[i_class] = np.sum(np.dot(np.log(p),sample.T))
                '''    '''
        else:
            raise ValueError("unknown distribution!")
        return log_likelihood
    ######################################################################################################
######################################################################################################


def calc_err(y_pred, y_true):
    ''''''
    error = np.mean(np.abs(y_pred-y_true)) # change this, we did 
    '''    '''
    return error
######################################################################################################


def run_train_and_test(X, y, train_ratio, test_size):

    # pre-process
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    # use a ratio of the full training set:
    num_train = int(train_ratio * X_train.shape[0])
    X_train = X_train[:num_train]
    y_train = y_train[:num_train]

   
    ## transform
    X_train_augmented = email_pipeline.fit_transform(X_train)
    X_test_augmented = email_pipeline.transform(X_test)
    ## train
    clf = MlabNaiveBayes(dist_type='multinomial_smooth',n_classes=2,use_log_prob=True)
    ##  test
    clf.fit(X_train_augmented,y_train)
    y_pred = clf.predict(X_test_augmented)

    ## calculate error
    err = calc_err(y_pred,y_test)  # change this, done!
    """  """

    return err
######################################################################################################



def run_train_and_test_on_train(X, y, train_ratio, test_size):

    # pre-process
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    num_train = int(train_ratio * X_train.shape[0])
    X_train = X_train[:num_train]
    y_train = y_train[:num_train]
    """
    Your Code Here
    """
    """ Starter code           """          
    ## transform
    X_train_augmented = email_pipeline.fit_transform(X_train)

    ## train
    clf = MlabNaiveBayes(dist_type='multinomial_smooth',n_classes=2,use_log_prob=True)
    ##  test
    clf.fit(X_train_augmented,y_train)
    y_pred = y_pred = clf.predict(X_train_augmented)

    ## calculate error
    err = calc_err(y_pred,y_train)  # change this
    """  """
    return err
######################################################################################################

