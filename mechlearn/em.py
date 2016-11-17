import numpy as np
from scipy.stats import multivariate_normal
import linear_model_fit as lmf
import mm_svm
from sklearn import svm
import pdb
import mm_viterbi as vit
import time
import copy
import mm_systems as mm
import signal

break_loop = False


def signal_handler(signal, frame):
    print('You pressed Ctrl+C!')
    global break_loop
    break_loop = True

signal.signal(signal.SIGINT, signal_handler)


#	Initializes all parameters and stores the samples
#
def init_model(system, dataCopy, guards, normalize_lik=False, scale_prob=False, K=-1):

    print "\n\n####### Initializing EM #######\n\n"
    all_datasets = lmf.split_datasets(dataCopy)
    datasets_per_model = lmf.group_models(all_datasets)
    K = len(datasets_per_model)  # True number of modes in the dataset

    print "***** Initial model estimates *****"
    # Initial estimate of the parameters using completely random data
    for i in range(K):
        [A, B] = lmf.linear_model_fit(datasets_per_model[i])
        system.Am.append(A)
        print "Model " + str(i) + ": A is \n" + str(A)
        if B != None:
            system.Bm.append(B)
            print "Model " + str(i) + ": B is \n" + str(B) + "\n"

    # Extracts the number of samples
    n = system.x.shape[0]

    # Dimension of the state vector
    dimx = system.x.shape[1]

    # Tests if there is any input
    if system.u.size == 0:
        dimu = 0
        print "System with no input."
    else:
        # Dimension of the input vector
        dimu = system.u.shape[1]
        print "Input with dimension " + str(dimu)

    # If Gaussian measurement likelihood should be normalized, makes the
    # normalization factor equal to the maximum value of the likelihood
    # across all models.
    if normalize_lik:
        mean = np.zeros(dimx)
        max_lik = -1
        for cov in system.Qm:
            val = multivariate_normal.pdf(mean, mean, cov)
            if val > max_lik:
                max_lik = val

            system.lik_norm = max_lik
            system.lik_norm_flag = True

    else:
        system.lik_norm_flag = False

    # If required, initializes data structures to allow the normalization
    # of alphas and betas (useful when dealing with large numbers of
    # data points)
    if scale_prob:
        system.scale_alpha_beta = True
        system.scale_vector = np.zeros(system.n)
    else:
        system.scale_alpha_beta = False

    #
    #	mj_1: current mode from where the transition originates
    #	m_active: guard that is currently active (Ac^mj_1->m_active){xj_1,uj_1})
    #	m_j: next mode
    #
    system.ptrans = np.ones([K, K, K]) * (1.0 / K)  # transition probabilities

    system.pm1 = np.ones([K]) * (1.0 / K)  # probability of initial mode
    system.n = n  # number of samples (just to make thing easier)
    system.K = K  # number of classes (just to make things easier)

    # No initial mode vector was provided, so generates a "uniformly"
    # (in fact, classes 0 and K-1 have half the probability of all the
    # other ones, but that doesn't really matter)
    # distributed array of modes with n samples distributed over K
    # classes. Converts the numbers to integers so that they can be
    # used as indexes.
    # if randomize:
    #	system.m = np.rint(np.random.rand(n)*(K-1)).astype(int)
    #	guards = mm_svm.svm_fit(data_list, K)
    #       print "Using randomly-generated mode assignments."
    # else:
    #	print "Using provided mode assignments."
    system.guards = guards

    # Forward-Backward probabilities
    system.ag_mat = np.zeros([n, K])
    system.bg_mat = np.zeros([n, K])

    # gamma_mat[i,j] = p(mi=j|x,u)
    system.gamma_mat = np.zeros([n, K])
    # xi_mat[i_1,mi_1,m_i] = p(mi_1,m_i|x,u), i_1 ending in n-2 (instead of
    # n-1)
    system.xi_mat = np.zeros([n - 1, K, K])

    print "\n\n####### EM initialized #######\n\n"

# Assumes initial model estimates & guard conditions are stored in system
# Assumes init_model has already been called


def learn_model_from_data(system, data, true_data, is_jmls, svm_mis_penalty=1000000):
    numSteps = 0
    avgExTime = 0
    iter_count = 0
    prev_iter_sign = 0
    conv_count = 0
    max_steps = 200
    dataCopy = copy.deepcopy(data)
    K = len(system.Am)
    prev_log_lik = avg_log_likelihood(system)
    global break_loop
    break_loop = False

    print "Initial Mode Assignment Distributions:"
    for i in range(system.K):
        print "Model " + str(i) + ": " + str(np.sum([system.m == i]))

    for i in range(max_steps):
        print "\nStep " + str(i) + ""
        # E step
        st = time.clock()
        compute_alphas(system)
        compute_betas(system)
        compute_gamma(system)
        compute_xi(system)

    #     #vit.log_viterbi(system)
        approx_viterbi(system)
        dataCopy.m = np.copy(system.m)
        for i in range(system.K):
            print "Model " + str(i) + ": " + str(np.sum([system.m ==  i]))

        if not(is_jmls):
            #Estimates the activation regions using estimated data
            data_list = mm_svm.svm_split_data(dataCopy, K)
            classifier_list = mm_svm.svm_fit(data_list, K, mis_penalty=svm_mis_penalty)
            system.guards = classifier_list

        #M step
        max_initial_mode_probs(system)
        max_linear_models(system)
        max_guarded_trans_probs(system)
        avgExTime += time.clock()-st

        avg_likelihood = avg_log_likelihood(system)
        print "Average log-likelihood: "+str(avg_likelihood)

        numSteps += 1
        lik_diff = prev_log_lik-avg_likelihood
        iter_sign = 1 if lik_diff > 0 else -1
        if iter_sign != prev_iter_sign:
            iter_count+=1
        else:
            iter_count = 0
        if iter_count >= 3:
            print "\n\n********* Model converged *********\n\n"
            break
        prev_iter_sign = iter_sign

        if np.abs(lik_diff)<1:
            conv_count+=1
        else:
            conv_count = 0

        if conv_count >= 5:
            print "\n\n********* Model converged *********\n\n"
            break
        prev_log_lik = avg_likelihood
    
        if break_loop:
            break
    avgExTime /= numSteps
    print "\nDone!\n"
    #error = mm.simulate_mm_system(system).x - true_data.x
    #system.W = np.identity(system.x.shape[0])
    # for i in range(system.x.shape[0]):
    #	system.W[i,i] = ((2/3)*np.max(error[i,:]))**2

    smooth_ptrans(system)

    return system, data, numSteps, avgExTime


def smooth_ptrans(system, max_prob=0.99):
    """Turns deterministic transitions into slightly probabilistic ones."""
    for mi_1 in range(system.ptrans.shape[0]):
        for m_ac in range(system.ptrans.shape[1]):
            max_prob_idx = np.argmax(system.ptrans[mi_1, m_ac, :])
            if system.ptrans[mi_1, m_ac, max_prob_idx] >= max_prob:
                system.ptrans[mi_1, m_ac, max_prob_idx] = max_prob
                for mi in range(len(system.ptrans[mi_1, m_ac, :])):
                    if(mi != max_prob_idx):
                        system.ptrans[mi_1, m_ac, mi] = (
                            1.0 - max_prob) / (len(system.ptrans[mi_1, m_ac, :]) - 1.0)


def compute_alphas(system):
    n = system.n
    K = system.K
    system.ag_mat = np.zeros([n, K])  # ag_mat(i,j) = ag(mi=j)

    system.ag_mat[0, :] = system.pm1  # current guess of initial mode prob.

    if system.scale_alpha_beta:
        factor = np.sum(system.ag_mat[0, :])
        if factor > 0.0:
            system.ag_mat[0, :] = system.ag_mat[0, :] / factor
            system.scale_vector[0] = factor  # will be used with betas
        else:
            print "ERROR: Scaling factor for alpha at step " + str(0) + " is " + str(factor)
            return

    for i_1 in range(1, n):
        for mi_1 in range(K):
            for mi_2 in range(K):
                system.ag_mat[i_1, mi_1] += system.ag_mat[i_1 - 1, mi_2] * dynamics_lik(
                    i_1, mi_2, system) * guarded_trans_prob(i_1, mi_2, mi_1, system)
                # pdb.set_trace()
        if system.scale_alpha_beta:
            factor = np.sum(system.ag_mat[i_1, :])
            if factor > 0.0:
                system.ag_mat[i_1, :] = system.ag_mat[i_1, :] / factor
                system.scale_vector[i_1] = factor  # will be used with betas
            else:
                print "ERROR: Scaling factor for alpha at step " + str(i_1) + " is " + str(factor)
                break


def compute_betas(system):
    n = system.n
    K = system.K
    system.bg_mat = np.zeros([n, K])  # bg_mat(i,j) = bg(mi=j)

    system.bg_mat[n - 1, :] = np.ones(K)

    if system.scale_alpha_beta:
        factor = system.scale_vector[n - 1]
        system.bg_mat[n - 1, :] = system.bg_mat[n - 1, :] / factor

    for i_1 in range(n - 2, -1, -1):  # Counts backwards
        for mi_1 in range(K):
            for mi in range(K):
                #print "N:"+str(system.bg_mat[i_1+1,mi])+" DLK:"+str(dynamics_lik(i_1+1,mi_1,system))+" GTP:"+str(guarded_trans_prob(i_1+1,mi_1,mi,system))
                system.bg_mat[i_1, mi_1] += system.bg_mat[i_1 + 1, mi] * dynamics_lik(i_1 + 1, mi_1, system) * guarded_trans_prob(i_1 + 1, mi_1, mi, system)
        if system.scale_alpha_beta:
            factor = system.scale_vector[i_1]
            system.bg_mat[i_1, :] = system.bg_mat[i_1, :] / factor


def compute_gamma(system):

    for i_1 in range(system.n):
        norm = 0
        for mi_1 in range(system.K):
            system.gamma_mat[i_1, mi_1] = system.ag_mat[i_1, mi_1] * system.bg_mat[i_1, mi_1]
            norm += system.gamma_mat[i_1, mi_1]

        if(norm > 0.0):
            system.gamma_mat[i_1, :] = system.gamma_mat[i_1, :] / norm
        else:
            print "ERROR: Normalization factor for gamma at step " + str(i_1) + " is " + str(norm)
            print str(system.ag_mat[i_1,:])
            print str(system.bg_mat[i_1,:])
            print str(system.gamma_mat[i_1,:])
            system.gamma_mat[i_1, :] = np.ones([system.K]) / (system.K)


def compute_xi(system):

    for i_1 in range(system.n - 1):
        norm = 0
        for mi_1 in range(system.K):
            for mi in range(system.K):
                system.xi_mat[i_1, mi_1, mi] = system.ag_mat[i_1, mi_1] * dynamics_lik(
                    i_1 + 1, mi_1, system) * guarded_trans_prob(i_1 + 1, mi_1, mi, system) * system.bg_mat[i_1 + 1, mi]
                norm += system.xi_mat[i_1, mi_1, mi]

        if(norm > 0.0):
            system.xi_mat[i_1, :, :] = system.xi_mat[i_1, :, :] / norm
        else:
            print "ERROR: Normalization factor for xi at step " + str(i_1) + " is " + str(norm)


#	For a linear dynamical model with Gaussian noise, returns the
#	likelihood of xi being generated by xi_1, ui_1 in model mi_1.
#
def dynamics_lik(i, mi_1, system):

    xi = system.x[i, :]
    xi_1 = system.x[i - 1, :]
    A = system.Am[mi_1]
    cov = system.Qm[mi_1]
    mean = np.dot(A, xi_1)

    # System with input
    if len(system.Bm) > 0:
        ui_1 = system.u[i - 1, :]
        B = system.Bm[mi_1]
        mean += np.dot(B, ui_1)

    # Because this is a density, numbers tend to get really high
    # alpha proceed. Therefore, it might make sense to divide the value
    # of the density by its maximum value in order to avoid numerical
    # problems.

    # pdb.set_trace()
    #print "LN:"+str(system.lik_norm)
    #print "A:"+str(A)
    #print "XI:"+str(xi)+", m:"+str(mean)+", cov:"+str(cov)
    #print "XI_1:"+str(xi_1)
    #print "MVPDF:"+str(multivariate_normal.pdf(xi, mean, cov))
    if system.lik_norm_flag:
        return multivariate_normal.pdf(xi, mean, cov) / system.lik_norm
    else:
        return multivariate_normal.pdf(xi, mean, cov)

# Returns the probability of the transition mi_1 -> mi depending on
# the current active guard.


def guarded_trans_prob(i, mi_1, mi, system):

    # Input data for the guard region
    xi_1 = system.x[i - 1, :]
    if(system.u.size > 0):
        ui_1 = system.u[i - 1, :]
        inputData = np.array([np.hstack((xi_1, ui_1))])
    else:
        inputData = np.array([xi_1])

    # Queries the SVM for the current active guard, given the input data
    # and the fact that we are in mode mi_1
    m_active = int(system.guards[mi_1].predict(inputData)[0])

    # Returns the probability of transitioning to mode mi, given that
    # we are in mode mi_1 and that the guard condition mi_1->m_active
    # is active. We hope to observe a high probability value for this
    # transition iff mi = m_active
    return system.ptrans[mi_1, m_active, mi]

# Returns a 1 for every sample with an active guard between mi_1
# and m_active


def get_guard_indices(mi_1, m_active, system):

    if system.u.size > 0:
        inputData = np.array([np.hstack((system.x[:system.n - 1], system.u[:system.n - 1]))])
    else:
        inputData = np.array([system.x[:system.n - 1]])

    # Active guards for every sample considering the current state
    # equals to mi_1
    active_guards = system.guards[mi_1].predict(inputData)

    # Labels all the samples with an active mi_1 -> m_active transition
    # with a 1
    indices = np.zeros(active_guards.shape)
    for i in range(active_guards.size):
        if active_guards[i] == m_active:
            indices[i] = 1

    return indices


#	M-step for the initial mode probabilities
#
def max_initial_mode_probs(system):
    # Copies marginal probabilities for m1
    system.pm1 = np.copy(system.gamma_mat[0, :])
    # Normalizes
    system.pm1 = system.pm1 / np.sum(system.pm1)


#	M-step for estimating linear models
#
def max_linear_models(system):

    for k in range(system.K):
        # Selects marginal probabilities for that model in all samples
        weights = system.gamma_mat[0:system.n - 1, k]
        # Uses weighted data set to estimate a new model
        print "Weights:"+str(weights)
        [A, B] = lmf.linear_model_fit(system, weights)
        # Updates model
        system.Am[k] = A
        if not B == None:
            system.Bm[k] = B

#	M-step for estimating transition probabilities
#
#


def max_guarded_trans_probs(system):

    for mi_1 in range(system.K):
        for m_active in range(system.K):
            for mi in range(system.K):
                # Gets list of indices of xs and us that are in the active
                # and inactive regions for a transition kmi_1->kmi
                ac_ind = get_guard_indices(mi_1, m_active, system)

                # Sum of posterior probabilities for active transitions
                sum_xis = np.dot(system.xi_mat[:, mi_1, mi], ac_ind)

                # Ensures that the probability will never end as 0
                if(sum_xis == 0.0):
                    sum_xis = 1.0e-10
                # Probability estimate for current active region
                system.ptrans[mi_1, m_active, mi] = sum_xis

            # Normalization
            sum_probs = np.sum(system.ptrans[mi_1, m_active, :])
            if sum_probs > 0.0:
                system.ptrans[mi_1, m_active, :] = system.ptrans[
                    mi_1, m_active, :] / sum_probs
            else:
                # system.ptrans[mi_1, m_active,:] = np.ones(
                system.ptrans[mi_1, m_active, :] = np.ones(
                    [system.K]) * (1.0 / system.K)
                print "\nERROR: guard mi_1= " + str(mi_1) + "->m_active= " + str(m_active) + " has zero probability\n"


# Computes the log-likelhood of the data
def avg_log_likelihood(system):

    loglik = 0.0

    for mi_1 in range(system.K):
        for i in range(1, system.n):
            if system.gamma_mat[i - 1, mi_1] != 0.0:
                loglik += system.gamma_mat[i - 1, mi_1] * \
                    np.log(dynamics_lik(i, mi_1, system))

    for mi in range(system.K):
        for mi_1 in range(system.K):
            for i_1 in range(system.n - 1):
                if system.xi_mat[i_1, mi_1, mi] != 0.0:
                    loglik += system.xi_mat[i_1, mi_1, mi] * \
                        np.log(guarded_trans_prob(i_1 + 1, mi_1, mi, system))

    for m0 in range(system.K):
        if system.gamma_mat[0, m0] != 0.0:
            loglik += system.gamma_mat[0, m0] * np.log(system.pm1[m0])

    return loglik

# This function approximates the output of Viterbi by taking the
# sequence of maximum a posteriori mode assignments based on the computed
# gammas


def approx_viterbi(system):
    for i in range(system.n):
        system.m[i] = np.argmax(system.gamma_mat[i, :])


def em_svm_training(mi_1, system):

    if system.u.size > 0:
        inputData = np.concatenate((system.x[:-1], system.u[:-1]), axis=1)
    else:
        inputData = np.copy(system.x[:-1])

    m_matrix = np.zeros([system.n - 1, system.K])
    m_training = np.zeros([system.n - 1])

    for i in range(1, system.n):
        for m in range(system.K):
            for mi in range(system.K):
                m_matrix[i - 1, m] += system.xi_mat[i - 1, mi_1,
                                                    mi] * np.log(system.ptrans[mi_1, m, mi])

        m_training[i - 1] = np.argmax(m_matrix[i - 1, :])

    # Only 0 or 1 labels
    if (np.sum(m_training) == 0.0):
        m_training = np.append(m_training, 1)
        inputData = np.vstack((inputData, np.zeros(inputData[0].shape)))
    else:
        if (np.sum(m_training) == (system.n - 1)):
            m_training = np.append(m_training, 0)
            inputData = np.vstack((inputData, np.zeros(inputData[0].shape)))

    #	I believe this is the wrong way to train the SVM
    #~ for i in range(system.n-1):
        #~ m_array[i] = np.argmax(system.xi_mat[i,mi_1,:])

    training_data = mm_svm.svm_data()

    training_data.x = inputData
    training_data.y = m_training

    return training_data


def em_svm_fit(training_data, K, kernel_type='linear', svm_mis_penalty=1000000):

    # Misclassification penalty
    #mis_penalty = 1000000
    #mis_penalty = 1000

    # Estimate SVMs
    #svm_ob = svm.SVC(kernel=kernel_type,C = mis_penalty, class_weight={1: 1})
    svm_ob = svm.SVC(kernel=kernel_type, C=svm_mis_penalty)
    svm_ob.fit(training_data.x, training_data.y)

    return svm_ob


# Returns the root mean square norm of the difference between two
# vectors.
def rmse(v1, v2):
    diff = v1 - v2

    # unidimensional differnce
    if(len(diff.shape) == 1):
        return np.sqrt(np.dot(diff, diff))
    else:
        n, dim = diff.shape
        rmse = 0.0
        # RMSE across all dimensions
        for i in range(dim):
            rmse += np.dot(diff[:, i], diff[:, i])

        return np.sqrt(rmse / n)

# Counts the number of different entries between two sequences


def count_different(seq1, seq2):
    length = min(len(seq1), len(seq2))

    errors = 0
    for i in range(length):
        if seq1[i] != seq2[i]:
            errors += 1

    return errors
