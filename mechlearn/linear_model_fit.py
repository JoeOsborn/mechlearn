import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
import mm_systems as mm
import pdb

#	Takes a list of data sets, pl
#
#


def linear_model_fit(data_list, weights=np.array([])):

    if not isinstance(data_list, list):
        data_list = [data_list]

    [nx, xdim] = data_list[0].x.shape

    # Builds the LSTSQ estimation matrices
    Phi, y = __build_lstsq(data_list)

    if not (Phi == None or y == None):
        # Solving Phi*Xk_1 = Xk
        if weights.size == 0:  # No weights
            theta = np.dot(linalg.pinv(Phi), y)
        else:  # Weighted least squares
            W = np.diag(weights)
            A = Phi.transpose().dot(W).dot(Phi)
            theta = linalg.inv(A).dot(Phi.transpose()).dot(W).dot(y)

        A = theta[:xdim, :]
        A = A.transpose()

        if not data_list[0].u.size == 0:
            B = theta[xdim:, :]
            B = B.transpose()
            return A, B
        else:
            return A, None
    else:
        return None, None


# Splits the data from different models into separate data structures
#
#	Xk = A(m-1)Xk-1+B(m-1)
#
#	Concatenating state vectors from different transition periods
#
#
#
def split_datasets(data):

    # Extracts the transition indices from the mode sequence
    data.get_transition_indices()

    # No transitions happened, so there is a single data set
    if len(data.trans_index) == 0:
        data_sets = [data]
    else:
        #	Initial list of data sets. There should be
        # 	len(data.trans_index)+1 in total at the end
        data_sets = []
        ind_acc = 0
        for t in data.trans_index:
            # New data set
            data_sets.append(mm.MMData())
            data_sets[-1].num_models = 1
            # Copies data up to the transition, plus the first state and
            # input vectors for the next mode. These will be used to
            # estimate the dynamical models of the current mode (the last
            # input vector will eventually be discarded, as it only affects
            # future state vectors).
            data_sets[-1].x = np.vstack(data.x[ind_acc:t + 2, :])
            if data.u.size > 0:
                data_sets[-1].u = np.vstack(data.u[ind_acc:t + 2])
            if data.m.size > 0:
                data_sets[-1].m = np.hstack(data.m[ind_acc:t + 2])

            ind_acc = t + 1

            # pdb.set_trace()

        # Last portion of the data
        data_sets.append(mm.MMData())
        data_sets[-1].x = np.vstack(data.x[ind_acc:, :])

        if data.u.size > 0:
            data_sets[-1].u = np.vstack(data.u[ind_acc:])
        if data.m.size > 0:
            data_sets[-1].m = np.hstack(data.m[ind_acc:])

    return data_sets


#	Takes a list of datasets and groups them into lists of datasets
#	corresponding to the same mode
#
def group_models(data_sets):
    # pdb.set_trace()
    if((not isinstance(data_sets, list)) or (len(data_sets) == 1)):
        data_models = [data_sets]  # single data set
    else:
        # Recovers a list of modes present in the data
        m_lst = []
        for data in data_sets:
            if not (data.m[0] in m_lst):
                m_lst.append(data.m[0])

        # One container per model (assuming that they are ordered from 0)
        data_models = [[] for i in range(len(m_lst))]

        # pdb.set_trace()
        for data in data_sets:
            data_models[data.m[0]].append(data)

    return data_models


#	Takes a list of data sets for the same model and properly assembles
#	them for least squares estimation
#
def __build_lstsq(data_list):

    Phi_lst = []
    y_lst = []

    # Builds the estimation matrices by properly stacking the data from
    # different blocks.
    for data_points in data_list:

        Xk_1 = data_points.x[:-1, :]  # All but last row

        if data_points.u.size == 0:
            Phi_lst.append(Xk_1)
        else:
            nx = data_points.x.shape[0]
            nu = data_points.u.shape[0]

            # The input value for the last state vector is missing
            if nu == nx - 1:
                Phi_lst.append(np.hstack((Xk_1, data_points.u)))
            else:
                # Discards the input value for the last state vector, since
                # it has no effect on the dynamics
                if nu == nx:
                    Phi_lst.append(np.hstack((Xk_1, data_points.u[:-1])))
                else:
                    print "Received " + str(nx) + " state vectors and " + str(nu) + " input vectors."
                    return None, None

        Xk = data_points.x[1:, :]  # All but first row
        y_lst.append(Xk)

    # Stacks all elements once - efficiency
    Phi = np.vstack(Phi_lst)
    y = np.vstack(y_lst)

    return Phi, y
