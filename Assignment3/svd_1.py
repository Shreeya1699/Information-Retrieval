import numpy as np
from numpy.linalg import svd as getsvdparam
import math
import operator
import time
from numpy import linalg as LA
from numpy.linalg import inv
import scipy.sparse.linalg as la



def getsvdparams(M, k):
	MT = M.transpose()
	MMT = M*MT
	MTM = MT*M
	if len(M) < len(M[0]):
		if k < min(len(M),len(M[0])):
			s,U = la.eigsh(MMT,k)
		else:
			s,U = LA.eigh(MMT)
		U = np.fliplr(U)
		s = np.flipud(s)
		s = np.sqrt(s)
		S = np.diag(s)
		Vt = np.dot(np.dot(inv(S), U.transpose() ), M)
	else:
		if k < min(len(M),len(M[0])):
			s,V = la.eigsh(MTM,k) #calculating eigen values and vectors
		else:
			s,V = LA.eigh(MTM)
		V = np.fliplr(V)
		s = np.flipud(s)
		s = np.sqrt(s)
		S = np.diag(s)
		U = M * V * inv(S)
		Vt = V.transpose
	return np.matrix(U),np.matrix(Vt),np.matrix(S),s

r=0
# Predict function for SVD
def predict(A, B, VT, user_offset, temp):
    V = VT.T
    number_of_predictions = 0
    squared_error_sum = 0
    error_sum = 0
    for i in range(len(A)):
        qV = np.dot(A[i], V)
        rating_for_q = np.dot(qV, VT)
        rating_for_q = rating_for_q + user_offset[i]

        for j in range(len(A[i])):
            if (B[i][j] != 0 and A[i][j] + user_offset[i] != B[i][j]):
                number_of_predictions += 1
                squared_error_sum += (rating_for_q[j] - B[i][j]) ** 2
                error_sum += abs(rating_for_q[j] - B[i][j])
                temp[i][j] = rating_for_q[j]
    frobenius_norm = math.sqrt(squared_error_sum)
    # print("No. of predictions: " + str(number_of_predictions))

    # Root mean square
    rmse = float(frobenius_norm / float(number_of_predictions))
    mae = float(error_sum) / float(number_of_predictions)

    return mae, rmse


def get_new_VT(VT, eigen_values):
    temp = []
    sum_of_squared_eigenvalues = 0.0
    for i in range(len(eigen_values)):
        temp.append([i, eigen_values[i]])
        sum_of_squared_eigenvalues += eigen_values[i] ** 2
    sorted_eigenvalues = sorted(temp, key=operator.itemgetter(1), reverse=True)
    allowed_loss_of_energy = 0.1 * sum_of_squared_eigenvalues

    sum = 0
    for i in range(len(eigen_values)):
        if (sum + eigen_values[-i - 1] ** 2 < allowed_loss_of_energy):
            sum += eigen_values[-i - 1] ** 2
        else:
            number_of_rows_to_be_retained_in_VT = len(eigen_values) - i
            break

    new_VT = np.zeros((number_of_rows_to_be_retained_in_VT, len(VT[0])))

    for i in range(number_of_rows_to_be_retained_in_VT):
        for j in range(len(VT[i])):
            new_VT[i][j] = VT[temp[i][0]][j]

    return new_VT


# SVD function
def svd_func(A, B, user_offset, temp):
    A_transpose = A.T

    start_time = time.time()
    U, eigen_values, VT = getsvdparam(A, full_matrices=False)
    sigma = np.zeros((len(A_transpose), len(A_transpose)))

    for i in range(len(eigen_values)):
        sigma[i][i] = math.sqrt(eigen_values[i])

    temp_time =  time.time() -start_time
    mae, rmse = predict(A, B, VT, user_offset, temp)
    print("########################\nSVD with 100% retention:")
    print("RMSE for SVD: " + str(rmse))
    print("MAE: " + str(mae))
    print("Execution Time : " + str(time.time() - start_time))


    start_time = time.time()
    VT = get_new_VT(VT, eigen_values)

    mae2, rmse2 = predict(A, B, VT, user_offset, temp)  # Here VT is the new VT after 90% retained energy
    print("########################\nSVD with 90% retention:")
    print("RMSE : " + str(rmse2))
    print("MAE : " + str(mae2))
    print("Execution Time : " + str(time.time() - start_time + temp_time))
    return
