import numpy as np
from numpy import linalg as la
from numpy.linalg import svd
import math
import operator
import time
import recommender
# Function to select random rows for CUR
def select_random_rows(B, r, isRepeatationAllowed):
    indices = [i for i in range(len(B))]
    square_of_frobenius_norm_of_B = 0
    for i in range(len(B)):
        for j in range(len(B[i])):
            square_of_frobenius_norm_of_B += B[i][j] ** 2

    p = np.zeros(len(B))
    for i in range(len(B)):
        sum_of_squared_values_in_row = 0
        for j in range(len(B[i])):
            sum_of_squared_values_in_row += B[i][j] ** 2
        p[i] = sum_of_squared_values_in_row / float(square_of_frobenius_norm_of_B)

    rows_selected = np.random.choice(indices, r, isRepeatationAllowed, p)

    R = np.zeros((r, len(B[0])))
    for i, row in zip(range(r), rows_selected):
        for j in range(len(B[row])):
            R[i][j] = B[row][j]
            R[i][j] = R[i][j] / float(math.sqrt(r * p[row]))

    return rows_selected, R

def get_new_sigma(sigma, X,YT,eigen_values):

    total_sum = 0
    dimensions = sigma.shape[0]
    i=0
    for i in range(dimensions):
        total_sum = total_sum + eigen_values[i]**2  # Find square of sum of all diagonals
    retained = total_sum
    while dimensions > 0:
        retained = retained - np.square(sigma[dimensions - 1, dimensions - 1])
        if retained / total_sum < 0.9:  # 90% energy retention
            break
        else:
            X = X[:, :-1:]
            YT = YT[:-1, :]
            sigma = sigma[:, :-1]
            sigma = sigma[:-1, :]
            dimensions = dimensions - 1  # Dimensionality reduction

    for i in range(sigma.shape[0]):
        sigma[i][i] = 1 / sigma[i][i]

    U = np.dot(np.dot(YT.T, sigma ** 2), X.T)
    return U

def find_Unew_and_rmse(B, r, row_indices, R, column_indices, C):
    print("########################\nCUR with 90% retention:")
    W = np.zeros((r, r))
    for i, row in zip(range(len(row_indices)), row_indices):
        for j, column in zip(range(len(column_indices)), column_indices):
            W[i][j] = B[row][column]

    X, eigen_values, YT = svd(W, full_matrices=False)

    sigma = np.zeros((r, r))
    sigma_plus = np.zeros((r, r))

    for i in range(len(eigen_values)):
        sigma[i][i] = math.sqrt(eigen_values[i])
        if (sigma[i][i] != 0):
            sigma_plus[i][i] = 1 / float(sigma[i][i])

    Unew = get_new_sigma(sigma, X, YT, eigen_values)
    new_cur_matrix = np.dot(np.dot(C, Unew), R)
    number_of_predictions = 0
    squared_error_new = 0
    sum_error_new = 0

    for i in range(len(B)):
        for j in range(len(B[i])):
            if (B[i][j] != 0):
                squared_error_new += (B[i][j] - new_cur_matrix[i][j]) ** 2
                sum_error_new += abs(B[i][j] - new_cur_matrix[i][j])
                number_of_predictions += 1

    # print(number_of_predictions)
    frobenius_norm2 = math.sqrt(squared_error_new)

    # Root mean square
    rmse_new = frobenius_norm2 / float(number_of_predictions)
    # mean average error
    mae_new = sum_error_new / float(number_of_predictions)

    return number_of_predictions, rmse_new, mae_new

def find_U_and_rmse(B, r, row_indices, R, column_indices, C):
    print("########################\nCUR With 100 retention")
    W = np.zeros((r, r))
    for i, row in zip(range(len(row_indices)), row_indices):
        for j, column in zip(range(len(column_indices)), column_indices):
            W[i][j] = B[row][column]

    X, eigen_values, YT = svd(W, full_matrices=False)

    sigma = np.zeros((r, r))
    sigma_plus = np.zeros((r, r))

    for i in range(len(eigen_values)):
        sigma[i][i] = math.sqrt(eigen_values[i])
        if (sigma[i][i] != 0):
            sigma_plus[i][i] = 1 / float(sigma[i][i])
            # new_sigma_plus[i][i]=1/float(new_sig[i][i])

    U = np.dot(np.dot(YT.T, np.dot(sigma_plus, sigma_plus)), X.T)
    # print(U)

       # CUR matrix
    cur_matrix = np.dot(np.dot(C, U), R)
    squared_error_sum = 0
    number_of_predictions = 0
    sum_error=0


    for i in range(len(B)):
        for j in range(len(B[i])):
            if (B[i][j] != 0):
                squared_error_sum += (B[i][j] - cur_matrix[i][j]) ** 2
                sum_error += abs(B[i][j] - cur_matrix[i][j])
                number_of_predictions += 1

    frobenius_norm = math.sqrt(squared_error_sum)


    # Root mean square
    rmse = frobenius_norm / float(number_of_predictions)
    # mean average error
    mae= sum_error/float(number_of_predictions)

    return number_of_predictions, squared_error_sum, rmse,mae

# CUR function
def cur_func(B, r):
    # print("CUR function")
    start_time = time.time()
    row_indices, temp_matrix = select_random_rows(B, r, True)
    R = temp_matrix
    column_indices, temp_matrix = select_random_rows(B.T, r, True)
    C = temp_matrix.T
    temp_time = time.time() - start_time

    n, squared_error_sum, rmse ,mae= find_U_and_rmse(B, r, row_indices, R, column_indices, C)
    print("RMSE : " + str(rmse))
    print("MAE : "+ str(mae))
    print("Execution Time : " + str(time.time() - start_time))

    start_time = time.time()
    n2,rmse2,mae_new =find_Unew_and_rmse(B,r,row_indices,R,column_indices,C)
    print("RMSE: "+ str(rmse2))
    print("MAE : "+ str(mae_new))
    print("Execution Time:" + str(time.time() - start_time + temp_time))

    return