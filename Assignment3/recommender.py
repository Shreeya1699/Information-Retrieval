import numpy as np
import math
import time
import collaborative
import svd_1
import cur_1
import latent

# Similarity function
def find_similarity(X, Y):
	numerator = 0.0
	sum_of_square_of_components_of_X = 0.0
	sum_of_square_of_components_of_Y = 0.0
	
	for i in range(len(X)):
		numerator += X[i] * Y[i]
		sum_of_square_of_components_of_X += X[i] ** 2
		sum_of_square_of_components_of_Y += Y[i] ** 2

	denomenator = math.sqrt(sum_of_square_of_components_of_X) * math.sqrt(sum_of_square_of_components_of_Y)
	if(denomenator == 0):
		return 0
	else:
		return float(numerator) / denomenator



user_ids_index = {}
movie_ids_index = {}
user_count = 0
movie_count = 0
count = 0
max_user_no = 0
max_movie_no = 0
movies_rated_by_user = {}
to_be_predicted = []
k = 50
r = 300

# Reading file for finding max movie id and max user id
with open("movies.data", "r") as data_file:
	for line in data_file:
		count += 1
		line_values = line.split("\t")
		a = int(line_values[0])
		b = int(line_values[1])
		if(a > max_user_no):
			max_user_no = a
		if(b > max_movie_no):
			max_movie_no = b

three_fourth_data_length = int(0.75 * count)
counter = 0
count_thousand_data_points = 0
A = np.zeros((max_user_no + 1, max_movie_no + 1))
temper = np.zeros((max_user_no + 1, max_movie_no + 1))
B = np.zeros((max_user_no + 1, max_movie_no + 1))


# Reading file
with open("movies.data", "r") as f:
	for line in f:
		tokens = line.split("\t")
		a = int(tokens[0])
		b = int(tokens[1])
		B[a][b] = float(tokens[2])
		if(counter <= three_fourth_data_length):
			A[a][b] = float(tokens[2])
			temper[a][b] = float(tokens[2])
			counter += 1
			if a not in movies_rated_by_user:
				movies_rated_by_user[a] = [b]
			else:
				movies_rated_by_user[a].append(b)
		elif(count_thousand_data_points < 120):
			to_be_predicted.append([b, a])
			count_thousand_data_points += 1

f.close()


no_of_neighbors = 5
temp = temper.copy()
start_time = time.time()

# Calling Colloborative function without baseline approach
collaborative.collaborative_filtering_func(A.T, B.T, no_of_neighbors, movies_rated_by_user, to_be_predicted, temp, False)
print("Execution Time : " + str(time.time() - start_time))
start_time = time.time()

# Calling Colloborative function with baseline approach
collaborative.collaborative_filtering_func(A.T, B.T, no_of_neighbors, movies_rated_by_user, to_be_predicted, temp, True)
print("Execution Time : " + str(time.time() - start_time))
user_offset = np.zeros(max_user_no + 1)
#
# # Normalizing A matrix
for i in range(max_user_no + 1):
	num = 0.0
	no_of_movies_rated_by_current_user = 0
	for j in range(max_movie_no + 1):
		if (A[i][j] != 0):
			num += A[i][j]
			no_of_movies_rated_by_current_user += 1
	if(no_of_movies_rated_by_current_user > 0):
		user_offset[i] = float(num / float(no_of_movies_rated_by_current_user))
	for j in range(max_movie_no + 1):
		if(A[i][j] != 0):
			A[i][j] = A[i][j] - user_offset[i]

temp = temper.copy()

# Calling SVD function
svd_1.svd_func(A, B, user_offset, temp)

print("")

# Calling CUR function
cur_1.cur_func(B, r)
print("#############################\nLatent Factor Model")

start_time=time.time()
rate_test, rate_train, us, mo = latent.make_ratings_matrix()
mf = latent.MF(rate_train, 7, 0.0001, 0.00001, 20)
mf.train()
rate_pred = np.zeros((us, mo))
for i in range(rate_test.shape[0]):
    for j in range(rate_test.shape[1]):
        if rate_test[i][j] != 0:
            rate_pred[i][j] = mf.get_rating(i, j)
print("Execution Time: "+str(time.time() - start_time ))
# Program ends