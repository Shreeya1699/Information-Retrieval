import numpy as np
from numpy.linalg import svd
import math
import operator
import time
import recommender

def collaborative_filtering_func(AT, BT, no_of_neighbors, movies_rated_by_user, to_be_predicted, temp,baseline_approach):
    avg_movie_rating = np.zeros(len(AT))  # 1684 movies
    total_rating = 0.0
    number_of_ratings = 0

    # Finding mean movie rating throughout matrix
    for i in range(len(AT)):
        for j in range(len(AT[i])):
            if (AT[i][j] != 0):
                total_rating += AT[i][j]
                number_of_ratings += 1
    mean_movie_rating = float(total_rating) / number_of_ratings

    rating_deviation_of_user = np.zeros(len(AT[0]))
    rating_deviation_of_movie = np.zeros(len(AT))

    # Rating deviation of each user
    # Given by: Average rating of user - mean movie rating
    for j in range(len(AT[0])):
        num = 0.0
        number_of_movies_rated = 0
        for i in range(len(AT)):
            if (AT[i][j] != 0):
                num += AT[i][j]
                number_of_movies_rated += 1
        if (number_of_movies_rated > 0):
            rating_deviation_of_user[j] = (float(num) / number_of_movies_rated) - mean_movie_rating

    # Rating deviation of each movie
    # Given by: Average rating of movie - mean movie rating
    # Normalizing rows of AT here
    for i in range(len(AT)):
        num = 0.0
        number_of_users_rated = 0
        for j in range(len(AT[i])):
            if (AT[i][j] != 0):
                num += AT[i][j]
                number_of_users_rated += 1
        if (number_of_users_rated > 0):
            rating_deviation_of_movie[j] = (float(num) / number_of_users_rated) - mean_movie_rating
            avg_movie_rating[i] = float(num / float(number_of_users_rated))
        for j in range(len(AT[i])):
            if (AT[i][j] != 0):
                AT[i][j] = AT[i][j] - avg_movie_rating[i]

    number_of_predictions = 0
    sum_of_squared_error = 0.0
    error_sum = 0.0
    count = 0

    # Predicting the ratings here
    for data in to_be_predicted:
        # data is of the form [movie, user]

        count += 1
        sim = []
        for movie in movies_rated_by_user[data[1]]:
            sim.append([movie, recommender.find_similarity(AT[data[0]], AT[movie])])
        sorted_sim = sorted(sim, key=operator.itemgetter(1), reverse=True)
        numerator = 0
        denomenator = 0
        for l, i in zip(range(no_of_neighbors), range(len(sorted_sim))):
            if (baseline_approach == True):
                numerator += sorted_sim[l][1] * (BT[sorted_sim[l][0]][data[1]] - (
                            mean_movie_rating + rating_deviation_of_user[data[1]] + rating_deviation_of_movie[
                        sorted_sim[l][0]]))
            else:
                numerator += sorted_sim[l][1] * AT[sorted_sim[l][0]][data[1]]
            denomenator += sorted_sim[l][1]
        if (denomenator > 0):
            if (baseline_approach == True):
                rating = mean_movie_rating + rating_deviation_of_user[data[1]] + rating_deviation_of_movie[data[0]] + (
                            numerator / float(denomenator))
            else:
                rating = (numerator / float(denomenator)) + avg_movie_rating[data[0]]
            sum_of_squared_error += (rating - BT[data[0]][data[1]]) ** 2
            error_sum += abs(rating - BT[data[0]][data[1]])
            temp[data[1]][data[0]] = rating
            number_of_predictions += 1

    # Root mean square
    rmse = math.sqrt(sum_of_squared_error/ number_of_predictions)
    mae = error_sum / number_of_predictions

    # Printing the results
    if (baseline_approach):
        print("########################\nCollaborative Filtering with Baseline Approach")
        print("RMSE : " + str(rmse))
        print("MAE : " + str(mae))
    else:
        print("########################\nCollaborative Filtering")
        print("RMSE : " + str(rmse))
        print("MAE : " + str(mae))

    return