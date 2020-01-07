
Movie-Recommender-System

This contains code for implementation of Collaborative filtering, SVD, CUR.

The packages used in the execution of the program are:
* numpy
* math
* operator
* time

*The code is divided into 9 functions as follows:*
1. **collaborative_filtering_func**: Function that carries out collaborative filtering
algorithm
1. **find_similarity**: It gives the similarity between two vectors
1. **svd_func**: Function to initiate the SVD algorithm
1. **get_new_VT**: This function shrinks the size of VT when 90% energy is being
maintained in SVD algorithm
1. **predict**: This function is called in svd_func function. It carries out the task of
predicting the testing data
1. **cur_func**: This function initiates the CUR algorithm
1. **select_random_rows**: This function is used to select random rows from a
given matrix where each row can have a probability of being selected.
1. **find_U_and_rmse**: This function finds the U matrix and also calculates
RMSE,MAE



