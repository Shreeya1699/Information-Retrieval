import nltk
import os
import pickle
from nltk.stem.porter import *
import math
import re, collections
import pyttsx3 as pyttsx


lengths = {}

""" 
Implemented Stemmer for Stemming the Query
"""
stemmer = PorterStemmer()

#Loading all the data structures created before
dict = pickle.load(open("dictionary.p","rb"))
invertedIndex = pickle.load(open("inverted_ind.p","rb"))
tf_idf = pickle.load(open("tf_idf.p","rb"))


N = len(dict)

"""
Finding the length of each vector(doc represented as a vector)
"""
for key in tf_idf:
	temp = 0.0
	for word in tf_idf[key]:
		temp = temp + tf_idf[key][word] * tf_idf[key][word]
	lengths[key] = math.sqrt(temp)

"""
Function to implement page ranking
"""
def Page_Ranking_Algo(query):
	Query_Dictionary = {}
	Query_List = []

	for word in query.split():  #Representing query as a vector
		word=word.lower()
		word = stemmer.stem(word)
		if word in Query_Dictionary:
			k=Query_Dictionary[word]
			Query_Dictionary[word] = k+1
		else:
			Query_Dictionary[word] = 1

	for key in Query_Dictionary:
		Query_List.append(key)
		print(key)

	score = {}
	# print len(dict)

	#Calculating the cosine similarity of the query vector with the docs
	for word in Query_List:
		weight_q = 0
		if word in invertedIndex:
			df = len(invertedIndex[word])
			idf = math.log( N/( df * 1.0 ), 10.0 )
			weight_q = idf * ( 1.0 + math.log( Query_Dictionary[word] , 10.0))

			for doc in invertedIndex[word]:
				if doc in score:
					temp = score[doc]
					weight_d = tf_idf[doc][word]
					score[doc] = temp + weight_q * weight_d
				else:
					weight_d = tf_idf[doc][word]
					score[doc] = weight_q * weight_d

	rank = []
	#Length Normalization of the cosine similarity
	for key in score:
		score[key] = score[key]/(1.0 * lengths[key])
		rank.append((key, score[key]))
		#print key, score[key], lengths[key]

	#Sorted(ranking,key=itemgetter(1))
	rank = sorted(rank , key=lambda x: x[1], reverse = True)  #sorting all the docs on the basis of their cosine similarity

	print(rank[:10])
	print("*************************************************************************************")
	text = '\n'.join(chunk[0] for chunk in rank[:min(len(rank),10)])  #Returning the top 10 search results
	return text

""" 
Show results function to show the output of query
 """
def Show_Results():
	key = input("Give the search Query")            # Enter the query in command prompt
	text=Page_Ranking_Algo(key)
	print(text.split())

Show_Results()