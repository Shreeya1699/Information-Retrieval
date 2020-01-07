"""
query.py
Check new document for plagiarism
"""
from datasketch import MinHash
import nltk
from nltk import ngrams
import utils
import textdistance
import time

lsh = utils.load_lsh(utils.WORD_05_FILE)
def check_dist(filename,result):
    """

    :param filename: (str) the path of query file
    :param result:(list) list contains the documents from corpus with similarity to query file
    :return: lists containing jaccard distances, hamming distances, edit distances and cosine distances with each document in the
                result with that of the corpus
    """
    hamming_dist=[]
    jaccard_dist=[]
    edit_dist=[]
    cosine_dist=[]
    file, content = utils.tokenize_file(filename)
    query_cont =set(content)

    for each_result in result:
        file1, content1 = utils.tokenize_file(each_result)
        doc_cont= set(content1)
        jac = nltk.jaccard_distance(query_cont, doc_cont)
        edit_dis =nltk.edit_distance(content,content1)
        hamming = textdistance.hamming(content, content1)
        cos=textdistance.cosine(content, content1)
        cosine_dist.append(cos)
        jaccard_dist.append(jac)
        edit_dist.append(edit_dis)
        hamming_dist.append(hamming)

    return jaccard_dist, edit_dist, hamming_dist,cosine_dist



def check_new_lsh(filename):
    """
    Read input file, calculate its LSH and query it in the LSH database
    Returns:
        empty list [] - the file is not plagiarized (no LSH collision)
        list [source_filenames] - list with file names of sources of plagiarism
    """
    filename, text = utils.tokenize_file(filename)
    min_hash = MinHash(num_perm=128)

    for d in ngrams(text, 3):
        min_hash.update(" ".join(d).encode('utf8'))

    return lsh.query(min_hash)


if __name__ == '__main__':
    start_time = time.time()
    check_file = 'queryfile'
    result = check_new_lsh(check_file)
    print(result)
    if len(result) == 0:
        print("The document is not plagiarized")
    else:

        print("The document is plagiarized.\n" \
              "The files used are:\n{}".format(result))
        print(len(result))
        jac, ed, ham,cos= check_dist(check_file,result)
        print("The jaccard distance with each document from which the query documnet is plagiarized: ",jac)
        print("The jaccard edit distance with each document from which the query documnet is plagiarized: ",ed)
        print("The hamming distance with each document from which the query documnet is plagiarized: ",ham)
        print("The cosine distance with each document from which the query documnet is plagiarized: ",cos)

    print("--- ",(time.time() - start_time)," seconds ---")


