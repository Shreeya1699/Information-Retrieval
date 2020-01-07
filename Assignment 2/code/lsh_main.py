
from datasketch import MinHash, MinHashLSH
import os
import pickle
import nltk
import utils
import time

"""
lsh_main.py
calculates the LSH for all the documents in the corpus
"""

def init_lshs(directory, type, threshold):
    """Initilizes and calculates LSH for the document corpus
    Args:
        directory (str): the directory with source files
        type (str): type of ngrams to use ('char', 'word')
        threshold (float): Jaccard threshold value
    Returns:
        lsh: datasketch object
    """
    # Create a MinHashLSH index using Redis as the storage layer
    lsh = MinHashLSH(threshold=threshold, num_perm=128,
                     storage_config={'type': 'redis', 'redis': {'host': 'localhost', 'port': 6379 },
                                     'name': "IR"})

    data_list = []

    for f in os.listdir(directory):
        minhash = MinHash(num_perm=128)
        if type == 'char':
            filename, text = utils.read_file(os.path.join(directory, f))
            print(filename)
            for d in nltk.ngrams(text, 3):
                minhash.update("".join(d).encode('utf8'))
        elif type == 'word':
            filename, text = utils.tokenize_file(os.path.join(directory, f))
            print(filename)
            for d in nltk.ngrams(text, 3):
                minhash.update(" ".join(d).encode('utf8'))

        data_list.append((filename, minhash))

    with lsh.insertion_session() as session:
        for key, minhash in data_list:
            session.insert(key, minhash)

    return lsh


def serialize(lsh_type, file):
    """
    Pickle lsh object
    """
    with open(file, 'wb') as handle:
        pickle.dump(lsh_type, handle)
    with open(file,'rb') as handle:
        print(pickle.load((handle)))


if __name__ == '__main__':
    start_time= time.time()
    lsh_word_05 = init_lshs('C:/Users/pratd/Documents/IRprojectMedicine/assign2/corpus-20090418', 'word', 0.5)
    serialize(lsh_word_05, utils.WORD_05_FILE)
    print("--- %s seconds ---" % (time.time() - start_time))