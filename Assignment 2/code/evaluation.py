"""
evaluation.py
this code calculates the precision and recall of the lsh algorithm implemented and stores it in a file called
"""
from __future__ import division
from datasketch import MinHash
from nltk import ngrams
import utils
import os
import time

def read_file(in_file, mode):
    """
    Reads file from the directory
    """
    if mode == 'char':
        filename, text = utils.read_file(in_file)
    elif mode == 'word':
        filename, text = utils.tokenize_file(in_file)
    return filename, text

def calculate_lsh(text, mode, lsh_type):
    """
    Calculate lsh for the file

    """
    min_hash = MinHash(num_perm=128)
    if mode == 'char':
        for d in ngrams(text, 3):
            min_hash.update("".join(d).encode('utf-8'))
    elif mode == 'word':
        for d in ngrams(text, 3):
            min_hash.update(" ".join(d).encode('utf-8'))
    result = lsh_type.query(min_hash)
    return result


def evaluate_true_negatives(input_dir, mode, lsh_type):
    """
    Calculate LSHs for the files that do not contain plagiarism (tn_docs folder)
    Arguments:
        input_dir (str): directory to the files that do not contain plagiarism
                        (true_positives folder)
        mode (str): type of ngrams to use ('char', 'word')
        lsh_type (MinHash object): previously calculated lsh
    Returns:
        tn: number of true negatives - files
            correctly marked as not plagiarized
        fp: number of false positives - files
            incorrectly marked as plagiarized
        writes names of the files to 'tn_log.txt'
        writes names of the files to 'fp.txt'
    """
    true_neg = false_pos = 0
    for f in os.listdir(input_dir):
        filename, text = read_file(os.path.join(input_dir, f), mode)
        result = calculate_lsh(text, mode, lsh_type)

        if len(result) > 0:
            false_pos += 1
            print("fp: ", filename, result)
            with open('false_pos_log.txt', 'a+') as false:
                false.write('{0}\t{1}\n'.format(filename, result))
        else:
            true_neg += 1
            print('true_negative: ', filename, result)
            with open('true_neg_log.txt', 'a+') as tn_log:
                tn_log.write('{0}\n'.format(filename))
    return true_neg, false_pos


def evaluate_true_positives(input_dir, mode, lsh_type):
    """Calculate LSHs for the files that contain plagiarism (true_positives folder)
    Args:
        input_dir (str): directory to the files that contain plagiarism
                        (true_positives folder)
        mode (str): type of ngrams to use ('char', 'word')
        lsh_type (MinHash object): previously calculated lsh
    Returns:
        true_pos: number of true positives - number of files
            correctly marked as plagiarized
        false_neg: number of false negatives - number of files
            incorrectly marked as not plagiarized
        writes names of the files to 'true_pos_log.txt'
        writes names of the files to 'false_neg.txt'
    """
    true_pos = false_neg = 0
    for f in os.listdir(input_dir):
        filename, text = read_file(os.path.join(input_dir, f), mode)
        result = calculate_lsh(text, mode, lsh_type)
        if len(result) == 0:
            false_neg += 1
            print("false negative: ", filename, result)
            with open('false_neg_log.txt', 'a+') as fn_log:
                fn_log.write('{0}\n'.format(filename))
        else:
            true_pos += 1
            print("true positives: ", filename, result)
            with open('true_pos_log.txt', 'a+') as tp_log:
                tp_log.write('{0}\t{1}\n'.format(filename, result))
    return true_pos, false_neg


def calculate_metrics(tn_directory, tp_directory, mode , lsh_type):
    """
    Calculate the results of the LSH algorithms performance
    """
    tp, fn = evaluate_true_positives(tp_directory, mode, lsh_type)
    tn, fp = evaluate_true_negatives(tn_directory, mode, lsh_type)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1, tp, tn, fp, fn


if __name__ == '__main__':
    start_time=time.time()
    tn_directory = 'data/true_negatives'
    tp_directory = 'data/true_positives/'
    lsh_word_03 = utils.load_lsh(utils.WORD_03_FILE)
    precision, recall, f1, tp, tn, fp, fn = calculate_metrics(tn_directory, tp_directory, 'word', lsh_word_03)
    print('lsh_word_03 evaluation:\nPrecision: {0}\nRecall: {1}\nF1 measure: {2}\nTPs: {3}\nTNs: {4}\nFPs: {5}\nFNs: ' \
          '{6}'.format(precision, recall, f1, tp, tn, fp, fn))
    with open('word_03_eval_results.txt', 'w+') as out_file:
        out_file.write('Precision: {0}\nRecall: {1}\nF1 measure: {2}\nTPs: {3}\nTNs: {4}\nFPs: {5}\nFNs: ' \
                       '{6}'.format(precision, recall, f1, tp, tn, fp, fn))

    print("--- %s seconds ---" % (time.time() - start_time))