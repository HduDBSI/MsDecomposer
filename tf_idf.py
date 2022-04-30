import math
from functools import reduce


def handle_tf(word, vocabulary=[]):
    return len(list(filter(lambda w: w == word, vocabulary))) / len(vocabulary)


def handle_idf(word, docs=[]):
    contain_count = len(list(filter(lambda doc: len(list(filter(lambda w: w == word, doc))) != 0, docs)))
    return math.log(len(docs) / contain_count + 1)


def handle_tf_idf(tf, idf, ls):
    return tf * idf * ls


# 计算相似度
def handle_cosine_similarity(frequency1, frequency2):
    # 分子
    molecular = 0
    for f1 in frequency1:
        for f2 in frequency2:
            if f1[0] == f2[0]:
                molecular += f1[1] * f2[1]

    # 分母
    denominator = sqrt_and_square(frequency1) * sqrt_and_square(frequency2)
    return molecular / denominator


def sqrt_and_square(frequency):
    return math.sqrt(reduce(lambda a, b: a + b[1] ** 2, frequency, 0))
