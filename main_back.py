import json
import nltk
from tf_idf import handle_tf, handle_idf, handle_tf_idf, handle_cosine_similarity
from nltk.corpus import stopwords
from nltk.stem.porter import *
from sklearn import metrics
from sklearn.cluster import SpectralClustering, spectral_clustering

import warnings

warnings.filterwarnings('ignore')


# from pattern.text.en import singularize
# nltk.download('stopwords')


# 返回 [{"url": url, "keys": url.split(" "), "words": words.split(" ")}]
def get_blog_apis():
    with open("./Blog.json") as blogFile:
        blog = json.load(blogFile)

    apis = blog["paths"]
    corpus = []
    for k, v in apis.items():
        for k1, v1 in v.items():
            keywords = list(filter(lambda w: w != "", k.split("/")))
            words = " ".join(keywords) + " " + v1["description"]
            corpus.append({"url": k1 + " " + k, "keys": filter_params(keywords), "words": words.split(" ")})
    return corpus


# 过滤参数
def filter_params(words):
    return list(filter(lambda w: not w.startswith("{"), words))


# 数据清洗
def data_clean(apis):
    new_apis = []
    for api in apis:
        words = []
        for word in api["words"]:
            # 移除 http 操作
            is_http_operation(word) or words.append(word)
        # 过滤 缺乏实际意义的 the, a, and 等词
        no_stop_words = filter_stopwords(words)
        # 过滤 缺乏实际意义的 the, a, and 等词
        stem_words = filter_stem_words(words)
        new_apis.append({"url": api["url"], "keys": filter_stem_words(api["keys"]), "words": stem_words})
    return new_apis


# 判断是不是 http 操作
def is_http_operation(word):
    return len(list(filter(lambda operation: operation == word, ["get", "put", "post", "update", "delete"]))) != 0


# 过滤相似单次
def filter_stem_words(tokens):
    stemmed = []
    for item in tokens:
        stemmed.append(PorterStemmer().stem(item))
    return stemmed


# 过滤 缺乏实际意义的 the, a, and 等词
def filter_stopwords(tokens):
    return [w for w in tokens if not w in stopwords.words()]


# 过滤重复单词
def filter_repeat_words(tokens):
    return list(set(tokens))


# 计算TF-IDF的值
def handle_api_tfidf(apis):
    docs = list(map(lambda api: api["words"], apis))

    result = []
    for api in apis:
        api_tfidf = []
        for key in api["keys"]:
            tf = handle_tf(key, api["words"])
            idf = handle_idf(key, docs)
            api_tfidf.append((key, handle_tf_idf(tf, idf)))
        result.append({"url": api["url"], "frequency": api_tfidf})
    return result


# 主题主题相似度
def handle_url_cosine_similarity(frequency):
    result = []
    for curURL in frequency:
        for nextURL in frequency:
            similarity = handle_cosine_similarity(curURL["frequency"], nextURL["frequency"])
            result.append({"source": curURL["url"], "target": nextURL["url"], "similarity": similarity})
    # for i in range(len(frequency) - 1):
    #     curURL = frequency[i]
    #     for nextURL in frequency[i + 1:]:
    #         # print((curURL, nextURL))
    #         similarity = handle_cosine_similarity(curURL["frequency"], nextURL["frequency"])
    #         result.append({"source": curURL["url"], "target": nextURL["url"], "similarity": similarity})
    return result


def print_line(list):
    for item in list:
        print(item)


# 获取邻接矩阵
def set_adjacency_matrix(similarities, nums):
    adj_mat = [[]]
    i = 0
    for similarity in similarities:
        if i == nums:
            adj_mat.append([])
            i = 0
        adj_mat[len(adj_mat) - 1].append(similarity["similarity"])
        i += 1
    return adj_mat


def main():
    apis = data_clean(get_blog_apis())
    api_frequency = handle_api_tfidf(apis)
    api_similarity = handle_url_cosine_similarity(api_frequency)
    api_similarity_adj_mat = set_adjacency_matrix(api_similarity, len(apis))

    recommend_microservices = []
    for index, k in enumerate((3, 4, 5, 6, 7, 8)):
        sc = SpectralClustering(n_clusters=k, affinity='precomputed')
        sc.fit(api_similarity_adj_mat)
        # metrics.calinski_harabaz_score(api_similarity_adj_mat, sc.labels_) 是用来计算当前的划分出来的图的分值
        # 计算 Calinski 和 Harabaz 得分。方差比标准。被定义为群内分散和群集间分散之间的比率。
        recommend_microservices.append((metrics.calinski_harabaz_score(api_similarity_adj_mat, sc.labels_), sc.labels_))

    # 找出最适合的微服务
    best = (0, 0)
    for microservice in recommend_microservices:
        if best[0] < microservice[0]:
            best = microservice

    print(best)


main()

#
# corpus = get_api_description()
# vectorizer = TfidfVectorizer()
# X = vectorizer.fit_transform(corpus)
# print(vectorizer.get_feature_names())
#
# print(X)
# print(cosine_similarity())
