import nltk

from tf_idf import handle_tf, handle_idf, handle_tf_idf, handle_cosine_similarity
from nltk.corpus import stopwords
from nltk.stem.porter import *
from functools import reduce


# from pattern.text.en import singularize
nltk.download('stopwords')

# 计算主题相似度similarity
def handle_topic_similarity(apis):
    cleaned_apis = data_clean(apis)
    api_frequency = handle_api_tfidf(cleaned_apis)
    return handle_url_cosine_similarity(api_frequency)


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


# 过滤相似单词
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


def filter_redundant_keys(keys):
    return 0


def get_all_keys(apis):
    return list(reduce(lambda keys, api: keys + list(api["keys"]), apis, []))


def get_all_docs(apis):
    return list(map(lambda api: api["words"], apis))


def is_meaningless(key, docs, keys):
    count = reduce(lambda n, item: n + (1 if item == key else 0), keys, 0)
    # return count == len(docs) or count == len(docs) - 1 or count == 1
    return count == len(docs)


# 计算TF-IDF的值
def handle_api_tfidf(apis):
    docs = get_all_docs(apis)
    keys = get_all_keys(apis)
    result = []
    # 根据位置加权重版本
    # for api in apis:
    #     api_tfidf = []
    #     api_keys = list(filter(lambda item: not is_meaningless(item, docs, keys), api["keys"]))
    #     # api_keys = api["keys"]
    #     for index, key in enumerate(api_keys):
    #         tf = handle_tf(key, api["words"])
    #         idf = handle_idf(key, docs)
    #         ls = 1 + (len(api_keys) - index) * (1 / len(api_keys))
    #         api_tfidf.append((key, handle_tf_idf(tf, idf, ls)))
    #     result.append({"url": api["url"], "frequency": api_tfidf})

    # 不加权重版本
    for api in apis:
        api_tfidf = []
        for key in api["keys"]:
            tf = handle_tf(key, api["words"])
            idf = handle_idf(key, docs)
            api_tfidf.append((key, handle_tf_idf(tf, idf, 1)))
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
