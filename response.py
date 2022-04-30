VERB = {
    'REF': "$ref",
    'NORMAL': "normal"
}


def calc_response_similarity(apis, api_definitions):
    api_responses = data_clean(apis)
    result = []
    for curURL in api_responses:
        for nextURL in api_responses:
            # if curURL != nextURL:
            similarity = handle_verb(curURL, nextURL, api_definitions)
            result.append({"source": curURL["url"], "target": nextURL["url"], "similarity": similarity})
    return result


def data_clean(apis):
    api_responses = []
    for api in apis:
        if api['verb'] != "delete":
            schema = api['responses']['200']['schema']
            res_type = VERB["REF"] if has_key(schema, '$ref') else VERB["NORMAL"]
            api_responses.append({"url": api["url"], "verb": api['verb'], "type": res_type, "schema": schema})
        else:
            api_responses.append({"url": api["url"], "verb": api['verb'], "type": VERB["NORMAL"], "schema": {}})
    return api_responses


def handle_type(api1, api2, api_definitions):
    if api1["type"] == VERB["REF"] and api2["type"] == VERB["REF"]:
        return 1.0 if get_ref_last_resource_name(api1["schema"]["$ref"], api_definitions) == get_ref_last_resource_name(
            api2["schema"]["$ref"], api_definitions) else 0.0
    elif api1["type"] != VERB["REF"] and api2["type"] != VERB["REF"]:
        return handle_jaccard(get_normal_resource(api1["schema"]), get_normal_resource(api2["schema"]))
    else:
        if api1["type"] == VERB["REF"]:
            return handle_jaccard(get_ref_last_resource(api1["schema"]["$ref"], api_definitions),
                                  get_normal_resource(api2["schema"]))
        else:
            return handle_jaccard(get_normal_resource(api1["schema"]),
                                  get_ref_last_resource(api2["schema"]["$ref"], api_definitions))


def handle_verb(api1, api2, api_definitions):
    if api1["verb"] == 'delete' or api2["verb"] == 'delete':
        return 0.0
    else:
        return handle_type(api1, api2, api_definitions)


# Jaccard 相似度算法计算
def handle_jaccard(words1, words2):
    intersection = [i for i in words1 if i in words2]
    union = list(set(words1 + words2))
    return len(intersection) / len(union)


def has_key(o, k):
    return o.__contains__(k)


# 获取 ref 最终引向的资源的名称
def get_ref_last_resource_name(definition, api_definitions):
    resource_name = get_ref_name(definition)
    print(resource_name)
    if api_definitions[resource_name]["type"] == "array":
        return get_ref_last_resource_name(api_definitions[resource_name]["items"]["$ref"], api_definitions)
    else:
        return resource_name


# 获取 ref 最终引向的资源的属性
def get_ref_last_resource(definition, api_definitions):
    resource_name = get_ref_name(definition)
    if api_definitions[resource_name]["type"] == "array":
        return get_ref_last_resource(api_definitions[resource_name]["items"]["$ref"], api_definitions)
    else:
        return list(api_definitions[resource_name]['properties'].keys())


def get_normal_resource(schema):
    if schema["type"] == "array":
        return list(schema["items"]["properties"].keys())
    else:
        return list(schema["properties"].keys())


# 获取 definition 资源的名称
def get_ref_name(definition):
    return definition.split('/').pop()


if __name__ == '__main__':
    words1 = ['id', 'title', 'username', 'createBy', 'udateBy', 'createDate', 'udateDate']
    words2 = ['id', 'title', 'username', 'createBy', 'udateBy', 'create', 'udate']
