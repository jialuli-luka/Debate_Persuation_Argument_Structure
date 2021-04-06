import json
import pickle
import numpy as np
from user_features import *

def calculate_unigram(debate,X,smoothing):
    unigram = np.zeros((len(X),5))

    unigram_list = ['policy','value','fact','testimony','reference']

    j = 0
    for round in debate['rounds']:
        for side in round:
            if side['side'] == 'Pro':
                tmp = 0
            else:
                tmp = 1
            attribute = side['nodes_cdcp_svm_strict'].split(',')
            sum = 0
            for i in range(len(attribute)):
                key = attribute[i]
                sum = sum + 1
                unigram[tmp+j,unigram_list.index(key)] = unigram[tmp+j,unigram_list.index(key)] + 1
            if smoothing:
                for k in range(5):
                    unigram[tmp+j,k] = unigram[tmp+j,k] + 1
                unigram[tmp+j,:] = unigram[tmp+j,:] / (sum + 5)
            else:
                if sum != 0:
                    unigram[tmp+j,:] = unigram[tmp+j,:] / sum
        j = j + 2

    return unigram


def calculate_bigram(debate,X,smoothing):
    bigram = np.zeros((len(X),8))

    bigram_list = ['value+value', 'testimony+value', 'value+testimony', 'value+policy', 'policy+value', 'fact+value', 'value+fact', 'testimony+testimony']
    # bigram_list = ['testimony+value', 'value+testimony', 'value+policy', 'policy+value', 'fact+value', 'value+fact']

    j = 0
    for round in debate['rounds']:
        for side in round:
            if side['side'] == 'Pro':
                tmp = 0
            else:
                tmp = 1
            attribute = side['nodes_cdcp_svm_strict'].split(',')
            sum = 0
            for i in range(len(attribute) - 1):
                key = attribute[i] + "+" + attribute[i + 1]
                if key in bigram_list:
                    sum = sum + 1
                    bigram[tmp+j,bigram_list.index(key)] = bigram[tmp+j,bigram_list.index(key)] + 1
            if smoothing:
                for k in range(8):
                    bigram[tmp+j,k] = bigram[tmp+j,k] + 1
                bigram[tmp+j,:] = bigram[tmp+j,:] / (sum + 8)
            else:
                if sum != 0:
                    bigram[tmp+j,:] = bigram[tmp+j,:] / sum
        j = j + 2

    return bigram


def calculate_trigram(debate,X,smoothing):
    trigram = np.zeros((len(X),10))

    trigram_list = ['value+value+value','testimony+value+value','value+value+policy','value+value+testimony', 'value+testimony+value', 'fact+value+value', 'policy+value+value', 'value+fact+value', 'value+policy+value', 'value+value+fact']

    j = 0
    for round in debate['rounds']:
        for side in round:
            if side['side'] == 'Pro':
                tmp = 0
            else:
                tmp = 1
            attribute = side['nodes_cdcp_svm_strict'].split(',')
            sum = 0
            for i in range(len(attribute)-4,len(attribute) - 2):
                key = attribute[i] + "+" + attribute[i+1] + "+" + attribute[i+2]
                if key in trigram_list:
                    sum = sum + 1
                    trigram[tmp+j,trigram_list.index(key)] = trigram[tmp+j,trigram_list.index(key)] + 1
            if smoothing:
                for k in range(10):
                    trigram[tmp+j,k] = trigram[tmp+j,k] + 1
                trigram[tmp+j,:] = trigram[tmp+j,:] / (sum + 10)
            else:
                if sum != 0:
                    trigram[tmp+j,:] = trigram[tmp+j,:] / sum
        j = j + 2

    return trigram


def generate_graph_representation(debate,X):
    graph = np.zeros((len(X),5))   # basic, divergent_2, divergent++, convergent_2, convergent++

    j = 0
    for round in debate['rounds']:
        for side in round:
            structure_s = {}
            structure_t = {}
            if side['side'] == 'Pro':
                tmp = 0
            else:
                tmp = 1
            links = [k for k in range(len(side['links_cdcp_svm_strict'].split(','))) if side['links_cdcp_svm_strict'].split(',')[k] == 'True']
            sent = len(side['nodes_cdcp_svm_strict'].split(','))
            sum = 0
            for num in links:
                source = num // (sent - 1)
                target = num % (sent - 1)
                if target >= source:
                    target = target + 1
                if source in structure_s:
                    structure_s[source].append(target)
                else:
                    structure_s[source] = [target]
                if target in structure_t:
                    structure_t[target].append(source)
                else:
                    structure_t[target] = [source]
            for k, v in structure_s.items():
                if len(v) == 1 and len(structure_t[v[0]]) == 1:
                    graph[tmp+j, 0] = graph[tmp+j, 0] + 1
                    sum = sum + 1
                elif len(v) == 2:
                    graph[tmp+j, 1] = graph[tmp+j, 1] + 1
                    sum = sum + 1
                elif len(v) > 2:
                    graph[tmp+j, 2] = graph[tmp+j, 2] + 1
                    sum = sum + 1
            for k, v in structure_t.items():
                if len(v) == 2:
                    graph[tmp+j, 3] = graph[tmp+j, 3] + 1
                    sum = sum + 1
                elif len(v) > 2:
                    graph[tmp+j, 4] = graph[tmp+j, 4] + 1
                    sum = sum + 1
            if sum != 0:
                graph[tmp+j,:] = graph[tmp+j,:] / sum

        j = j + 2

    return graph


def calculate_density(debate,X):
    density = np.zeros((len(X),1))

    j = 0
    for round in debate['rounds']:
        for side in round:
            if side['side'] == 'Pro':
                tmp = 0
            else:
                tmp = 1
            links = [k for k in range(len(side['links_cdcp_svm_strict'].split(','))) if side['links_cdcp_svm_strict'].split(',')[k] == 'True']
            sent = len(side['nodes_cdcp_svm_strict'].split(','))
            nodes = []
            for num in links:
                source = num // (sent - 1)
                target = num % (sent - 1)
                if target >= source:
                    target = target + 1
                if target not in nodes:
                    nodes.append(target)
            density[tmp+j,0] = len(nodes) / sent
        j = j + 2

    return density


def calculate_argument_bigram(debate,X):
    bigram = np.zeros((len(X),4))

    bigram_list = ['value+value','value+policy','fact+value','testimony+value']

    j = 0
    for round in debate['rounds']:
        for side in round:
            structure_s = {}
            if side['side'] == 'Pro':
                tmp = 0
            else:
                tmp = 1
            links = [i for i in range(len(side['links_cdcp_svm_strict'].split(','))) if side['links_cdcp_svm_strict'].split(',')[i] == 'True']
            sent = len(side['nodes_cdcp_svm_strict'].split(','))
            attribute = side['nodes_cdcp_svm_strict'].split(',')
            sum = 0
            for num in links:
                source = num // (sent - 1)
                target = num % (sent - 1)
                if target >= source:
                    target = target + 1
                if source in structure_s:
                    structure_s[source].append(target)
                else:
                    structure_s[source] = [target]
            for k, v in structure_s.items():
                for u in v:
                    if k < sent and u < sent:
                        key = attribute[k] + "+" + attribute[u]
                        if key in bigram_list:
                            bigram[tmp+j,bigram_list.index(key)] = bigram[tmp+j,bigram_list.index(key)] + 1
                            sum = sum + 1
            if sum != 0:
                bigram[tmp+j,:] = bigram[tmp+j,:] / sum
        j = j + 2

    return bigram


def extract_argument_features(file):
    with open('../data_all_argument.json', 'r') as f:
        debates = json.load(f)
    f.close()

    with open(file,'rb') as f:
        data = pickle.load(f)
    f.close()

    target_key = data['key']
    X = data['X']

    n = 13786
    smoothing = False

    argument_feature = np.zeros((n,33))
    index = 0

    for i in range(len(target_key)):
        key = target_key[i]
        unigram = calculate_unigram(debates[key],X[i],smoothing)
        argument_feature[index:index+len(X[i]),0:5] = unigram
        bigram = calculate_bigram(debates[key],X[i],smoothing)
        argument_feature[index:index+len(X[i]),5:13] = bigram
        trigram = calculate_trigram(debates[key],X[i],smoothing)
        argument_feature[index:index+len(X[i]),13:23] = trigram
        argument_feature[index:index+len(X[i]),23:27] = calculate_argument_bigram(debates[key],X[i])
        argument_feature[index:index+len(X[i]),27:32] = generate_graph_representation(debates[key],X[i])
        argument_feature[index:index+len(X[i]),32:33] = calculate_density(debates[key],X[i])
        index = index + len(X[i])

    return argument_feature


def extract_user_features(file):
    with open('../data_all_argument.json', 'r') as f:
        debates = json.load(f)
    f.close()

    with open('../users.json', 'r') as f:
        users = json.load(f)
    f.close()

    with open(file,'rb') as f:
        data = pickle.load(f)
    f.close()

    target_key = data['key']
    X = data['X']

    user_feature = np.zeros((2606,6))
    index = 0

    bigissues_dict = build_bigissues_dict(users)
    X_userbased = []
    for i in range(len(target_key)):
        key = target_key[i]
        debate = debates[key]
        debater1 = debate['participant_1_name']
        debater2 = debate['participant_2_name']
        if debater1 not in users or debater2 not in users:
            continue
        voters = []
        for vote in debate['votes']:
            if vote['user_name'] in users:
                voters.append(vote['user_name'])
        for voter in voters:
            user_feature[i,0] = user_feature[i,0] + get_bigissues(bigissues_dict,voter,debater1)
            user_feature[i,1] = user_feature[i,1] + get_bigissues(bigissues_dict,voter,debater2)
            user_feature[i,2] = user_feature[i,2] + get_matching(users,voter,debater1,'Politics')
            user_feature[i,3] = user_feature[i,3] + get_matching(users,voter,debater2,'Politics')
            user_feature[i,4] = user_feature[i,4] + get_matching(users,voter,debater1,'Religion')
            user_feature[i,5] = user_feature[i,5] + get_matching(users,voter,debater2,'Religion')
        if len(voters) != 0:
            user_feature[i,:] = user_feature[i,:] / len(voters)

    return user_feature


def calculate_num_attribute(debate, relative = True):
    num_fact = np.zeros((2,1))
    num_policy = np.zeros((2,1))
    num_value = np.zeros((2,1))
    num_testimony = np.zeros((2,1))
    num_reference = np.zeros((2,1))

    i = 0
    sum = 0
    for round in debate['rounds']:
        for side in round:
            if side['side'] == 'Pro':
                tmp = i
            else:
                tmp = i + 1
            for node in side['nodes_cdcp_svm_strict'].split(','):
                sum = sum + 1
                if node == 'policy':
                    num_policy[tmp, 0] = num_policy[tmp, 0] + 1
                elif node == 'fact':
                    num_fact[tmp, 0] = num_fact[tmp, 0] + 1
                elif node == 'value':
                    num_value[tmp, 0] = num_value[tmp, 0] + 1
                elif node == 'testimony':
                    num_testimony[tmp, 0] = num_testimony[tmp, 0] + 1
                elif node == 'reference':
                    num_reference[tmp, 0] = num_reference[tmp, 0] + 1
                else:
                    raise Exception('Nodes not listed')
    if relative:
        num_policy[i, 0] = num_policy[i, 0] / sum
        num_policy[i + 1, 0] = num_policy[i + 1, 0] / sum
        num_fact[i, 0] = num_fact[i, 0] / sum
        num_fact[i + 1, 0] = num_fact[i + 1, 0] / sum
        num_value[i, 0] = num_value[i, 0] / sum
        num_value[i + 1, 0] = num_value[i + 1, 0] / sum
        num_testimony[i, 0] = num_testimony[i, 0] / sum
        num_testimony[i + 1, 0] = num_testimony[i + 1, 0] / sum
        num_reference[i, 0] = num_reference[i, 0] / sum
        num_reference[i + 1, 0] = num_reference[i + 1, 0] / sum

    return num_policy, num_fact, num_value, num_testimony, num_reference


def calculate_num_links(debate,relative = True):
    num_links = np.zeros((2, 1))
    i = 0
    for round in debate['rounds']:
        for side in round:
            if side['side'] == 'Pro':
                tmp = i
            else:
                tmp = i + 1
            num_links[tmp, 0] = num_links[tmp, 0] + len(
                [j for j in range(len(side['links_cdcp_svm_strict'].split(','))) if
                 side['links_cdcp_svm_strict'].split(',')[j] == 'True'])
    if num_links[i, 0] != 0 or num_links[i + 1, 0] != 0 and relative:
        num_links[i, 0] = num_links[i, 0] / (num_links[i, 0] + num_links[i + 1, 0])
        num_links[i + 1, 0] = num_links[i + 1, 0] / (num_links[i, 0] + num_links[i + 1, 0])

    return num_links


def calculate_support_structure(debate,relative = True):
    basic = np.zeros((2, 1))
    divergent = np.zeros((2, 1))

    i = 0
    links = []
    sent = 0
    source = 0
    target = 0

    sum = 0
    for round in debate['rounds']:
        for side in round:
            structure_s = {}
            if side['side'] == 'Pro':
                tmp = i
            else:
                tmp = i + 1
            links = [j for j in range(len(side['links_cdcp_svm_strict'].split(','))) if
                     side['links_cdcp_svm_strict'].split(',')[j] == 'True']
            sent = len(side['nodes_cdcp_svm_strict'])
            for num in links:
                source = num // (sent - 1)
                target = num % (sent - 1)
                if target >= source:
                    target = target + 1
                if source in structure_s:
                    structure_s[source].append(target)
                else:
                    structure_s[source] = [target]
            for k, v in structure_s.items():
                if len(v) == 1:
                    basic[tmp, 0] = basic[tmp, 0] + 1
                    sum = sum + 1
                elif len(v) > 1:
                    divergent[tmp, 0] = divergent[tmp, 0] + 1
                    sum = sum + 1

    if sum != 0 and relative:
        basic[i, 0] = basic[i, 0] / sum
        basic[i + 1, 0] = basic[i + 1, 0] / sum
        divergent[i, 0] = divergent[i, 0] / sum
        divergent[i + 1, 0] = divergent[i + 1, 0] / sum

    return basic, divergent


def calculate_bigram_logistic(debate, relative = True):
    bigram = np.zeros((2,8))

    bigram_list = ['value+value', 'testimony+value', 'value+testimony', 'value+policy', 'policy+value', 'fact+value', 'value+fact', 'testimony+testimony']

    sum = 0

    for round in debate['rounds']:
        for side in round:
            if side['side'] == 'Pro':
                tmp = 0
            else:
                tmp = 1
            attribute = side['nodes_cdcp_svm_strict'].split(',')
            for i in range(len(attribute) - 1):
                key = attribute[i] + "+" + attribute[i + 1]
                if key in bigram_list:
                    sum = sum + 1
                    bigram[tmp,bigram_list.index(key)] = bigram[tmp,bigram_list.index(key)] + 1

    if sum != 0 and relative:
        bigram[:,:] = bigram[:,:] / sum

    return bigram


def calculate_trigram_logistic(debate, relative=True):
    trigram = np.zeros((2,10))

    trigram_list = ['value+value+value','testimony+value+value','value+value+policy','value+value+testimony', 'value+testimony+value', 'fact+value+value', 'policy+value+value', 'value+fact+value', 'value+policy+value', 'value+value+fact']

    sum = 0

    for round in debate['rounds']:
        for side in round:
            if side['side'] == 'Pro':
                tmp = 0
            else:
                tmp = 1
            attribute = side['nodes_cdcp_svm_strict'].split(',')
            for i in range(len(attribute) - 2):
                key = attribute[i] + "+" + attribute[i+1] + "+" + attribute[i+2]
                if key in trigram_list:
                    sum = sum + 1
                    trigram[tmp,trigram_list.index(key)] = trigram[tmp,trigram_list.index(key)] + 1

    if sum != 0 and relative:
        trigram[:,:] = trigram[:,:] / sum

    return trigram


def extract_feature(debates, feature_matrix, relative_matrix, criterion):
    m = 0
    for k,v in feature_matrix.items():
        if v == 1:
            m = m + 1
    if feature_matrix['attribute_bigram'] == 1:
        m = m + 8 -1
    if feature_matrix['attribute_trigram'] == 1:
        m = m + 10 - 1

    n = len(debates)
    X = np.zeros((n*2,m))
    Y = np.zeros((n*2,1))
    i = 0
    j = 0

    for key,debate in debates.items():

        # feature set 1: num of attribute
        if feature_matrix['num_fact'] == 1 or feature_matrix['num_value'] == 1 or feature_matrix['num_policy'] == 1 or feature_matrix['num_testimony'] == 1 or feature_matrix['num_reference'] == 1:
            num_policy, num_fact, num_value, num_testimony, num_reference = calculate_num_attribute(debate,relative_matrix['attribute'])
        if feature_matrix['num_policy'] == 1:
            X[i, j] = num_policy[0, 0]
            X[i + 1, j] = num_policy[1, 0]
            j = j + 1
        if feature_matrix['num_fact'] == 1:
            X[i, j] = num_fact[0, 0]
            X[i + 1, j] = num_fact[1, 0]
            j = j + 1
        if feature_matrix['num_value'] == 1:
            X[i, j] = num_value[0, 0]
            X[i + 1, j] = num_value[1, 0]
            j = j + 1
        if feature_matrix['num_testimony'] == 1:
            X[i, j] = num_testimony[0, 0]
            X[i + 1, j] = num_testimony[1, 0]
            j = j + 1
        if feature_matrix['num_reference'] == 1:
            X[i, j] = num_reference[0, 0]
            X[i + 1, j] = num_reference[1, 0]
            j = j + 1

        # feature set 2: num of links
        if feature_matrix['num_links'] == 1:
            num_links = calculate_num_links(debate, relative_matrix['link'])
            X[i, j] = num_links[0, 0]
            X[i + 1, j] = num_links[1, 0]
            j = j + 1

        # feature set 3: supporting structure
        if feature_matrix['basic'] == 1 or feature_matrix['divergent'] == 1:
            basic, divergent = calculate_support_structure(debate, relative_matrix['structure'])
        if feature_matrix['basic'] == 1:
            X[i,j] = basic[0,0]
            X[i+1,j] = basic[1,0]
            j = j + 1
        if feature_matrix['divergent'] == 1:
            X[i, j] = divergent[0, 0]
            X[i + 1, j] = divergent[1, 0]
            j = j + 1

        # feature set 4: n-gram attribute
        if feature_matrix['attribute_bigram'] == 1:
            bigram = calculate_bigram_logistic(debate, relative_matrix['n-gram'])
            X[i,j:j+8] = bigram[0,:]
            X[i+1,j:j+8] = bigram[1,:]
            j = j + 8
        if feature_matrix['attribute_trigram'] == 1:
            trigram = calculate_trigram_logistic(debate, relative_matrix['n-gram'])
            X[i,j:j+10] = trigram[0,:]
            X[i+1,j:j+10] = trigram[1,:]
            j = j + 10

        i = i + 2
        j = 0

    return X


def extract_logistic_features(file):
    criterion = 'argument'  # {'points','attitude','argument'}
    feature_matrix = {'num_fact': 0,
                      'num_policy': 0,
                      'num_value': 0,
                      'num_testimony': 0,
                      'num_reference': 0,
                      'attribute_bigram': 1,
                      'attribute_trigram': 1,
                      'num_links': 0,
                      'basic': 0,
                      'divergent': 0}

    relative_matrix = {'attribute': True,
                       'structure': True,
                       'link': True,
                       'n-gram': True}

    debates = {}
    with open("../data_all_argument.json", 'r') as f:
        debates = json.load(f)
    f.close()

    with open(file, 'rb') as f:
        data = pickle.load(f)
    f.close()
    key = data['key']
    debates_new = dict()
    for k in key:
        debates_new[k] = debates[k]
    print(len(debates_new))
    X = extract_feature(debates_new, feature_matrix, relative_matrix, criterion)

    return X

