from dependency import *
from language_features import *
from user_features import *


def calculate_num_attribute(debate, relative = True):
    num_fact = np.zeros((2,1))
    num_policy = np.zeros((2,1))
    num_value = np.zeros((2,1))
    num_testimony = np.zeros((2,1))
    num_reference = np.zeros((2,1))

    sum = 0
    for round in debate['rounds']:
        for side in round:
            if side['side'] == 'Pro':
                tmp = 0
            else:
                tmp = 1
            attribute = side['nodes_cdcp_svm_strict'].split(',')
            for node in attribute:
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
        num_policy[0, 0] = num_policy[0, 0] / sum
        num_policy[1, 0] = num_policy[1, 0] / sum
        num_fact[0, 0] = num_fact[0, 0] / sum
        num_fact[1, 0] = num_fact[1, 0] / sum
        num_value[0, 0] = num_value[0, 0] / sum
        num_value[1, 0] = num_value[1, 0] / sum
        num_testimony[0, 0] = num_testimony[0, 0] / sum
        num_testimony[1, 0] = num_testimony[1, 0] / sum
        num_reference[0, 0] = num_reference[0, 0] / sum
        num_reference[1, 0] = num_reference[1, 0] / sum

    return num_policy, num_fact, num_value, num_testimony, num_reference


def calculate_num_links(debate,relative = True):
    num_links = np.zeros((2, 1))
    for round in debate['rounds']:
        for side in round:
            if side['side'] == 'Pro':
                tmp = 0
            else:
                tmp = 1
            links = side['links_cdcp_svm_strict']
            num_links[tmp, 0] = num_links[tmp, 0] + len([j for j in range(len(links.split(','))) if links.split(',')[j] == 'True'])

    if num_links[0, 0] != 0 or num_links[1, 0] != 0 and relative:
        num_links[0, 0] = num_links[0, 0] / (num_links[0, 0] + num_links[1, 0])
        num_links[1, 0] = num_links[1, 0] / (num_links[0, 0] + num_links[1, 0])

    return num_links


def calculate_support_structure(debate,relative = True):
    basic = np.zeros((2, 1))
    divergent = np.zeros((2, 1))
    convergent = np.zeros((2, 1))

    links = []
    sent = 0
    source = 0
    target = 0

    sum = 0
    for round in debate['rounds']:
        for side in round:
            if side['side'] == 'Pro':
                tmp = 0
            else:
                tmp = 1
            structure_s = {}
            structure_t = {}
            nodes = side['nodes_cdcp_svm_strict']
            links = side['links_cdcp_svm_strict']
            link = [j for j in range(len(links.split(','))) if links.split(',')[j] == 'True']
            sent = len(nodes.split(','))
            for num in link:
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
                if len(v) == 1:
                    basic[tmp, 0] = basic[tmp, 0] + 1
                    sum = sum + 1
                elif len(v) > 1:
                    divergent[tmp, 0] = divergent[tmp, 0] + 1
                    sum = sum + 1
            for k, v in structure_t.items():
                if len(v) > 1:
                    convergent[tmp, 0] = convergent[tmp, 0] + 1
                    sum = sum + 1

    if sum != 0 and relative:
        basic[0, 0] = basic[0, 0] / sum
        basic[1, 0] = basic[1, 0] / sum
        divergent[0, 0] = divergent[0, 0] / sum
        divergent[1, 0] = divergent[1, 0] / sum
        convergent[0, 0] = convergent[0, 0] / sum
        convergent[1, 0] = convergent[1, 0] / sum

    return basic, divergent, convergent


def calculate_bigram(debate, relative = True):
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
                    bigram[tmp, bigram_list.index(key)] = bigram[tmp, bigram_list.index(key)] + 1

    if sum != 0 and relative:
        bigram[:,:] = bigram[:,:] / sum

    return bigram


def calculate_trigram(debate, relative=True):
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
                key = attribute[i] + "+" + attribute[i + 1] + "+" + attribute[i + 2]
                if key in trigram_list:
                    sum = sum + 1
                    trigram[tmp, trigram_list.index(key)] = trigram[tmp, trigram_list.index(key)] + 1

    if sum != 0 and relative:
        trigram[:,:] = trigram[:,:] / sum

    return trigram


def calculate_argument_bigram(debate, relative=True):
    bigram = np.zeros((2,4))

    bigram_list = ['value+value','value+policy','fact+value','testimony+value']

    sum = 0
    for round in debate['rounds']:
        for side in round:
            if side['side'] == 'Pro':
                tmp = 0
            else:
                tmp = 1
            structure_s = {}
            nodes = side['nodes_cdcp_svm_strict']
            links = side['links_cdcp_svm_strict']
            link = [j for j in range(len(links.split(','))) if links.split(',')[j] == 'True']
            sent = len(nodes.split(','))
            attribute = nodes.split(',')
            for num in link:
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
                            bigram[tmp, bigram_list.index(key)] = bigram[tmp, bigram_list.index(key)] + 1
                            sum = sum + 1

    if sum != 0 and relative:
        bigram[tmp,:] = bigram[tmp,:] / sum

    return bigram


def generate_graph_representation(debate, relative = True):
    graph = np.zeros((2,5))   # basic, divergent_2, divergent++, convergent_2, convergent++

    sum = 0
    for round in debate['rounds']:
        for side in round:
            if side['side'] == 'Pro':
                tmp = 0
            else:
                tmp = 1
            structure_s = {}
            structure_t = {}
            nodes = side['nodes_cdcp_svm_strict']
            links = side['links_cdcp_svm_strict']
            link = [j for j in range(len(links.split(','))) if links.split(',')[j] == 'True']
            sent = len(nodes.split(','))
            for num in link:
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
                    graph[tmp, 0] = graph[tmp, 0] + 1
                    sum = sum + 1
                elif len(v) == 2:
                    graph[tmp, 1] = graph[tmp, 1] + 1
                    sum = sum + 1
                elif len(v) > 2:
                    graph[tmp, 2] = graph[tmp, 2] + 1
                    sum = sum + 1
            for k, v in structure_t.items():
                if len(v) == 2:
                    graph[tmp, 3] = graph[tmp, 3] + 1
                    sum = sum + 1
                elif len(v) > 2:
                    graph[tmp, 4] = graph[tmp, 4] + 1
                    sum = sum + 1

    if sum != 0 and relative:
        graph[tmp,:] = graph[tmp,:] / sum

    return graph


def calculate_user_feature(debate, users, bigissues_dict):
    user_feature = np.zeros((1,6))
    debater1 = debate['participant_1_name']
    debater2 = debate['participant_2_name']
    if debater1 not in users or debater2 not in users:
        return user_feature
    voters = []
    for vote in debate['votes']:
        if vote['user_name'] in users:
            voters.append(vote['user_name'])
    for voter in voters:
        user_feature[0, 0] += get_bigissues(bigissues_dict, voter, debater1)
        user_feature[0, 1] += get_bigissues(bigissues_dict, voter, debater2)
        user_feature[0, 2] += get_matching(users, voter, debater1, 'Politics')
        user_feature[0, 3] += get_matching(users, voter, debater2, 'Politics')
        user_feature[0, 4] += get_matching(users, voter, debater1, 'Religion')
        user_feature[0, 5] += get_matching(users, voter, debater2, 'Religion')
    if len(voters) != 0:
        user_feature[0, :] = user_feature[0, :] / len(voters)

    return user_feature


def generate_label(debate):

    Y = np.zeros((2,1))

    temp_pro = 0
    temp_con = 0
    Pro = debate['participant_1_name']
    Con = debate['participant_2_name']
    for user in debate['votes']:
        for name, attitude in user['votes_map'].items():
            if 'Made more convincing arguments' in attitude:
                if name == Pro and attitude['Made more convincing arguments'] == True:
                    temp_pro = temp_pro + 1
                elif name == Con and attitude['Made more convincing arguments'] == True:
                    temp_con = temp_con + 1
    if temp_pro > temp_con:
        Y[0,0] = 1
    elif temp_pro < temp_con:
        Y[1,0] = 1
    else:
        print("error")

    return Y


def extract_feature(debates, users, feature_matrix, relative_matrix):
    m = 0
    for k,v in feature_matrix.items():
        if v == 1:
            m = m + 1
    if feature_matrix['language'] == 1:
        m = m + 23 - 1
    if feature_matrix['attribute_bigram'] == 1:
        m = m + 8 -1
    if feature_matrix['attribute_trigram'] == 1:
        m = m + 10 - 1
    if feature_matrix['link_bigram'] == 1:
        m = m + 4 - 1
    if feature_matrix['graph'] == 1:
        m = m + 5 - 1
    if feature_matrix['tfidf'] == 1:
        m = m + 50 - 1
    if feature_matrix['user'] == 1:
        m = m + 6 - 1
    
    n=0
    for k,debate in debates.items():
        if debate['category'] == 'Politics':
            n += 1
#    n = len(debates)
    X = np.zeros((n*2,m))
    Y = np.zeros((n*2,1))
    i = 0
    j = 0
    vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=50, stop_words="english")
    text_all = []
    if feature_matrix['tfidf'] == 1:
        for k,debate in debates.items():
            text_input = ['', '']
            for round in debate['rounds']:
                for side in round:
                    if side['side'] == 'Pro':
                        text_input[0] += side['text']
                    else:
                        text_input[1] += side['text']
            text_all.append(text_input[0])
            text_all.append(text_input[1])
        vectorizer.fit_transform(text_all)

    if feature_matrix['user'] == 1:
        bigissues_dict = build_bigissues_dict(users)

    for k,debate in debates.items():
        if debate['category'] != 'Politics':
            continue
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
        if feature_matrix['basic'] == 1 or feature_matrix['divergent'] == 1 or feature_matrix['convergent'] == 1:
            basic, divergent, convergent = calculate_support_structure(debate, relative_matrix['structure'])
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
            bigram = calculate_bigram(debate, relative_matrix['n-gram'])
            X[i,j:j+8] = bigram[0,:]
            X[i+1,j:j+8] = bigram[1,:]
            j = j + 8
        if feature_matrix['attribute_trigram'] == 1:
            trigram = calculate_trigram(debate, relative_matrix['n-gram'])
            X[i,j:j+10] = trigram[0,:]
            X[i+1,j:j+10] = trigram[1,:]
            j = j + 10

        # feature set 5: support link
        if feature_matrix['link_bigram'] == 1:
            link_bigram = calculate_argument_bigram(debate)
            X[i,j:j+4] = link_bigram[0,:]
            X[i+1,j:j+4] = link_bigram[1,:]
            j = j + 4

        # feature set 5: graph
        if feature_matrix['graph'] == 1:
            graph_feature = generate_graph_representation(debate)
            X[i,j:j+5] = graph_feature[0,:]
            X[i+1,j:j+5] = graph_feature[1,:]
            j = j + 5

        if feature_matrix['language'] == 1:
            text_input = ['','']
            for round in debate['rounds']:
                for side in round:
                    if side['side'] == 'Pro':
                        text_input[0] += side['text']
                    else:
                        text_input[1] += side['text']
            language_feature_pro = text_to_features(text_input[0])
            language_feature_con = text_to_features(text_input[1])
            for k in range(len(language_feature_pro)):
                X[i,j] = language_feature_pro[k]
                X[i+1,j] = language_feature_con[k]
                j = j + 1

        if feature_matrix['tfidf'] == 1:
            text_input = ['', '']
            for round in debate['rounds']:
                for side in round:
                    if side['side'] == 'Pro':
                        text_input[0] += side['text']
                    else:
                        text_input[1] += side['text']
            X[i,j:j+50] = np.array(vectorizer.transform([text_input[0]]).toarray())
            X[i+1,j:j+50] = np.array(vectorizer.transform([text_input[1]]).toarray())
            j = j + 50

        if feature_matrix['user'] == 1:
            user_features = calculate_user_feature(debate, users, bigissues_dict)
            X[i,j:j+6] = user_features
            X[i+1,j:j+6] = user_features
            j = j + 6

        label = generate_label(debate)
        Y[i,0] = label[0,0]
        Y[i+1,0] = label[1,0]

        i = i + 2
        j = 0

    return X,Y




