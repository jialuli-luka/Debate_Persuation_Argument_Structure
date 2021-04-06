from dependency import *
from scipy.stats import wilcoxon

def count_unigram_difference(debates,label,key_list):
    unigram_t = {}
    unigram_f = {}
    unigram_tt = {}
    unigram_ff = {}

    for k,debate in debates.items():
        for round in debate['rounds']:
            for side in round:
                if label[key_list.index(k)] == 0:
                    if side['side'] == 'Pro':
                        attribute = side['node_cdcp_svm_strict'].split(',')
                        for i in range(len(attribute)):
                            key = attribute[i]
                            unigram_f[key] = unigram.get(key, 0) + 1
                        for m,n in unigram_f.items():
                            n = n / len(attribute)
                            if m in unigram_ff:
                                unigram_ff[m].append(n)
                            else:
                                unigram_ff[m] = [n]


def count_bigram(debates):
    bigram = {}

    for _,debate in debates.items():
        for round in debate['rounds']:
            for side in round:
                attribute = side['nodes_cdcp_svm_strict'].split(',')
                for i in range(len(attribute)-1):
                    key = attribute[i] + "+" + attribute[i+1]
                    bigram[key] = bigram.get(key,0) + 1

    return bigram

def count_bigram_difference(debates,label,key_list,attri_list):
    bigram_t = {}
    bigram_f = {}
    bigram_tt = {}
    bigram_ff = {}

    for k,debate in debates.items():
        for round in debate['rounds']:
            for side in round:
                for a in attri_list:
                    bigram_t[a] = 0
                    bigram_f[a] = 0
                if label[key_list.index(k)] == 0:
                    if side['side'] == 'Pro':
                        attribute = side['nodes_cdcp_svm_strict'].split(',')
                        for i in range(len(attribute) - 1):
                            key = attribute[i] + "+" + attribute[i + 1]
                            bigram_f[key] = bigram_f.get(key, 0) + 1
                        for m,n in bigram_f.items():
                            n = n / (len(attribute) - 1)
                            if m in bigram_ff:
                                bigram_ff[m].append(n)
                            else:
                                bigram_ff[m] = [n]
                    else:
                        attribute = side['nodes_cdcp_svm_strict'].split(',')
                        for i in range(len(attribute) - 1):
                            key = attribute[i] + "+" + attribute[i + 1]
                            bigram_t[key] = bigram_t.get(key, 0) + 1
                        for m,n in bigram_t.items():
                            n = n / (len(attribute) - 1)
                            if m in bigram_tt:
                                bigram_tt[m].append(n)
                            else:
                                bigram_tt[m] = [n]
                else:
                    if side['side'] == 'Pro':
                        attribute = side['nodes_cdcp_svm_strict'].split(',')
                        for i in range(len(attribute) - 1):
                            key = attribute[i] + "+" + attribute[i + 1]
                            bigram_t[key] = bigram_t.get(key, 0) + 1
                        for m,n in bigram_t.items():
                            n = n / (len(attribute) - 1)
                            if m in bigram_tt:
                                bigram_tt[m].append(n)
                            else:
                                bigram_tt[m] = [n]
                    else:
                        attribute = side['nodes_cdcp_svm_strict'].split(',')
                        for i in range(len(attribute) - 1):
                            key = attribute[i] + "+" + attribute[i + 1]
                            bigram_f[key] = bigram_f.get(key, 0) + 1
                        for m,n in bigram_f.items():
                            n = n / (len(attribute) - 1)
                            if m in bigram_ff:
                                bigram_ff[m].append(n)
                            else:
                                bigram_ff[m] = [n]

    for a,b in bigram_tt.items():
        bigram_tt[a] = np.mean(b)

    for a,b in bigram_ff.items():
        bigram_ff[a] = np.mean(b)

    return bigram_tt,bigram_ff


def count_trigram(debates):
    trigram = {}

    for _,debate in debates.items():
        for round in debate['rounds']:
            for side in round:
                attribute = side['nodes_cdcp_svm_strict'].split(',')
                for i in range(len(attribute)-2):
                    key = attribute[i] + "+" + attribute[i+1] + "+" + attribute[i+2]
                    trigram[key] = trigram.get(key,0) + 1

    return trigram


def count_trigram_difference(debates,label,key_list,attri_list):
    trigram_t = {}
    trigram_f = {}
    trigram_tt = {}
    trigram_ff = {}

    for k, debate in debates.items():
        for round in debate['rounds']:
            for side in round:
                for a in attri_list:
                    trigram_t[a] = 0
                    trigram_f[a] = 0
                if label[key_list.index(k)] == 0:
                    if side['side'] == 'Pro':
                        attribute = side['nodes_cdcp_svm_strict'].split(',')
                        for i in range(len(attribute) - 2):
                            key = attribute[i] + "+" + attribute[i + 1] + "+" + attribute[i+2]
                            trigram_f[key] = trigram_f.get(key, 0) + 1
                        for m, n in trigram_f.items():
                            n = n / (len(attribute) - 2)
                            if m in trigram_ff:
                                trigram_ff[m].append(n)
                            else:
                                trigram_ff[m] = [n]
                    else:
                        attribute = side['nodes_cdcp_svm_strict'].split(',')
                        for i in range(len(attribute) - 2):
                            key = attribute[i] + "+" + attribute[i + 1] + "+" + attribute[i+2]
                            trigram_t[key] = trigram_t.get(key, 0) + 1
                        for m, n in trigram_t.items():
                            n = n / (len(attribute) - 2)
                            if m in trigram_tt:
                                trigram_tt[m].append(n)
                            else:
                                trigram_tt[m] = [n]
                else:
                    if side['side'] == 'Pro':
                        attribute = side['nodes_cdcp_svm_strict'].split(',')
                        for i in range(len(attribute) - 2):
                            key = attribute[i] + "+" + attribute[i + 1] + "+" + attribute[i+2]
                            trigram_t[key] = trigram_t.get(key, 0) + 1
                        for m, n in trigram_t.items():
                            n = n / (len(attribute) - 2)
                            if m in trigram_tt:
                                trigram_tt[m].append(n)
                            else:
                                trigram_tt[m] = [n]
                    else:
                        attribute = side['nodes_cdcp_svm_strict'].split(',')
                        for i in range(len(attribute) - 2):
                            key = attribute[i] + "+" + attribute[i + 1] + "+" + attribute[i+2]
                            trigram_f[key] = trigram_f.get(key, 0) + 1
                        for m, n in trigram_f.items():
                            n = n / (len(attribute) - 2)
                            if m in trigram_ff:
                                trigram_ff[m].append(n)
                            else:
                                trigram_ff[m] = [n]

    for a, b in trigram_tt.items():
        trigram_tt[a] = np.mean(b)

    for a, b in trigram_ff.items():
        trigram_ff[a] = np.mean(b)

    return trigram_tt, trigram_ff


def count_argument_bigram(debates):
    bigram = {}

    for _,debate in debates.items():
        for round in debate['rounds']:
            for side in round:
                structure_s = {}
                attribute = side['nodes_cdcp_svm_strict'].split(',')
                links = [i for i in range(len(side['links_cdcp_svm_strict'].split(','))) if side['links_cdcp_svm_strict'].split(',')[i] == 'True']
                sent = len(side['nodes_cdcp_svm_strict'].split(','))
                for num in links:
                    source = num // (sent - 1)
                    target = num % (sent - 1)
                    if target >= source:
                        target = target + 1
                    if source in structure_s:
                        structure_s[source].append(target)
                    else:
                        structure_s[source] = [target]
                for k,v in structure_s.items():
                    for j in v:
                        if k < len(attribute) and j < len(attribute):
                            key = attribute[k] + "+" + attribute[j]
                            bigram[key] = bigram.get(key,0) + 1

    return bigram


def count_argument_bigram_difference(debates,label,key_list,attri_list):
    bigram_t = {}
    bigram_f = {}
    bigram_tt = {}
    bigram_ff = {}

    for b, debate in debates.items():
        for round in debate['rounds']:
            for side in round:
                for a in attri_list:
                    bigram_t[a] = 0
                    bigram_f[a] = 0
                structure_s = {}
                attribute = side['nodes_cdcp_svm_strict'].split(',')
                links = [i for i in range(len(side['links_cdcp_svm_strict'].split(','))) if
                         side['links_cdcp_svm_strict'].split(',')[i] == 'True']
                sent = len(side['nodes_cdcp_svm_strict'].split(','))
                for num in links:
                    source = num // (sent - 1)
                    target = num % (sent - 1)
                    if target >= source:
                        target = target + 1
                    if source in structure_s:
                        structure_s[source].append(target)
                    else:
                        structure_s[source] = [target]
                # print("s",structure_s)
                count = 0
                if label[key_list.index(b)] == 0:
                    if side['side'] == 'Pro':
                        for k, v in structure_s.items():
                            for j in v:
                                if k < len(attribute) and j < len(attribute):
                                    key = attribute[k] + "+" + attribute[j]
                                    bigram_f[key] = bigram_f.get(key, 0) + 1
                                    count = count + 1
                        for m, n in bigram_f.items():
                            count = len(links)
                            if count != 0:
                                n = n / count
                                if m in bigram_ff:
                                    bigram_ff[m].append(n)
                                else:
                                    bigram_ff[m] = [n]
                        # print("f",bigram_f)
                        # print("ff",bigram_ff)
                    else:
                        for k, v in structure_s.items():
                            for j in v:
                                if k < len(attribute) and j < len(attribute):
                                    key = attribute[k] + "+" + attribute[j]
                                    bigram_t[key] = bigram_t.get(key, 0) + 1
                                    count = count + 1
                        for m, n in bigram_t.items():
                            count = len(links)
                            if count != 0:
                                n = n / count
                                if m in bigram_tt:
                                    bigram_tt[m].append(n)
                                else:
                                    bigram_tt[m] = [n]
                else:
                    if side['side'] == 'Pro':
                        for k, v in structure_s.items():
                            for j in v:
                                if k < len(attribute) and j < len(attribute):
                                    key = attribute[k] + "+" + attribute[j]
                                    bigram_t[key] = bigram_t.get(key, 0) + 1
                                    count = count + 1
                        for m, n in bigram_t.items():
                            count = len(links)
                            if count != 0:
                                n = n / count
                                if m in bigram_tt:
                                    bigram_tt[m].append(n)
                                else:
                                    bigram_tt[m] = [n]
                    else:
                        for k, v in structure_s.items():
                            for j in v:
                                if k < len(attribute) and j < len(attribute):
                                    key = attribute[k] + "+" + attribute[j]
                                    bigram_f[key] = bigram_f.get(key, 0) + 1
                                    count = count + 1
                        for m, n in bigram_f.items():
                            count = len(links)
                            if count != 0:
                                n = n / count
                                if m in bigram_ff:
                                    bigram_ff[m].append(n)
                                else:
                                    bigram_ff[m] = [n]



    for a, b in bigram_tt.items():
        bigram_tt[a] = np.mean(b)

    for a, b in bigram_ff.items():
        bigram_ff[a] = np.mean(b)

    return bigram_tt, bigram_ff


def count_graph_sub(graph_tmp,graph,structure_s,structure_t):
    graph_f = graph_tmp
    graph_ff = graph
    sum = 0
    for k, v in structure_s.items():
        if len(v) == 1 and len(structure_t[v[0]]) == 1:
            graph_f['basic'] = graph_f['basic'] + 1
            sum = sum + 1
        elif len(v) == 2:
            graph_f['divergent'] = graph_f['divergent'] + 1
            sum = sum + 1
        elif len(v) > 2:
            graph_f['divergent+'] = graph_f['divergent+'] + 1
            sum = sum + 1
    for k, v in structure_t.items():
        if len(v) == 2:
            graph_f['convergent'] = graph_f['convergent'] + 1
            sum = sum + 1
        elif len(v) > 2:
            graph_f['convergent+'] = graph_f['convergent+'] + 1
            sum = sum + 1
    for m, n in graph_f.items():
        if sum != 0:
            n = n / sum
            if m in graph_ff:
                graph_ff[m].append(n)
            else:
                graph_ff[m] = [n]

    return graph_f,graph_ff


def count_graph_representation(debates,label,key_list):
    graph_tt = {}
    graph_ff = {}
    graph_t = {}
    graph_f = {}

    attri_list = ['basic','divergent','divergent+','convergent','convergent+']

    for b,debate in debates.items():
        for round in debate['rounds']:
            for side in round:
                for a in attri_list:
                    graph_t[a] = 0
                    graph_f[a] = 0
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
                if label[key_list.index(b)] == 0:
                    if side['side'] == 'Pro':
                        graph_f,graph_ff = count_graph_sub(graph_f, graph_ff, structure_s, structure_t)
                    else:
                        graph_t,graph_tt = count_graph_sub(graph_t,graph_tt,structure_s,structure_t)
                else:
                    if side['side'] == 'Pro':
                        graph_t,graph_tt = count_graph_sub(graph_t,graph_tt,structure_s,structure_t)
                    else:
                        graph_f,graph_ff = count_graph_sub(graph_f, graph_ff, structure_s, structure_t)

    for a,b in graph_tt.items():
        graph_tt[a] = np.mean(b)

    for a,b in graph_ff.items():
        graph_ff[a] = np.mean(b)

    return graph_tt,graph_ff


def count_unigram_wilcoxon(debates,label,key_list):
    win = []
    lose = []
    for b,debate in debates.items():
        testimony1 = [[], []]
        # count = 0
        for round in debate['rounds']:
            testimony = [0, 0]
            for side in round:
                if side['side'] == 'Pro':
                    tmp = 0
                else:
                    tmp = 1
                attribute = side['nodes_cdcp_svm_strict'].split(',')
                count = len(attribute)
                for node in attribute:
                    if node == 'reference':
                        testimony[tmp] += 1
                testimony[tmp] = testimony[tmp] / count
                testimony1[tmp].append(testimony[tmp])
        if label[key_list.index(b)] == 0:
            win.append(np.mean(testimony1[1]))
            lose.append(np.mean(testimony1[0]))
        else:
            win.append(np.mean(testimony1[0]))
            lose.append(np.mean(testimony1[1]))

    print(sum(win))
    print(sum(lose))

    return wilcoxon(win,lose)


def count_link_wilcoxon(debates,label,key_list):
    win = []
    lose = []
    for b,debate in debates.items():
        testimony1 = [[], []]
        count = 0
        for round in debate['rounds']:
            testimony = [0, 0]
            for side in round:
                if side['side'] == 'Pro':
                    tmp = 0
                else:
                    tmp = 1
                structure_s = {}
                attribute = side['nodes_cdcp_svm_strict'].split(',')
                links = [i for i in range(len(side['links_cdcp_svm_strict'].split(','))) if
                         side['links_cdcp_svm_strict'].split(',')[i] == 'True']
                sent = len(side['nodes_cdcp_svm_strict'].split(','))
                count += len(links)
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
                    for j in v:
                        if k < len(attribute) and j < len(attribute):
                            key = attribute[k] + "+" + attribute[j]
                            if key == 'fact+value':
                                testimony[tmp] += 1
                if count != 0:
                    testimony[tmp] = testimony[tmp] / count
                    testimony1[tmp].append(testimony[tmp])
        if label[key_list.index(b)] == 0:
            win.append(np.mean(testimony1[1]) if len(testimony1[1]) != 0 else 0)
            lose.append(np.mean(testimony1[0]) if len(testimony1[0]) != 0 else 0)
        else:
            win.append(np.mean(testimony1[0]) if len(testimony1[0]) != 0 else 0)
            lose.append(np.mean(testimony1[1]) if len(testimony1[1]) != 0 else 0)

    print(sum(win))
    print(sum(lose))

    return wilcoxon(win,lose)


def count_graph_wilcoxon(debates,label,key_list):
    win = []
    lose = []
    for b, debate in debates.items():
        testimony1 = [[], []]
        count = 0
        for round in debate['rounds']:
            testimony = [0, 0]
            for side in round:
                if side['side'] == 'Pro':
                    tmp = 0
                else:
                    tmp = 1
                structure_s = {}
                structure_t = {}
                attribute = side['nodes_cdcp_svm_strict'].split(',')
                links = [i for i in range(len(side['links_cdcp_svm_strict'].split(','))) if
                         side['links_cdcp_svm_strict'].split(',')[i] == 'True']
                sent = len(side['nodes_cdcp_svm_strict'].split(','))
                count += len(links)
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
                    if len(v) >= 2:
                        testimony[tmp] += 1
                if count != 0:
                    # testimony[tmp] = testimony[tmp] / count
                    testimony1[tmp].append(testimony[tmp])
        if label[key_list.index(b)] == 0:
            win.append(np.mean(testimony1[1]) if len(testimony1[1]) != 0 else 0)
            lose.append(np.mean(testimony1[0]) if len(testimony1[0]) != 0 else 0)
        else:
            win.append(np.mean(testimony1[0]) if len(testimony1[0]) != 0 else 0)
            lose.append(np.mean(testimony1[1]) if len(testimony1[1]) != 0 else 0)

    print(sum(win))
    print(sum(lose))

    return wilcoxon(win, lose)



if __name__ == '__main__':
    debates = {}
    with open("data_all_argument.json", 'r') as f:
        debates = json.load(f)
    f.close()

    with open("./BERT/dataforBERT_X_all_argument_last_three_sentence_full.pickle",'rb') as f:
        data = pickle.load(f)
    f.close()
    key = data['key']
    Y = data['Y']

    debates_new = dict()
    for k in key:
        debates_new[k] = debates[k]
    print(len(debates_new))

    w,p = count_graph_wilcoxon(debates, Y, key)
    print(w,p)
