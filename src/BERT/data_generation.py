import numpy as np
import pickle
from nltk import word_tokenize
import random
import json
import os
import shutil
import tempfile
import sys
from subprocess import run, PIPE


CORENLP_PATH = os.path.join('/stanford-corenlp-full-2015-12-09','*')


def parse_generate_link(structure):

    links = [j for j in range(len(structure['links_cdcp_svm_strict'].split(','))) if
             structure['links_cdcp_svm_strict'].split(',')[j] == 'True']
    sent = len(structure['nodes_cdcp_svm_strict'].split(','))
    link_dic = dict()
    for num in links:
        source = num // (sent - 1)
        target = num % (sent - 1)
        if target >= source:
            target = target + 1
        if source in link_dic:
            link_dic[source].append(target)
        else:
            link_dic[source] = [target]
    return link_dic


def find_top_sentence_support_window(generate_link):

    generate_dic = {}

    for k, v in generate_link.items():
        if k in generate_dic:
            generate_dic[k] = generate_dic[k] + len(v)
        else:
            generate_dic[k] = len(v)

    generate_sent = sorted(generate_dic.items(), key=lambda item: item[1], reverse=True)

    if len(generate_sent) == 0:
        return -1
    else:
        return generate_sent[0][0]


def generate_label(debate,criterion):
    if criterion == 'points':
        if debate['participant_1_position'] == 'Pro' and debate['participant_1_status'] == 'Winning':
            return 1
        elif debate['participant_2_position'] == 'Con' and debate['participant_2_status'] == 'Winning':
            return 0
        else:
            print("error")
    elif criterion == 'argument':
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
            return 1
        elif temp_pro < temp_con:
            return 0
        else:
            print("error")


def generate_json(filename,corenlp_mem="2g"):
    result = run([
        'java', '-cp', CORENLP_PATH, "-Xmx{}".format(corenlp_mem),
        "edu.stanford.nlp.pipeline.StanfordCoreNLP", "-annotators",
        "tokenize,ssplit,pos,lemma,ner,parse,depparse",
        "-file", filename, "-outputFormat", "json", "-outputDirectory", "../debate_corenlp_json/"],
        stdout=PIPE, stderr=PIPE)

    if result.returncode != 0:
        raise ValueError("CoreNLP failed: {}".format(result.stderr.decode()))



def generate_preliminary_data():
    file = '../data_all_argument.json'
    with open(file,'r') as f:
        debates = json.load(f)
    f.close()
    print("total debates:",len(debates))
    # criterion = 'points'
    criterion = 'argument'
    # debates, Y = data_processing_for_RNN(file, criterion)
    X = []
    core_sentence = []
    node_attribute = []
    title = []
    Y = []
    key_remain = []
    label = 0
    for key, debate in debates.items():
        print(label)
        label = label + 1
        Seq = []
        top_sentence = []
        nodes = []
        eliminate = False
        i = 0
        for round in debate['rounds']:
            i = i + 1
            Pro = []
            Con = []
            for side in round:
                if side['side'] == 'Pro':
                    if not os.path.exists('../debate_corenlp_json/' + str(key) + '_' + str(i) + '_' + side['side'] + '.txt.json'):
                        with open('../debate_corenlp_json/' + str(key) + '_' + str(i) + '_' + side['side'] + '.txt','w') as f:
                            f.write(side['text'])
                        f.close()
                        generate_json('../debate_corenlp_json/' + str(key) + '_' + str(i) + '_' + side['side'] + '.txt')
                    with open('../debate_corenlp_json/' + str(key) + '_' + str(i) + '_' + side['side'] + '.txt.json','r') as f:
                        parse_data = json.load(f)
                    f.close()
                    for sent in parse_data['sentences']:
                        embedding_pro = []
                        for token in sent['tokens']:
                            embedding_pro.append(token['word'])
                        Pro.append(embedding_pro)
                    links = parse_generate_link(side)
                    nodes_pro = side['nodes_cdcp_svm_strict']
                    top_sentence_pro = find_top_sentence_support_window(links)
                else:
                    if not os.path.exists('../debate_corenlp_json/' + str(key) + '_' + str(i) + '_' + side['side'] + '.txt.json'):
                        with open('../debate_corenlp_json/' + str(key) + '_' + str(i) + '_' + side['side'] + '.txt','w') as f:
                            f.write(side['text'])
                        f.close()
                        generate_json('../debate_corenlp_json/' + str(key) + '_' + str(i) + '_' + side['side'] + '.txt')
                    with open('../debate_corenlp_json/' + str(key) + '_' + str(i) + '_' + side['side'] + '.txt.json','r') as f:
                        parse_data = json.load(f)
                    f.close()
                    for sent in parse_data['sentences']:
                        embedding_con = []
                        for token in sent['tokens']:
                            embedding_con.append(token['word'])
                        Con.append(embedding_con)
                    links = parse_generate_link(side)
                    nodes_con = side['nodes_cdcp_svm_strict']
                    top_sentence_con = find_top_sentence_support_window(links)
            if eliminate == True:
                break
            Seq.append(Pro)
            Seq.append(Con)
            top_sentence.append(top_sentence_pro)
            top_sentence.append(top_sentence_con)
            nodes.append(nodes_pro)
            nodes.append(nodes_con)
        if eliminate != True:
            X.append(Seq)
            core_sentence.append(top_sentence)
            node_attribute.append(nodes)
            Y.append(generate_label(debate, criterion))
            title.append(debate['title'])
            key_remain.append(key)

    data = dict()
    data['X'] = X
    data['Y'] = Y
    data['core_sentence'] = core_sentence
    data['nodes'] = node_attribute
    data['title'] = title
    data['key'] = key_remain

    print(len(X))
    print(len(Y))
    print(len(core_sentence))
    print(len(node_attribute))
    print(len(title))
    print(len(key_remain))

    with open('dataforBERT_all_argument.pickle', 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()



if __name__ == '__main__':

    generate_preliminary_data()

    X = data['X']
    Y = data['Y']
    core_sentence = data['core_sentence']
    nodes = data['nodes']
    title = data['title']
    key_remain = data['key']

    with open('../data_all_argument.json','r') as f:
        debates = json.load(f)
    f.close()

    data_X_tmp = []
    data_X = []
    data_attribute_tmp = []
    data_attribute_X = []
    title_X = []
    Y_X = []
    key_X = []

    # last 3 sentences
    for i in range(len(X)):
        data_X_tmp = []
        for j in range(len(X[i])):
            data_X_tmp.append(X[i][j][-3:])
            data_attribute_tmp.append(nodes[i][j].split(',')[-3:])
        data_X.append(data_X_tmp)
        data_attribute_X.append(data_attribute_tmp)
        title_X.append(title[i])
        Y_X.append(Y[i])
        key_X.append(key_remain[i])

    data = dict()
    data['X'] = data_X
    data['Y'] = Y_X
    data['title'] = title_X
    data['nodes'] = data_attribute_X
    data['key'] = key_X
    print(len(data_X))
    with open('dataforBERT_X_all_argument_last_three_sentence_full.pickle', 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()
