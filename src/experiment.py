from dependency import *
# from data_processing import *
from feature_extraction import *
from model import *

def get_results(X, Y, tag):
    P = LR_train_validation(X,Y)

    sum = 0
    std = 0
    for precision in P:
        sum = sum + precision
    average = sum / len(P)
    for precision in P:
        std = std + (precision - average)**2
    std = (std/len(P))**(1/2)

    print(tag)
    print("Precision:",P)
    print("Precision:",average)
    print("Std:",std)

feature_matrix =   {'num_fact': 1,
                    'num_policy': 1,
                    'num_value': 1,
                    'num_testimony': 1,
                    'num_reference': 1,
                    'attribute_bigram': 1,
                    'attribute_trigram': 1,
                    'num_links': 0,
                    'link_bigram': 1,
                    'basic': 0,
                    'divergent': 0,
                    'convergent': 0,
                    'graph': 1,
                    'language': 1,
                    'tfidf': 1,
                    'user': 1}

relative_matrix =  {'attribute': True,
                    'structure': False,
                    'link': False,
                    'n-gram': False}


with open('data_all_argument.json','r') as f:
    debates = json.load(f)
f.close()

with open('users.json', 'r') as f:
    users = json.load(f)
f.close()

X, Y = extract_feature(debates, users, feature_matrix, relative_matrix)

X_arg = X[:,:-79]
X_ling = X[:,-79:-23]
X_ling_arg = X[:,:-23]
X_ling_user = X[:,-79:]

print(len(X),len(Y))

majority = 0
for i in range(len(Y)):
    if i % 2 == 0 and Y[i,0] == 1:
        majority = majority + 1
print(majority)

get_results(X, Y, 'all')
get_results(X_arg, Y, 'arg_only')
get_results(X_ling, Y, 'ling_only')
get_results(X_ling_arg, Y, 'ling+arg')
get_results(X_ling_user, Y, 'ling+user')
