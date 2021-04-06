import json
import os
import math
import matplotlib.pyplot as plt


def filter_no_arg(debates):
    # filter out debates without argument structure(eliminated during generation)

    debate_generate = {}
    flag = False
    for key, debate in debates.items():
        for round in debate['rounds']:
            for side in round:
                if 'nodes_cdcp_svm_strict' in side:
                    flag = True
                else:
                    flag = False
        if flag:
            debate_generate[key] = debate
            flag = False

    return debate_generate


def filter_none_arg(debates):
    # filter out debates that are too short which results in None argument structure
    debate_generate = {}
    flag = False
    for key, debate in debates.items():
        for round in debate['rounds']:
            for side in round:
                if side['nodes_cdcp_svm_strict'] == 'None':
                    flag = True
        if flag == False:
            debate_generate[key] = debate
        else:
            flag = False

    return debate_generate


def filter_no_text(debates):
    # filter out debates that one side says nothing
    debate_generate = {}
    for key, debate in debates.items():
        text = ['', '']
        for round in debate['rounds']:
            # print(debate)
            for side in round:
                if side['side'] == 'Pro':
                    text[0] = text[0] + side['text']
                else:
                    text[1] = text[1] + side['text']
        if len(text[0]) == 0 or len(text[1]) == 0:
            continue
        else:
            debate_generate[key] = debate

    return debate_generate


def filter_tied(debates):
    # filter out tied debates
    debate_generate = {}
    for key, debate in debates.items():
        if debate['participant_1_status'] != debate['participant_2_status']:
            debate_generate[key] = debate

    return debate_generate


def filter_no_one_changes_mind(debates):
    # filter out debates that no one change their mind
    debate_generate = {}
    flag = False
    for key, debate in debates.items():
        for user in debate['votes']:
            for name, attitude in user['votes_map'].items():
                if 'Agreed with before the debate' in attitude and 'Agreed with after the debate' in attitude:
                    if attitude['Agreed with before the debate'] != attitude['Agreed with after the debate']:
                        debate_generate[key] = debate
                        flag = True
                        break
                else:
                    print(key)
            if flag == True:
                flag = False
                break

    return debate_generate


def filter_tied_convincing_argument(debates):
    # filter out debates that are tied on Made more convincing arguments
    debate_generate = {}
    flag = False
    for key,debate in debates.items():
        for user in debate['votes']:
            for name,attitude in user['votes_map'].items():
                if 'Made more convincing arguments' in attitude:
                    if attitude['Made more convincing arguments'] == True:
                        debate_generate[key] = debate
                        flag = True
                        break
                else:
                    print(key)
            if flag == True:
                flag = False
                break

    return debate_generate


def count_avg_points_difference(debates):

    diff = []
    for key, debate in debates.items():
        diff.append(abs(debate['participant_1_points'] - debate['participant_2_points']))

    sum = 0
    std = 0
    for d in diff:
        sum = sum + d
    average = sum / len(diff)
    for d in diff:
        std = std + (d - average) ** 2
    std = (std / len(diff)) ** (1 / 2)

    return diff,average,std


def filter_points_less_than_x(debates,x=1):
    # filter out debates that the difference of points of two sides is less than x

    debate_generate = {}
    for key,debate in debates.items():
        if abs(debate['participant_1_points'] - debate['participant_2_points']) > x:
            debate_generate[key] = debate

    return debate_generate


def count_avg_ppl_change_mind(debates):
    num = []

    for key, debate in debates.items():
        temp_pro = 0
        temp_con = 0
        Pro = debate['participant_1_name']
        Con = debate['participant_2_name']
        for user in debate['votes']:
            for name, attitude in user['votes_map'].items():
                if name == Pro and attitude['Agreed with before the debate'] == True:
                    agree_before = Pro
                elif name == Con and attitude['Agreed with before the debate'] == True:
                    agree_before = Con
                else:
                    agree_before = 'Tied'
                if name == Pro and attitude['Agreed with after the debate'] == True:
                    agree_after = Pro
                elif name == Con and attitude['Agreed with after the debate'] == True:
                    agree_after = Con
                else:
                    agree_after = 'Tied'
                if (agree_before == Pro or agree_before == 'Tied') and agree_after == Con:
                    temp_con = temp_con + 1
                elif (agree_before == Con or agree_before == 'Tied') and agree_after == Pro:
                    temp_pro = temp_pro + 1
        # print(temp_pro,temp_con)
        if temp_pro != 0:
            num.append(temp_pro)
        if temp_con != 0:
            num.append(temp_con)

    sum = 0
    std = 0
    for n in num:
        sum = sum + n
    average = sum / len(num)
    for n in num:
        std = std + (n - average) ** 2
    std = (std / len(num)) ** (1 / 2)

    return num,average,std


def count_ppl_change_mind_over_x(debates,x):
    count = 0

    for key, debate in debates.items():
        temp_pro = 0
        temp_con = 0
        Pro = debate['participant_1_name']
        Con = debate['participant_2_name']
        for user in debate['votes']:
            for name, attitude in user['votes_map'].items():
                if name == Pro and attitude['Agreed with before the debate'] == True:
                    agree_before = Pro
                elif name == Con and attitude['Agreed with before the debate'] == True:
                    agree_before = Con
                else:
                    agree_before = 'Tied'
                if name == Pro and attitude['Agreed with after the debate'] == True:
                    agree_after = Pro
                elif name == Con and attitude['Agreed with after the debate'] == True:
                    agree_after = Con
                else:
                    agree_after = 'Tied'
                if (agree_before == Pro or agree_before == 'Tied') and agree_after == Con:
                    temp_con = temp_con + 1
                elif (agree_before == Con or agree_before == 'Tied') and agree_after == Pro:
                    temp_pro = temp_pro + 1
        # print(temp_pro,temp_con)
        if temp_pro - temp_con >= x:
            count += 1
        elif temp_con - temp_pro >= x:
            count += 1

    return count


def count_avg_percent_ppl_change_mind(debates):
    percent = []

    for key, debate in debates.items():
        temp_pro = 0
        temp_con = 0
        Pro = debate['participant_1_name']
        Con = debate['participant_2_name']
        for user in debate['votes']:
            for name, attitude in user['votes_map'].items():
                if name == Pro and attitude['Agreed with before the debate'] == True:
                    agree_before = Pro
                elif name == Con and attitude['Agreed with before the debate'] == True:
                    agree_before = Con
                else:
                    agree_before = 'Tied'
                if name == Pro and attitude['Agreed with after the debate'] == True:
                    agree_after = Pro
                elif name == Con and attitude['Agreed with after the debate'] == True:
                    agree_after = Con
                else:
                    agree_after = 'Tied'
                if (agree_before == Pro or agree_before == 'Tied') and agree_after == Con:
                    temp_con = temp_con + 1
                elif (agree_before == Con or agree_before == 'Tied') and agree_after == Pro:
                    temp_pro = temp_pro + 1
        # print(temp_pro,temp_con)
        if temp_pro != 0:
            percent.append(temp_pro/len(debate['votes']))
        if temp_con != 0:
            percent.append(temp_con/len(debate['votes']))

    sum = 0
    std = 0
    for p in percent:
        sum = sum + p
    average = sum / len(percent)
    for p in percent:
        std = std + (p - average) ** 2
    std = (std / len(percent)) ** (1 / 2)

    return percent, average, std

def count_percent_ppl_change_mind(debates,x):
    num = 0

    for key, debate in debates.items():
        temp_pro = 0
        temp_con = 0
        Pro = debate['participant_1_name']
        Con = debate['participant_2_name']
        for user in debate['votes']:
            for name, attitude in user['votes_map'].items():
                if name == Pro and attitude['Agreed with before the debate'] == True:
                    agree_before = Pro
                elif name == Con and attitude['Agreed with before the debate'] == True:
                    agree_before = Con
                else:
                    agree_before = 'Tied'
                if name == Pro and attitude['Agreed with after the debate'] == True:
                    agree_after = Pro
                elif name == Con and attitude['Agreed with after the debate'] == True:
                    agree_after = Con
                else:
                    agree_after = 'Tied'
                if (agree_before == Pro or agree_before == 'Tied') and agree_after == Con:
                    temp_con = temp_con + 1
                elif (agree_before == Con or agree_before == 'Tied') and agree_after == Pro:
                    temp_pro = temp_pro + 1
        # print(temp_pro,temp_con)
        if temp_pro / len(debate['votes']) > x:
            num = num + 1
        if temp_con / len(debate['votes']) > x:
            num = num + 1
        # if temp_pro > x:
        #     num = num + 1
        # if temp_con > x:
        #     num = num + 1

    return num

def count_percent_convincing_arg(debates,x):
    num = 0

    for key, debate in debates.items():
        temp_pro = 0
        temp_con = 0
        Pro = debate['participant_1_name']
        Con = debate['participant_2_name']
        for user in debate['votes']:
            for name, attitude in user['votes_map'].items():
                if name == Pro and attitude['Made more convincing arguments'] == True:
                    temp_pro = temp_pro + 1
                elif name == Con and attitude['Made more convincing arguments'] == True:
                    temp_con = temp_con + 1

        if temp_pro > x:
            num = num + 1
        if temp_con > x:
            num = num + 1

    return num

def filter_tied_argument(debates):
    debate_generate = {}

    for key,debate in debates.items():
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
        if temp_pro != temp_con:
            debate_generate[key] = debate

    return debate_generate

def filter_tied_argument_less_than_x(debates,x):
    debate_generate = {}

    for key,debate in debates.items():
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
        if temp_pro - temp_con > x or temp_con - temp_pro > x:
            debate_generate[key] = debate

    return debate_generate


def avg_votes_convincing_argument(debates):
    vote = []
    for key,debate in debates.items():
        vote.append(len(debate['votes']))

    return vote



if __name__ == '__main__':

    debates = {}
    for i in range(1,77029):
        if os.path.exists("./data/" + str(i) + '.json'):
            with open("./data/" + str(i) + '.json','r') as f:
                debates[i] = json.load(f)
            f.close()
        else:
            print(i)

    debates = filter_no_arg(debates)
    debates = filter_no_text(debates)
    debates = filter_none_arg(debates)

    print("debate_num:",len(debates))

    debate_points = filter_tied(debates)
    print("filter_out_tied:",len(debate_points))


    debate_points = filter_points_less_than_x(debate_points,3)
    print("filter_out_similar_points:",len(debate_points))

    debate_argument = filter_tied_argument(debates)
    print("filter_out_tied_argument:",len(debate_argument))

    debate_argument = filter_tied_argument_less_than_x(debate_argument, 1)
    print("argument less than 1:",len(debate_argument))

    vote = avg_votes_convincing_argument(debate_argument)
    num_bins = 20
    n, bins, patches = plt.hist(vote, num_bins, facecolor='blue',alpha=0.5)
    plt.xlabel('number of votes in one debate')
    plt.ylabel('Count')
    plt.show()

    debate_attitude = filter_no_one_changes_mind(debates)
    print(len(debate_attitude))

    num,avg,std = count_avg_ppl_change_mind(debate_attitude)

    print("avg_ppl",avg,std)

    count = count_ppl_change_mind_over_x(debate_attitude, 2)

    print("change_diff",count)

    percent,avg,std = count_avg_percent_ppl_change_mind(debate_attitude)

    print("avg_percent",avg,std)

    num = count_percent_ppl_change_mind(debate_attitude,0.5)

    print("percent",num)


    with open("data_all_points.json",'w') as f:
        json.dump(debate_points, f)
    f.close()

    with open("data_all_argument.json","w") as f:
        json.dump(debate_argument, f)
    f.close()

    with open("data_all_attitude.json","w") as f:
        json.dump(debate_attitude, f)
    f.close()

    with open("data_all.json","w") as f:
        json.dump(debates, f)
