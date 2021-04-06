from dependency import *

def data_processing(file,criterion):
    debates = {}
    with open(file,'r') as f:
        debates = json.load(f)
    f.close()

    n = len(debates)
    print("Debates:",n)
    Y = np.zeros((n*2,1))
    i = 0
    majority = 0

    if criterion == 'points':
        for key,debate in debates.items():
            if debate['participant_1_position'] == 'Pro' and debate['participant_1_status'] == 'Winning':
                Y[i,0] = 1
            elif debate['participant_2_position'] == 'Con' and debate['participant_2_status'] == 'Winning':
                Y[i+1,0] = 1
                majority = majority + 1
            i = i + 2
        print("majority:",majority)

    elif criterion == 'attitude':
        for key,debate in debates.items():
            if debate['participant_1_position'] != 'Pro':
                print("error")
            Pro_count = 0
            Con_count = 0
            for user in debate['votes']:
                agree_before = 'None'
                agree_after = 'None'
                for name,attitude in user['votes_map'].items():
                    if 'Agreed with after the debate' in attitude and 'Agreed with before the debate' in attitude:
                        if attitude['Agreed with after the debate'] == True:
                            agree_after = name
                        if attitude['Agreed with before the debate'] == True:
                            agree_before = name
                    else:
                        print(key)
                # print("agree before:",agree_before)
                # print("agree after:",agree_after)
                if agree_after != agree_before and agree_after == debate['participant_1_name']:
                    Pro_count = Pro_count + 1
                elif agree_after != agree_before and agree_after == debate['participant_2_name']:
                    Con_count = Con_count + 1
                elif agree_after == 'Tie' and agree_before == debate['participant_1_name']:
                    print("Pro->Tie")
                    Con_count = Con_count + 1
                elif agree_after == 'Tie' and agree_before == debate['participant_2_name']:
                    print("Con->Tie")
                    Pro_count = Pro_count + 1
            if Pro_count > 0:
                Y[i,0] = 1
                majority = majority + 1
            if Con_count > 0:
                Y[i+1,0] = 1
                majority = majority + 1
            i = i + 2
        print("majority",majority)

    else:
        print("Criterion Input error")

    return debates,Y

def data_processing_for_RNN(file,criterion):

    with open(file,'r') as f:
        debates = json.load(f)
    f.close()

    n = len(debates)
    print("Debates:", n)
    Y = np.zeros((n * 2, 1))

    if criterion == 'points':
        i = 0
        for key,debate in debates.items():
            if debate['participant_1_position'] == 'Pro' and debate['participant_1_status'] == 'Winning':
                Y[i,0] = 1
            elif debate['participant_2_position'] == 'Con' and debate['participant_2_status'] == 'Winning':
                Y[i,0] = 0
            i = i + 1

    return debates,Y
