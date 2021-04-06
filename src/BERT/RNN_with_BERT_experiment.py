from RNN_with_BERT import *
import json
import pickle
import torch
import time
from tqdm import tqdm
import numpy as np
from extract_feature import *
import sys


if __name__ == '__main__':
    n = 13786
    hidden_unit = 768

    X_tmp = np.zeros((n,hidden_unit))

    with open('../data_all_argument.json', 'r') as f:
        debates = json.load(f)
    f.close()

    with open('output_data_BERT_argument_last_three_sentence_full.pickle','rb') as f:
        X_tmp = pickle.load(f)
    f.close()

    with open('dataforBERT_X_all_argument_last_three_sentence_full.pickle', 'rb') as f:
        data = pickle.load(f)
    f.close()

    with open('dataforBERT_all_argument.pickle', 'rb') as f:
        attribute = pickle.load(f)
    f.close()


    X = data['X']
    Y = data['Y']
    target_key = attribute['key']
    argument_feature = extract_argument_features('dataforBERT_X_all_argument_last_three_sentence_full.pickle')
    with open("argument_feature_rnn.pickle",'wb') as f:
        pickle.dump(argument_feature,f)
    f.close()

    argument_feature = np.zeros((13786,32))
    argument_feature = argument_feature_tmp[:,0:32]

    argument_feature = (argument_feature - np.mean(argument_feature,axis=0)) / np.std(argument_feature,axis=0)

    user_feature = extract_user_features('dataforBERT_X_all_argument_last_three_sentence_full.pickle')
    with open("user_feature_rnn.pickle",'wb') as f:
        pickle.dump(user_feature,f)
    f.close()

    user_feature = (user_feature - np.mean(user_feature,axis=0)) / np.std(user_feature,axis=0)
    text_b = None

    train_index = []
    validation_index = []
    test_index = []
    X_train = []
    X_validation = []
    X_test = []
    Y_train = []
    Y_validation = []
    Y_test = []

    X_train = X[:2200]
    X_validation = X[2200:2400]
    X_test = X[2400:]
    Y_train = Y[:2200]
    Y_validation = Y[2200:2400]
    Y_test = Y[2400:]

    num_epochs = 20
    mode = sys.argv[1]
    if mode == 'full':
        combine = 'rnn'
        model = combine_model()
        model.double()
        opt = torch.optim.Adagrad(model.parameters(),lr=0.005,weight_decay = 0.01)
    elif mode == 'linguistic':
        combine = 'none'
        model = model()
        model.double()
        opt = torch.optim.Adagrad(model.parameters(),lr=0.005,weight_decay = 0.01)

    minibatch_size = 10
    num_minibatches = len(X_train) // minibatch_size

    loss_list = []
    timer_list = []
    epoch_loss = []

    max_validation = 0
    max_epoch = 0
    max_test = 0
    test_acc = []

    start_training = time.time()
    for epoch in (range(num_epochs)):
        loss_sum = None
        # Training
        print("Training")
        # Put the model in training mode
        model.train()
        start_train = time.time()

        index = 0
        for group in tqdm(range(num_minibatches)):
            total_loss = None
            opt.zero_grad()

            for i in range(group * minibatch_size, (group + 1) * minibatch_size):
                if combine == 'rnn':
                    input_bert = X_tmp[index:index+len(X_train[i])]
                    input_arg = argument_feature[index:index+len(X_train[i])]
                    outputs = model(input_bert, input_arg)
                elif combine == 'logistic':
                    input_bert = X_tmp[index:index+len(X_train[i])]
                    input_user = user_feature[i]
                    outputs = model(input_bert, input_user)
                else:
                    input_seq = X_tmp[index:index+len(X_train[i])]
                    outputs = model(input_seq)
                index = index + len(X_train[i])
                label = torch.tensor([Y_train[i]])
                loss = model.compute_Loss(outputs.squeeze(1), label)
                # On the first gradient update
                if total_loss is None:
                    total_loss = loss
                else:
                    total_loss += loss
            timer_list.append(time.time() - start_training)
            loss_list.append(total_loss.data.cpu().numpy())
            total_loss.backward()
            opt.step()
            if loss_sum is None:
                loss_sum = loss
            else:
                loss_sum = loss_sum + total_loss
        epoch_loss.append(loss_sum.data.cpu().numpy())
        print(loss_sum)
        if epoch >= 1:
            if epoch_loss[epoch-1] - epoch_loss[epoch] < 0.0001:
                break
        print("Training time: {} for epoch {}".format(time.time() - start_train, epoch))


        # Evaluation
        print("Evaluation")
        # Put the model in evaluation mode
        model.eval()
        start_eval = time.time()


        predictions = 0
        correct = 0  # number of tokens predicted correctly
        index = 0
        for i in range(len(X_train)):
            if combine == 'rnn':
                output = model(X_tmp[index:index + len(X_train[i])],argument_feature[index:index + len(X_train[i])])
            elif combine == 'logistic':
                output = model(X_tmp[index:index+len(X_train[i])],user_feature[i])
            else:
                output = model(X_tmp[index:index + len(X_train[i])])
            index = index + len(X_train[i])
            if output.squeeze(1)[0][0].item() > output.squeeze(1)[0][1].item():
                pred = 0
            else:
                pred = 1
            if pred == Y_train[i]:
                correct = correct + 1
                predictions = predictions + 1
            else:
                predictions = predictions + 1
        accuracy = correct / predictions
        assert 0 <= accuracy <= 1
        log = "Evaluation time: {} for epoch {}, Training Accuracy: {}".format(time.time() - start_eval, epoch, accuracy)
        print(log)

        predictions = 0
        correct = 0  # number of tokens predicted correctly
        loss_validation = None
        for i in range(len(X_validation)):
            if combine == 'rnn':
                output = model(X_tmp[index:index+len(X_validation[i])],argument_feature[index:index+len(X_validation[i])])
            elif combine == 'logistic':
                output = model(X_tmp[index:index+len(X_validation[i])],user_feature[i+2200])
            else:
                output = model(X_tmp[index:index+len(X_validation[i])])
            if loss_validation is None:
                loss_validation = model.compute_Loss(output.squeeze(1), torch.tensor([Y_validation[i]]))
            else:
                loss_validation = loss_validation + model.compute_Loss(output.squeeze(1), torch.tensor([Y_validation[i]]))

            index = index + len(X_validation[i])
            if output.squeeze(1)[0][0] > output.squeeze(1)[0][1]:
                pred = 0
            else:
                pred = 1
            if pred == Y_validation[i]:
                correct = correct + 1
                predictions = predictions + 1
            else:
                predictions = predictions + 1
        accuracy = correct / predictions
        print(loss_validation)
        assert 0 <= accuracy <= 1
        if accuracy >= max_validation:
            torch.save(model.state_dict(), 'rnn_validation_last_three_sentence.model')
            max_validation = accuracy
            max_epoch = epoch
        log = "Evaluation time: {} for epoch {}, Accuracy: {}".format(time.time() - start_eval, epoch, accuracy)
        print(log)

        predictions = 0
        correct = 0  # number of tokens predicted correctly
        for i in range(len(X_test)):
            if combine == 'rnn':
                output = model(X_tmp[index:index + len(X_test[i])],argument_feature[index:index+len(X_test[i])])
            elif combine == 'logistic':
                output = model(X_tmp[index:index+len(X_test[i])],user_feature[i+2400])
            else:
                output = model(X_tmp[index:index + len(X_test[i])])
            index = index + len(X_test[i])
            if output.squeeze(1)[0][0] > output.squeeze(1)[0][1]:
                pred = 0
            else:
                pred = 1
            if pred == Y_test[i]:
                correct = correct + 1
                predictions = predictions + 1
            else:
                predictions = predictions + 1
        accuracy = correct / predictions
        test_acc.append(accuracy)
        assert 0 <= accuracy <= 1
        if accuracy >= max_test:
            torch.save(model.state_dict(), 'rnn_test_last_three_sentence.model')
            max_test = accuracy
        log = "Evaluation time: {} for epoch {}, Test Accuracy: {}".format(time.time() - start_eval, epoch, accuracy)
        print(log)

    print("Epoch: {}, Max Validation: {}, Test Accuracy: {}".format(max_epoch, max_validation, test_acc[max_epoch]))
