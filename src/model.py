from dependency import *

def svm_train_validation(X,Y):

    clf = svm.SVC(gamma='scale',kernel='rbf',probability=True)

    P = []
    kf = KFold(n_splits=5, shuffle=False)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        clf.fit(X_train, np.ravel(Y_train))
        correct = 0
        predict = 0
        for i in range(0,len(X_test),2):
            if i < len(X_test)-1:
                Y_pred1 = clf.predict_proba([X_test[i,:]])
                Y_pred2 = clf.predict_proba([X_test[i+1,:]])
                if Y_pred1[0][0] >= Y_pred2[0][0]:
                    Y_pred1 = 0
                    Y_pred2 = 1
                else:
                    Y_pred1 = 1
                    Y_pred2 = 0
                if Y_pred1 == Y_test[i,0]:
                    correct = correct + 2
                    predict = predict + 2
                else:
                    predict = predict + 2
        P.append(correct/predict)

    return P


def LR_train_validation(X,Y):

    clf = LogisticRegressionCV(Cs = 1, cv = 3,solver='lbfgs',multi_class='multinomial',max_iter=1000)

    P = []
    kf = KFold(n_splits=5, shuffle=False)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        correct = 0
        predict = 0
        clf.fit(X_train, np.ravel(Y_train))
        for i in range(0,len(X_test),2):
            if i < len(X_test)-1:
                Y_pred1 = clf.predict_proba([X_test[i,:]])
                Y_pred2 = clf.predict_proba([X_test[i+1,:]])
                if Y_pred1[0][0] >= Y_pred2[0][0]:
                    Y_pred1 = 0
                    Y_pred2 = 1
                else:
                    Y_pred1 = 1
                    Y_pred2 = 0
                if Y_pred1 == Y_test[i,0]:
                    correct = correct + 2
                    predict = predict + 2
                else:
                    predict = predict + 2

        P.append(correct / predict)

    return P

def LR_unbanlance_train_validation(X,Y):

    clf = LogisticRegressionCV(Cs=1, cv=2, solver='lbfgs', multi_class='multinomial', max_iter=1000)
    P = []
    balance = []
    kf = KFold(n_splits=5, shuffle=True)
    for train_index, test_index in kf.split(X):
        balance_count = 0
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        clf.fit(X_train, np.ravel(Y_train))
        correct = 0
        predict = 0
        for i in range(0, len(X_test)):
            Y_pred = clf.predict([X_test[i, :]])
            if Y_pred == 1:
                balance_count = balance_count + 1
            if Y_pred == Y_test[i,0]:
                correct = correct + 1
                predict = predict + 1
            else:
                predict = predict + 1
        P.append(correct / predict)
        balance.append(balance_count)

    return P,balance
