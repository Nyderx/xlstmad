from sklearn.preprocessing import MinMaxScaler

from models.xlstmad_pred import XLSTMADPred
from models.xlstmad_pred_softdtw import XLSTMADSoftDTWPred
from models.xlstmad_rec_ad import XLSTMADRec
from models.xlstmad_rec_soft_dtw import XLSTMADRecSoftDTW


def run_xLSTM_reconstruct(data_train, data_test, window_size=50, lr=0.005, hidden_dim=40):
    clf = XLSTMADRec(feats=data_test.shape[1], hidden_dim=hidden_dim, window_size=window_size, lr=lr)
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    score = MinMaxScaler(feature_range=(0, 1)).fit_transform(score.reshape(-1, 1)).ravel()
    return score, clf.model


def run_xLSTM_reconstruct_soft_dtw(data_train, data_test, window_size=50, lr=0.001, hidden_dim=20):
    clf = XLSTMADRecSoftDTW(feats=data_test.shape[1], hidden_dim=hidden_dim, window_size=window_size, lr=lr)
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    score = MinMaxScaler(feature_range=(0, 1)).fit_transform(score.reshape(-1, 1)).ravel()
    return score, clf.model


def run_xLSTM_prediction_soft_dtw(data_train, data_test, window_size=25, lr=0.005, pred_len=5, hidden_dim=20):
    clf = XLSTMADSoftDTWPred(feats=data_test.shape[1], hidden_dim=hidden_dim, window_size=window_size, lr=lr,
                             pred_len=pred_len)
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    score = MinMaxScaler(feature_range=(0, 1)).fit_transform(score.reshape(-1, 1)).ravel()
    return score, clf.model


def run_xLSTM_prediction(data_train, data_test, window_size=50, lr=0.0008, pred_len=5, hidden_dim=20):
    clf = XLSTMADPred(feats=data_test.shape[1], hidden_dim=hidden_dim, window_size=window_size, lr=lr,
                      pred_len=pred_len)
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    score = MinMaxScaler(feature_range=(0, 1)).fit_transform(score.reshape(-1, 1)).ravel()
    return score, clf.model
