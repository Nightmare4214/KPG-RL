import numpy as np
import utils
from sklearn.svm import SVC
from keypointguide_POT.linearprog import lp

domains = ["amazon", "dslr", "webcam"]
num_labeled = 1
seed = 0

Tasks = []
Accs = []

for source in domains:
    for target in domains:
        print("\nhandling task {} --> {}".format(source, target))
        feat_s, label_s, feat_tl, label_tl, feat_tu, label_tu = utils.load_data(source, target, num_labeled,
                                                                                partial=True)

        # key point
        I = []
        J = []
        t = 0
        feat_sl = []
        for l in label_tl:
            I.append(t)
            J.append(t)
            fl = feat_s[label_s == l]
            feat_sl.append(np.mean(fl, axis=0))
            t += 1
        feat_sl = np.vstack(feat_sl)
        feat_s_ = np.vstack((feat_sl, feat_s))
        feat_t_ = np.vstack((feat_tl, feat_tu))
        Cs = utils.cost_matrix(feat_s_, feat_s_)
        Cs /= Cs.max()
        Ct = utils.cost_matrix(feat_t_, feat_t_)
        Ct /= Ct.max()
        p = np.ones(len(Cs))/len(Cs)
        q = np.ones(len(Ct))/len(Ct)
        C = utils.structure_metrix_relation(Cs, Ct, I, J)
        C = C/C.max()
        # mask
        M = np.ones_like(C)
        M[I, :] = 0
        M[:, J] = 0
        M[I, J] = 1

        # key point OT
        print("solving partial kpg-ot...")
        s = len(p)/len(q)
        thr = 1e-3
        p = p*s
        xi = C.max()
        C_ = np.vstack((C, xi * np.ones((1, len(q)))))
        b = np.ones(len(q))
        b[J] = 0
        M_ = np.vstack((M, b.reshape((1, -1))))
        p_ = np.hstack((p, np.sum(q)-s))

        pi_ = lp(p_, q, C_, M_)

        num_class = 10
        out_index = np.argwhere(pi_[-1, len(feat_tl):] > thr).reshape((-1,))
        pred_out = np.ones_like(label_tu)
        pred_out[out_index] = num_class + 1
        label_tu[label_tu >= num_class + 1] = num_class + 1
        acc_unk = np.mean(label_tu[label_tu == num_class + 1]
                          == pred_out[label_tu == num_class + 1])*100

        select_index = np.argwhere(pi_[-1, :] < thr).reshape((-1,))
        pi = pi_[:, select_index]
        pi = pi[:-1, :]
        feat_s_trans = pi@feat_t_[select_index]/p.reshape((-1, 1))
        feat_train = np.vstack((feat_tl, feat_s_trans[len(feat_tl):]))
        label_train = np.hstack((label_tl, label_s))

        print("train svm...")
        clf = SVC(gamma='auto')
        clf.fit(feat_train, label_train)
        in_index = np.argwhere(pi_[-1, len(feat_tl):] <= thr).reshape((-1,))
        pred = clf.predict(feat_tu[in_index])
        pred_out[in_index] = pred
        acc_kno = np.mean(label_tu[label_tu != num_class + 1]
                          == pred_out[label_tu != num_class + 1])*100
        h_score = 2 * acc_kno * acc_unk / (acc_kno + acc_unk)
        # print("acc_kno:{:.4f} \t acc_unk:{:.4f} \t h_score:{:.4f}".format(acc_kno, acc_unk, h_score))
        Tasks.append(source[0].upper() + "2" + target[0].upper())
        Accs.append([acc_kno, acc_unk, h_score])

'''
task:   acc_kno acc_unk h_score
A2A:    41.82   77.06   54.21
A2D:    48.18   74.89   58.64
A2W:    38.18   73.16   50.18
D2A:    52.73   82.68   64.39
D2D:    80.00   92.64   85.86
D2W:    73.64   89.18   80.67
W2A:    52.73   82.25   64.26
W2D:    78.18   90.91   84.07
W2W:    71.82   88.31   79.22
avg:    59.70   83.45   69.05
'''
Tasks.append("avg")
Accs.append(np.mean(np.array(Accs), axis=0))
print("\ntask:\tacc_kno\tacc_unk\th_score")
for k in range(len(Tasks)):
    print("{:}:\t{:.2f}\t{:.2f}\t{:.2f}".format(
        Tasks[k], Accs[k][0], Accs[k][1], Accs[k][2]))
