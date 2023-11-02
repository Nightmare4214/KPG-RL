import numpy as np
import utils
from sklearn.svm import SVC

domains = ["amazon", "dslr", "webcam"]
num_labeled = 1
seed = 0

Tasks = []
Accs = []

for source in domains:
    for target in domains:

        print("handling task {} --> {}".format(source, target))

        feat_s, label_s, feat_tl, label_tl, feat_tu, label_tu = utils.load_data(source, target, num_labeled,
                                                                                partial=True)
        dist = utils.cost_matrix(feat_tu, feat_tl)
        dist_min = np.min(dist, axis=1)
        out_index = np.argsort(dist_min)[len(feat_s):]
        pred_out = np.ones_like(label_tu)
        num_class = 10
        pred_out[out_index] = num_class + 1
        label_tu[label_tu >= num_class + 1] = num_class + 1
        acc_unk = np.mean(label_tu[label_tu == num_class + 1]
                          == pred_out[label_tu == num_class + 1])*100

        clf = SVC(gamma="auto")
        clf.fit(feat_tl, label_tl)

        in_index = np.argsort(dist_min)[len(feat_s):]
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
A2A:    40.00   71.00   51.17
A2D:    37.27   85.28   51.87
A2W:    29.09   82.68   43.04
D2A:    40.00   71.00   51.17
D2D:    37.27   85.28   51.87
D2W:    29.09   82.68   43.04
W2A:    40.00   71.00   51.17
W2D:    37.27   85.28   51.87
W2W:    29.09   82.68   43.04
avg:    35.45   79.65   48.69
'''
Tasks.append("avg")
Accs.append(np.mean(np.array(Accs), axis=0))
print("\ntask:\tacc_kno\tacc_unk\th_score")
for k in range(len(Tasks)):
    print("{:}:\t{:.2f}\t{:.2f}\t{:.2f}".format(
        Tasks[k], Accs[k][0], Accs[k][1], Accs[k][2]))
