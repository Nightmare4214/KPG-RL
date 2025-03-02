import matplotlib
import matplotlib.pyplot as plt
import data
import utils
from keypointguide_OT.linearprog import lp
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

num = 20
source_, target_ = data.get_data(num)

source = np.vstack(source_)
target = np.vstack(target_)
plt.xticks([])
plt.yticks([])
plt.tight_layout()

p = np.ones(len(source))/len(source)
q = np.ones(len(target))/len(target)
C = utils.cost_matrix(source, target)
C = C/C.max()
pi = lp(p, q, C)
source_transport = pi@target/p.reshape((-1, 1))
for i in range(len(source)):
    plt.plot([source[i][0], source_transport[i][0]], [source[i][1], source_transport[i][1]],
             '-', color="grey", linewidth=0.5)
s = ["+", "o", "^"]
for i in range(3):
    plt.plot(source_[i][:, 0], source_[i][:, 1], 'b{}'.format(
        s[i]), markersize=10, markerfacecolor="white")
    plt.plot(target_[i][:, 0], target_[i][:, 1], 'g{}'.format(
        s[i]), markersize=10, markerfacecolor="white")
if not os.path.exists("figure"):
    os.makedirs("figure")

labels = [0]*num + [1]*num + [2]*num
labels = np.array(labels)
pred = np.argmax(pi, axis=1)
pred = labels[pred]
acc = np.mean(labels == pred)

plt.text(-3.8, -0.5,
         "Matching\naccuracy: {:.1f}%".format(acc*100), fontsize=22)
plt.savefig("figure/KP.pdf")
plt.show()
