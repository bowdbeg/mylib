import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties

font_prop = FontProperties(fname="/usr/share/fonts/opentype/ipaexfont-gothic/ipaexg.ttf")
font_prop = FontProperties(fname="/usr/share/fonts/truetype/migmix/migmix-1p-regular.ttf")
plt.rcParams["font.family"] = font_prop.get_name()
plt.rcParams["font.size"] = 10

# names = ["Wenら  +GIEE", "Zhaoら +GIEE"]
names = ["各ステップ"]
dev = [
    [82.1, 82.05, 82.52, 82.49, 82.33, 82.10],
    [81.1, 81.14, 81.33, 81.34, 80.27, 80.27],
]
test = [[80.7, 80.66, 80.94, 80.21, 80.08, 79.44], [78.2, 78.25, 78.44, 78.41, 78.28, 78.28]]
ensemble_dev = [82.5, 81.4]
ensemble_test = [81.1, 78.5]

ensemble_itr_dev = [82.1, 82.4, 82.3, 82.4, 82.4, 82.5, 82.5]
ensemble_itr_test = [80.7, 80.7, 80.6, 81.0, 80.8, 81.1, 80.8]

# colors = ["r", "b"]
colors = ["b", "r"]
iteration = np.arange(0, 6, dtype=int)
ymin = float("inf")
ymax = -float("inf")

# plt.plot(list(range(0, 7)), ensemble_itr_dev, c="r", label="アンサンブル (開発)", ls="--", marker=".", linewidth=0.5)
# plt.plot(list(range(0, 7)), ensemble_itr_test, c="r", label="アンサンブル (評価)", ls="-", marker=".", linewidth=0.5)

w = 0.1
if True:
    plt.bar(np.array(list(range(0, 7)))-w/2, ensemble_itr_dev, color="b", label="アンサンブル    （開発）",width=w,alpha=0.3)
    plt.bar(np.array(list(range(0, 7)))+w/2, ensemble_itr_test, color="b", label="アンサンブル    （評価）",width=w,alpha=0.5)


for n, d, t, c, ensd, enst in zip(names, dev, test, colors, ensemble_dev, ensemble_test):
    if False:
        plt.plot(iteration, d, c=c, label=n + "（開発）", marker=".", ls="--",alpha=0.3)
        plt.plot(iteration, t, c=c, label=n + "（評価）", marker=".", ls="-",alpha=0.5)
    if True:
        plt.plot((-1, 7), (ensd, ensd), c=c, marker="", ls="--", label="", linewidth=0.5,alpha=0.3)
        plt.plot((-1, 7), (enst, enst), c=c, marker="", ls="-", label="", linewidth=0.5,alpha=0.5)
    ymin = min(min(d), ymin)
    ymin = min(min(t), ymin)
    ymax = max(max(d), ymax)
    ymax = max(max(t), ymax)
    break


# Wenら\cite{wen-ji-2021-utilizing} & 81.7 & 82.1 & 80.7 & 82.5 & 81.1 \\
# Zhaoら\cite{zhao-etal-2021-effective} & 79.6 & 81.1 & 78.2 & 81.4 & 78.5 \\
# plot ensemble
plt.xlim((min(iteration) - 0.1, max(iteration) + 0.1 + 1))
plt.ylim((int(ymin), ymax + 0.3))
print(ymax)
tcks = list(range(int(ymin), int(ymax)))
tcks.extend(ensemble_dev)
tcks.extend(ensemble_test)
# plt.legend(loc=(0.62, 0.6), fontsize=10)
plt.legend(loc=(0.02, 0.15), fontsize=10)
plt.xlabel("反復回数", fontsize=14)
plt.ylabel("F値", fontsize=14)
plt.xticks(range(0, 7), fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
# plt.savefig("a.pdf")
# plt.savefig("iterationwise.png",dpi=720)
plt.savefig("iteration.png",dpi=720)
