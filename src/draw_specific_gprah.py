# 101016jssc200704044
from relation_data import RelationData
from pathlib import Path

inputs = [
    # "/home/makino.15083/brat/data/olivetti/data_rel/devel",
    "/home/makino.15083/euro/brat/data/graph/tuning/0031/0123/devel",
    # '/home/makino.15083/euro/brat/data/graph/acl_survey/1111/0095/',
    # '/home/makino.15083/euro/brat/data/graph/acl_survey/0000/0087/',
    # "/home/makino.15083/euro/brat/data/graph/empty/0000/0101/devel/",
    # '/home/makino.15083/euro/brat/data/graph/acl_survey/0008/0064/',
    # "/home/makino.15083/brat/data/olivetti/pred_rel/devel",
]
# outs = ["gold", "ours", "empty", "rule"]
# outs = ["gold", "ours", "oneshot", "rule"]
# outs = ["gold", "itr-rule", "itr",'noitr-rule', "rule"]
outs = ['best_master']
output_dir = Path("/home/makino.15083/graphvis/101016jssc200704044")

pos_dic = {
    "prepared": [-1+0.05, -0.1],
    "air": [-1, -0.2],
    "SrMo1-xNixO4_": [-1+0.05, -0.3],
    "polycrystalline": [-1+0.05, -0.4],
    "solid-state reaction method": [2 / 9 * 3 - 1, 0.5],
    "mixed": [-1, 0.05],
    "MoO3": [-1, 0.15],
    "SrCO3": [-0.95, 0.25],
    "Ni": [-0.87, 0.15],
    "high-purity": [-1, 0.5],
    "powders": [-0.9, 0.45],
    "desired stoichiometry": [-0.8, 0.4],
    "Appropriate proportions": [-0.7, 0.35],
    "prefired": [2 / 9 * 1 - 1, -0.05],
    "h__": [2 / 9 * 1 - 1, 0.05],
    "24_": [2 / 9 * 1 - 1, 0.15],
    "C__": [2 / 9 * 1 - 1, -0.15],
    "900": [2 / 9 * 1 - 1, -0.25],
    "ground_": [2 / 9 * 2 - 1, 0.05],
    "powders_": [2 / 9 * 2 - 1, 0.15],
    "pelletized": [2 / 9 * 3 - 1, -0.05],
    "calcined": [2 / 9 * 4 - 1, 0.05],
    "h_": [2 / 9 * 4 - 1, 0.1],
    "24": [2 / 9 * 4 - 1, 0.25],
    "C_": [2 / 9 * 4 - 1, -0.1],
    "1000": [2 / 9 * 4 - 1 - 0.2, -0.3],
    "1100": [2 / 9 * 4 - 1, -0.3],
    "1200": [2 / 9 * 4 - 1 + 0.2, -0.3],
    "obtained_": [2 / 9 * 5 - 1, -0.05],
    "SrMo1-xNixO4": [2 / 9 * 5 - 1, 0.2],
    "compounds_": [2 / 9 * 5 - 1, 0.4],
    "ground": [2 / 9 * 6 - 1, 0.05],
    "compounds": [2 / 9 * 6 - 1, -0.15],
    "pressed": [2 / 9 * 7 - 1, -0.05],
    "pellets_": [2 / 9 * 7 - 1, 0.1],
    "mm": [2 / 9 * 7 - 1 - 0.2, 0.15],
    "thickness": [2 / 9 * 7 - 1 - 0.25, 0.3],
    "2": [2 / 9 * 7 - 1 - 0.15, 0.35],
    "mm_": [2 / 9 * 7 - 1 , 0.35],
    "diameter": [2 / 9 * 7 - 1 - 0.1, 0.4],
    "10": [2 / 9 * 7 - 1 , 0.45],
    "reduced": [2 / 9 * 8 - 1, 0.05],
    "C": [2 / 9 * 8 - 1 - 0.05, 0.2],
    "920": [2 / 9 * 8 - 1 - 0.05, 0.4],
    "h": [2 / 9 * 8 - 1 + 0.05, 0.2],
    "12": [2 / 9 * 8 - 1 + 0.05, 0.4],
    "pellets": [2 / 9 * 8 - 1 - 0.05, -0.1],
    "H2/Ar": [2 / 9 * 8 - 1 + 0.05, -0.2],
    "%": [2 / 9 * 8 - 1 , -0.3],
    "95": [2 / 9 * 8 - 1 , -0.4],
    "%_": [2 / 9 * 8 - 1 + 0.075, -0.3],
    "5": [2 / 9 * 8 - 1 + 0.075, -0.4],
    "obtained": [2 / 9 * 9 - 1, -0.05],
    "SrMo1-xNixO3": [2 / 9 * 9 - 1, 0.3],
}

data = RelationData(inputs[0], pattern="*.ann")
dat = data["101016jssc200704044"]
pkey = [e["entity"] for e in dat["entity"].values()]
while True:
    flag = False
    for i, e in enumerate(pkey):
        if pkey.count(e) > 1:
            pkey[i] = e + "_"
            flag = True
    if not flag:
        break
pos = [pos_dic[k] for k in pkey]

for idir, o in zip(inputs, outs):
    data = RelationData(idir, pattern="*.ann")
    data["101016jssc200704044"].vis_graph(pos=pos_dic, out=output_dir / (o + ".pdf"))
