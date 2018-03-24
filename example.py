from Mylibs.Tools import *
from Mylibs.Methods import *
from Mylibs.Experiments import *
from Mylibs.Word2Vec import *

import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import heapq
import progressbar
from mpl_toolkits.mplot3d import Axes3D
import resource

resource.setrlimit(resource.RLIMIT_NOFILE, (20480,1048576))
resource.getrlimit(resource.RLIMIT_NOFILE)

file_path, file_prefix = './PoiProcessed/', 'Gowalla'

# parameters
successive_time_constrain = 6
precision = 20
top_N = 10
embedding_dim = 200
learn_rate = 0.005
maxiter = 10
epsilon = 0.0001
win_len = 5
seq_length = 3

# csv to list of tuple(uid, lid, date, time) 
# and split to train tune test
tr_list, tu_list, te_list = dataset_splitting(file_path, file_prefix, 0.7, 0.1)

datalist = tr_list + tu_list + te_list

# filter out those poi not in the checkin list (which removed infrequent user)
#gen_pois_info(datalist, file_path, file_prefix)
# encode poi lat&lng to a binary string indicated path of tree
#gen_poi_encode(file_path,file_prefix,precision)

# generate dict of list: Dict[user] = [(context, current, after)]
nosuccessive_behavior, successive_behavior = gen_sequence_split_by_existt_context(tr_list,successive_time_constrain)
tune_nosuccessive_behavior, tune_successive_behavior = gen_successive_behavior(tu_list,successive_time_constrain)
test_nosuccessive_behavior, test_successive_behavior = gen_successive_behavior(te_list,successive_time_constrain)

# generate dict of list: Dict[loc] = [(context, current, after)]
geo_dict = poi_loader(file_path + file_prefix + '_PoiInfo.txt')
user_checkins = listtodict(tr_list, 0)

# count user checkin frequence: Dict[User][Loc] = "number"
user_checked_freq = user_checked(tr_list) 
alluser_checked_freq = user_checked(datalist)

# generate poi checkin freq : Dict[Loc] = "checkin count"
all_pois_freq_dict = gen_pois_frequency(datalist, geo_dict.keys())

# load poi info
poi_info = poi_loader(file_path + file_prefix + "_PoiInfo.txt")

print('w2v start...')
hparas = get_default_hparas()
hparas.tree_type = tree_type = "simularity"
hparas.simu_func = "cos"
hparas.simu_metric = "max"
hparas.max_process = 20

w2v_model = Word2Vec(hparas)
w2v_model.initial_word_dict(all_pois_freq_dict)
w2v_model.initial_user_dict(list(alluser_checked_freq.keys()))
w2v_model.train_all(successive_behavior, nosuccessive_behavior, user_checked_freq)

# generate user visited dict
user_visit = {}
for user in list(alluser_checked_freq.keys()):
    user_visit[user] = set()

for user in nosuccessive_behavior:
    for query in nosuccessive_behavior[user]:
        bef, target, aft = query
        user_visit[user].add(target)

for user in successive_behavior:
    for query in successive_behavior[user]:
        bef, target, aft = query
        user_visit[user].add(target)
    
    


case_count = 0
item_count = 0
match_count = 0


total_query = len(test_successive_behavior)
current_count = 0
pgbar = progressbar.ProgressBar(widgets=[progressbar.Percentage(),progressbar.Bar()],maxval=total_query).start()


for user in test_successive_behavior:
    for query in test_successive_behavior[user]:
        bef, cur, aft = query
        if len(set(aft) - user_visit[user]) == 0:
            continue
        top_k = w2v_model.get_top_k(user, query, poi_info)
        case_count += 1
        item_count += len(aft)
        match_count += len(set(aft) & set(top_k))
           
    time.sleep(0.001)
    current_count += 1
    pgbar.update(current_count)

pgbar.finish()
time.sleep(1)

pre = match_count / float(case_count * 10)
rec = match_count / float(item_count)

print('Precision a @ ' + str(10) + ' : ' + str(pre))
print('Recall a @ ' + str(10) + ' : ' + str(rec))


# iter + 20
w2v.model.hparas.max_iter = 20
w2v_model.train_all(successive_behavior, nosuccessive_behavior, user_checked_freq)

total_query = len(test_successive_behavior)
current_count = 0
pgbar = progressbar.ProgressBar(widgets=[progressbar.Percentage(),progressbar.Bar()],maxval=total_query).start()


for user in test_successive_behavior:
    for query in test_successive_behavior[user]:
        bef, cur, aft = query
        if len(set(aft) - user_visit[user]) == 0:
            continue
        top_k = w2v_model.get_top_k(user, query, poi_info)
        case_count += 1
        item_count += len(aft)
        match_count += len(set(aft) & set(top_k))
           
    time.sleep(0.001)
    current_count += 1
    pgbar.update(current_count)

pgbar.finish()
time.sleep(1)

pre = match_count / float(case_count * 10)
rec = match_count / float(item_count)

print('Precision a @ ' + str(10) + ' : ' + str(pre))
print('Recall a @ ' + str(10) + ' : ' + str(rec))



# iter + 20
w2v.model.hparas.max_iter = 20
w2v_model.train_all(successive_behavior, nosuccessive_behavior, user_checked_freq)

total_query = len(test_successive_behavior)
current_count = 0
pgbar = progressbar.ProgressBar(widgets=[progressbar.Percentage(),progressbar.Bar()],maxval=total_query).start()


for user in test_successive_behavior:
    for query in test_successive_behavior[user]:
        bef, cur, aft = query
        if len(set(aft) - user_visit[user]) == 0:
            continue
        top_k = w2v_model.get_top_k(user, query, poi_info)
        case_count += 1
        item_count += len(aft)
        match_count += len(set(aft) & set(top_k))
           
    time.sleep(0.001)
    current_count += 1
    pgbar.update(current_count)

pgbar.finish()
time.sleep(1)

pre = match_count / float(case_count * 10)
rec = match_count / float(item_count)

print('Precision a @ ' + str(10) + ' : ' + str(pre))
print('Recall a @ ' + str(10) + ' : ' + str(rec))
