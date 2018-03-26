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

# count user checkin frequence: Dict[User][Loc] = "number"
user_checked_freq = user_checked(tr_list) 
alluser_checked_freq = user_checked(datalist)

# filter out those poi not in the checkin list (which removed infrequent user)
#gen_pois_info(datalist, file_path, file_prefix)
# encode poi lat&lng to a binary string indicated path of tree
#gen_poi_encode(file_path,file_prefix,precision)

# generate dict of list: Dict[user] = [(context, current, after)]
nosuccessive_behavior, successive_behavior = gen_sequence_split_by_exist_context(tr_list,successive_time_constrain)
tu_nosuccessive_behavior, tu_successive_behavior = gen_sequence_split_by_exist_context(tu_list,successive_time_constrain)
tune_successive_behavior = gen_testing(tr_list          , tu_list,successive_time_constrain, alluser_checked_freq)
test_successive_behavior = gen_testing(tr_list + tu_list, te_list,successive_time_constrain, alluser_checked_freq)
#tune_nosuccessive_behavior, tune_successive_behavior = gen_successive_behavior(tu_list,successive_time_constrain)
#test_nosuccessive_behavior, test_successive_behavior = gen_successive_behavior(te_list,successive_time_constrain)

# generate dict of list: Dict[loc] = [(context, current, after)]
geo_dict = poi_loader(file_path + file_prefix + '_PoiInfo.txt')
user_checkins = listtodict(tr_list, 0)


# generate poi checkin freq : Dict[Loc] = "checkin count"
all_pois_freq_dict = gen_pois_frequency(datalist, geo_dict.keys())

# load poi info
poi_info = poi_loader(file_path + file_prefix + "_PoiInfo.txt")

res_file = open('./simu_results.csv', 'w')
res_file.write("max_iter,learning_rate,window_size,tree_type,time_decay,alpha,beta,simu_bound,pre@10,recall@10\n")

hparas = get_default_hparas()
iter_list = [10,20,30,40]# [10,20,30]
window_size_list = [1,3,5] #[1,2,3]
simu_bound_list = [0.8,0.7,0.6,0.5]
hparas.decay = 1
hparas.max_iter = 10
hparas.tree_type = "simularity"
hparas.simu_func = "cos"
hparas.simu_metric = "max"

for simu_bound in simu_bound_list:
    hparas.merge_bound = simu_bound
    for ws_item in window_size_list:
        hparas.window_size = ws_item
        w2v_model = Word2Vec(hparas)
        w2v_model.initial_word_dict(all_pois_freq_dict)
        w2v_model.initial_user_dict(list(alluser_checked_freq.keys()))
        
        for iter_item in iter_list:
            w2v_model.train_all(successive_behavior, nosuccessive_behavior, user_checked_freq)
                
            # tuning part start
            max_alpha = 0
            max_beta = 0
            max_pre = 0
    
            tunebar = progressbar.ProgressBar(widgets=[progressbar.Percentage(),progressbar.Bar()],maxval=11).start()
            a_list = [x/10.0 for x in range(0,11,1)]
            for alpha in a_list:
                beta = 1 - alpha
                
                case_count = 0
                item_count = 0
                match_count = 0 
                total_query = len(tune_successive_behavior)
                current_count = 0
                
                for user in tune_successive_behavior:
                    for query in tune_successive_behavior[user]:
                        bef, cur, aft = query
                        
                        top_k = w2v_model.get_top_k(user, query, poi_info, 10, alpha, beta)
                        case_count += 1
                        item_count += len(aft)
                        match_count += len(set(aft) & set(top_k))
                        
                    time.sleep(0.001)
                    current_count += 1
                
                time.sleep(0.1)
                
                pre = match_count / float(case_count * 10)
                rec = match_count / float(item_count)
            
                print('Tuning: A = ' + str(alpha) )
                print('Precision a @ ' + str(10) + ' : ' + str(pre) + 'Recall a @ ' + str(10) + ' : ' + str(rec))
                if pre > max_pre:
                    max_alpha = alpha
                    max_beta  = beta
                    max_pre   = pre
                tunebar.update(alpha*10+1)
            tunebar.finish()
            # tuning part end
            
            # traing tuning part
            w2v_model.train_all(tu_successive_behavior, tu_nosuccessive_behavior)
                
            # testing part start
            case_count = 0
            item_count = 0
            match_count = 0 
            total_query = len(test_successive_behavior)
            current_count = 0
            pgbar = progressbar.ProgressBar(widgets=[progressbar.Percentage(),progressbar.Bar()],maxval=total_query).start()
            
            for user in test_successive_behavior:
                for query in test_successive_behavior[user]:
                    bef, cur, aft = query
    
                    top_k = w2v_model.get_top_k(user, query, poi_info, 10, max_alpha, max_beta)
                    case_count += 1
                    item_count += len(aft)
                    match_count += len(set(aft) & set(top_k))
                    
                time.sleep(0.001)
                current_count += 1
                pgbar.update(current_count)
            
            pgbar.finish()
            time.sleep(0.1)
            
            pre = match_count / float(case_count * 10)
            rec = match_count / float(item_count)
            
            res_file.write(str(iter_item)+","+str(hparas.learn_rate)+","+str(hparas.window_size)+",")
            res_file.write(hparas.tree_type+","+str(hparas.decay)+","+str(max_alpha)+","+str(max_beta)+","+str(simu_bound)+",")
            res_file.write(str(pre)+","+str(rec)+"\n")
            print('At iter_ration # ' + str(iter_item) + " and window_size " + str(hparas.window_size))
            print('Precision a @ ' + str(10) + ' : ' + str(pre))
            print('Recall a @ ' + str(10) + ' : ' + str(rec))
            # testing part end
res_file.close()
print("fin")
