import os
import sys
import numpy as np
import pandas as pd

TPS_DIR = './data'


TP_file = os.path.join(TPS_DIR, 'train_triplets.txt')


tp = pd.read_table(TP_file, header=None, names=['uid', 'sid', 'count'])


MIN_USER_COUNT = 20
MIN_SONG_COUNT = 200



def split_train_test_proportion(data, test_prop=0.2):
    data_grouped_by_user = data.groupby('uid')
    tr_list, te_list = list(), list()

    np.random.seed(98765)

    for i, (_, group) in enumerate(data_grouped_by_user):
        n_items_u = len(group)

        if n_items_u >= MIN_USER_COUNT:
            idx = np.zeros(n_items_u, dtype='bool')
            idx[np.random.choice(n_items_u, size=int(test_prop * n_items_u), replace=False).astype('int64')] = True

            tr_list.append(group[np.logical_not(idx)])
            te_list.append(group[idx])
        else:
            tr_list.append(group)

        if i % 1000 == 0:
            print("%d users sampled" % i)
            sys.stdout.flush()

    data_tr = pd.concat(tr_list)
    data_te = pd.concat(te_list)

    return data_tr, data_te


def get_count(tp, id):
    playcount_groupbyid = tp[[id, 'count']].groupby(id, as_index=False)
    count = playcount_groupbyid.size()
    return count


def numerize(tp, profile2id, show2id):
    uid = map(lambda x: profile2id[x], tp['uid'])
    sid = map(lambda x: show2id[x], tp['sid'])
    return pd.DataFrame(data={'uid': list(uid), 'sid': list(sid)}, columns=['uid', 'sid'])


def filter_triplets(tp, min_uc=MIN_USER_COUNT, min_sc=MIN_SONG_COUNT):

    songcount = get_count(tp, 'sid')
    tp = tp[tp['sid'].isin(songcount.index[songcount >= min_sc])]

    usercount = get_count(tp, 'uid')
    tp = tp[tp['uid'].isin(usercount.index[usercount >= min_uc])]

    usercount, songcount = get_count(tp, 'uid'), get_count(tp, 'sid')
    return tp, usercount, songcount


tp, usercount, songcount = filter_triplets(tp)
raw_data = tp
sparsity_level = float(tp.shape[0]) / (usercount.shape[0] * songcount.shape[0])
print ("After filtering, there are %d triplets from %d users and %d songs (sparsity level %.3f%%)" % (tp.shape[0],
                                                                                                      usercount.shape[0],
                                                                                                      songcount.shape[0],
                                                                                                      sparsity_level * 100))


unique_uid = usercount.index

np.random.seed(98765)

n_users = unique_uid.size

n_heldout_users = 50000
tr_users = unique_uid[:(n_users - n_heldout_users * 2)]
vd_users = unique_uid[(n_users - n_heldout_users * 2): (n_users - n_heldout_users)]
te_users = unique_uid[(n_users - n_heldout_users):]

train_plays = raw_data.loc[raw_data['uid'].isin(tr_users)]
unique_sid = pd.unique(train_plays['sid'])
show2id = dict((sid, i) for (i, sid) in enumerate(unique_sid))
profile2id = dict((pid, i) for (i, pid) in enumerate(unique_uid))

pro_dir = os.path.join(TPS_DIR, 'pro_sg3')
if not os.path.exists(pro_dir):
    os.makedirs(pro_dir)
with open(os.path.join(pro_dir, 'unique_sid.txt'), 'w') as f:
    for sid in unique_sid:
        f.write('%s\n' % sid)
vad_plays = raw_data.loc[raw_data['uid'].isin(vd_users)]
vad_plays = vad_plays.loc[vad_plays['sid'].isin(unique_sid)]
vad_plays_tr, vad_plays_te = split_train_test_proportion(vad_plays)
test_plays = raw_data.loc[raw_data['uid'].isin(te_users)]
test_plays = test_plays.loc[test_plays['sid'].isin(unique_sid)]
test_plays_tr, test_plays_te = split_train_test_proportion(test_plays)

# Save the data into (user_index, item_index) format
train_data = numerize(train_plays, profile2id, show2id)
train_data.to_csv(os.path.join(pro_dir, 'train.csv'), index=False)
vad_data_tr = numerize(vad_plays_tr, profile2id, show2id)
vad_data_tr.to_csv(os.path.join(pro_dir, 'validation_tr.csv'), index=False)
vad_data_te = numerize(vad_plays_te, profile2id, show2id)
vad_data_te.to_csv(os.path.join(pro_dir, 'validation_te.csv'), index=False)
test_data_tr = numerize(test_plays_tr, profile2id, show2id)
test_data_tr.to_csv(os.path.join(pro_dir, 'test_tr.csv'), index=False)
test_data_te = numerize(test_plays_te, profile2id, show2id)
test_data_te.to_csv(os.path.join(pro_dir, 'test_te.csv'), index=False)

