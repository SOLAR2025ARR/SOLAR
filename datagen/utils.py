import sys
import copy
import random
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Queue

# sampler for batch generation
def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t

def sample_function(user_train, usernum, itemnum, batch_size, maxlen, result_queue, SEED):
    def sample(uid):

        # uid = np.random.randint(1, usernum + 1)
        while len(user_train[uid]) <= 1: uid = np.random.randint(1, usernum + 1)

        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        nxt = user_train[uid][-1]
        idx = maxlen - 1

        ts = set(user_train[uid])
        for i in reversed(user_train[uid][:-1]):
            seq[idx] = i
            pos[idx] = nxt
            if nxt != 0: neg[idx] = random_neq(1, itemnum + 1, ts)
            nxt = i
            idx -= 1
            if idx == -1: break

        return (uid, seq, pos, neg)

    np.random.seed(SEED)
    uids = np.arange(1, usernum+1, dtype=np.int32)
    counter = 0
    while True:
        if counter % usernum == 0:
            np.random.shuffle(uids)
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample(uids[counter % usernum]))
            counter += 1
        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                      usernum,
                                                      itemnum,
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue,
                                                      np.random.randint(2e9)
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()


# train/val/test data generation
def data_partition(fname, test_ratio = 0.2, val_ratio = 0.1):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = defaultdict(list)
    user_valid = defaultdict(list)
    user_test = defaultdict(list)
    # assume user/item index starting from 1
    f = open('../data/%s.txt' % fname, 'r')
    read_title = False
    for line in f:
        if not read_title:
            read_title = True
            usernum, itemnum = list(map(int, line.rstrip().split(' ')))
            continue
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        User[u].append(i)

    all_users = list(User.keys())
    random.seed(42) # set seed for reproducability
    random.shuffle(all_users)
    num_users = len(all_users)
    num_val = int(num_users * val_ratio)
    num_test = int(num_users * test_ratio)

    val_users = set(all_users[:num_val])
    test_users = set(all_users[num_val:num_val + num_test])
    train_users = set(all_users[num_val + num_test:])

    for user in train_users:
        user_train[user] = User[user]
    for user in val_users:
        user_valid[user] = User[user]
    for user in test_users:
        user_test[user] = User[user]

    return [user_train, user_valid, user_test, usernum, itemnum]

# evaluate on test set
def evaluate(model, dataset, args):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0

    HT_1 = 0.0
    NDCG_1 = 0.0

    HT_5 = 0.0
    NDCG_5 = 0.0

    test_users = set(test.keys())
    if len(test_users) > 10000:
        users = random.sample(test_users, 10000)
    else:
        users = test_users

    for u in users:
        if len(test[u]) < 1:
            continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1

        # fill the sequence with the most recent interactions
        for i in reversed(test[u][:-1]):
            seq[idx] = i
            idx -= 1
            if idx == -1:
                break

        rated = set(test[u][:-1])
        rated.add(0)
        item_idx = [test[u][-1]]  # use the last item in the test set as the target
        for idx in range(1, itemnum + 1):
            if idx not in rated:
                item_idx.append(idx)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if rank < 1:
            HT_1 += 1
            NDCG_1 += 1 / np.log2(rank + 2)
        if rank < 5:
            HT_5 += 1
            NDCG_5 += 1 / np.log2(rank + 2)

        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()
    print(f"HT@1: {HT_1 / valid_user}, NDCG@1: {NDCG_1 / valid_user}")
    print(f"HT@5: {HT_5 / valid_user}, NDCG@5: {NDCG_5 / valid_user}")
    return NDCG / valid_user, HT / valid_user


# evaluate on val set
def evaluate_valid(model, dataset, args):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0

    valid_users = set(valid.keys())
    if len(valid_users) > 10000:
        users = random.sample(valid_users, 10000)
    else:
        users = valid_users

    for u in users:
        if len(valid[u]) < 1:
            continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1

        # fill the sequence with the most recent interactions
        for i in reversed(valid[u][:-1]):
            seq[idx] = i
            idx -= 1
            if idx == -1:
                break

        rated = set(valid[u][:-1])
        rated.add(0)
        item_idx = [valid[u][-1]]  # use the last item in the test set as the target
        for _ in range(10000):
            t = np.random.randint(1, itemnum + 1)
            while t in rated:
                t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user