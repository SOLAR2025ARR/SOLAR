from utils import *
from collections import defaultdict
from models import SASRec
import torch
import pandas as pd
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Export inference result")
    parser.add_argument('--dataset', type=str, required=True, help='dataset name')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--topk', type=int, default=20, help='topk')
    parser.add_argument('--device', type=str, default='cuda', help='device')
    parser.add_argument('--ckpt', type=str, required=True, help='checkpoint path')
    parser.add_argument('--hidden_units', type=int, default=512, help='hidden units')
    parser.add_argument('--num_blocks', type=int, default=2, help='num blocks')
    parser.add_argument('--num_heads', type=int, default=2, help='num heads')
    parser.add_argument('--maxlen', type=int, default=100, help='maxlen')
    parser.add_argument('--dropout_rate', type=float, default=0.2, help='dropout rate')
    parser.add_argument('--keep_test_only', action='store_true', help='keep test only')
    return parser.parse_args()

def main():
    args = parse_args()

    [user_train, user_valid, user_test, usernum, itemnum] = data_partition(args.dataset)
    all_users = defaultdict(list)
    all_users.update(user_train)
    all_users.update(user_valid)
    all_users.update(user_test)

    if args.keep_test_only:
        all_users = user_test


    model = SASRec(usernum, itemnum, args.maxlen, args.hidden_units, args.dropout_rate, args.num_blocks, args.num_heads, args.device).to(args.device)

    model.load_state_dict(torch.load(args.ckpt))

    model.eval()

    result = []

    for u in all_users:
        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        history = all_users[u][:-1]
        target = all_users[u][-1]
        for i in reversed(history):
            seq[idx] = i
            idx -= 1
            if idx == -1:
                break
        seq = torch.tensor(seq, device=args.device) # shape: (maxlen, )
        seq = seq.unsqueeze(0) # shape: (1, maxlen)
        items = range(1, itemnum + 1)
        scores = model.predict([u], seq, items) # shape: (1, itemnum)
        # set scores of already interacted items to -inf
        scores[0, [i - 1 for i in history]] = float('-inf') # shape: (1, itemnum)
        _, indices = torch.topk(scores, args.topk) # shape: (1, topk)
        indices = indices.cpu()
        # save user_id, history, topk items
        result.append((u, history, [idx + 1 for idx in indices[0].tolist()], target))

    # save to dataframe
    df = pd.DataFrame(result, columns=['user', 'history', 'topk_items', 'target'])
    df.to_csv(f'inference_result_{args.dataset}_sasrec.csv', index=False)

if __name__ == '__main__':
    main()