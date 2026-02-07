import torch



ks = [1, 5, 10, 20]



def recall(rank, user):
    unique_user, idx, num_pos = torch.unique(user, return_inverse=True, return_counts=True)

    recall_ks = []
    for k in ks:
        hit = (rank < k).float()
        num_hit = torch.zeros_like(unique_user).float()
        num_hit.scatter_add_(0, idx, hit)

        recall_k = torch.mean(num_hit / num_pos.float())
        recall_ks.append(recall_k.item())

    return recall_ks



def ndcg(rank, user):
    unique_user, idx, num_pos = torch.unique(user, return_inverse=True, return_counts=True)
    discounts = 1.0 / torch.log2(torch.arange(ks[-1]).float().to(rank.device) + 2.0)
    idcgs = torch.cumsum(discounts, 0)

    ndcg_ks = []
    for k in ks[1:]:
        dcgs = torch.where(rank < k, 1.0 / torch.log2(rank + 2.0), 0.0)
        dcg = torch.zeros_like(unique_user).float()
        dcg.scatter_add_(0, idx, dcgs)

        idcg = idcgs[torch.minimum(num_pos, torch.tensor(k)) - 1]

        ndcg_k = torch.mean(dcg / idcg)
        ndcg_ks.append(ndcg_k.item())

    return ndcg_ks



metrics = ['Recall@1', 'Recall@5', 'Recall@10', 'Recall@20', 'NDCG@5', 'NDCG@10', 'NDCG@20']
metric_funcs = [recall, ndcg]
