import random

from torch.utils.data import Dataset



def sample_negs(num_item, pos, k):
    negs = []
    for _ in range(k):
        neg = random.randint(0, num_item-1)
        while neg in pos + negs:
            neg = random.randint(0, num_item-1)
        negs.append(neg)

    return negs



class TrainSampler(Dataset):
    def __init__(self, data, num_item):
        self.data = {user: items[0] for user, items in data.items()}
        self.num_item = num_item
        self.num_data = sum(len(items) for items in self.data.values())

    def __getitem__(self, index):
        usr, pos, neg = self.triplets[index]

        return usr, pos, neg

    def __len__(self): return 4 * self.num_data

    def negative_sampling(self):
        self.triplets = []
        for usr, items in self.data.items():
            for pos in items:
                negs = sample_negs(self.num_item, items, 4)
                for neg in negs:
                    self.triplets.append((usr, pos, neg))



class EvalSampler(Dataset):
    def __init__(self, data, i):
        self.pairs = []
        for usr, items in data.items():
            for pos in items[i]:
                self.pairs.append((usr, pos))

    def __getitem__(self, index):
        usr, pos = self.pairs[index]

        return usr, pos

    def __len__(self): return len(self.pairs)
