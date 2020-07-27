class SegmentTree:
    def __init__(self, capacity, init_value=[0.0, float("inf")]):
        self.capacity = capacity
        self.tree = [init_value for _ in range(2 * capacity)]

    def sum(self):
        return self.tree[1][0]

    def min(self):
        return self.tree[1][1]

    def retrieve(self, upperbound):
        idx = 1
        while idx < self.capacity:
            left = 2 * idx
            right = left + 1
            if self.tree[left][0] > upperbound:
                idx = left
            else:
                upperbound -= self.tree[left][0]
                idx = right
        return idx - self.capacity

    def __setitem__(self, idx, val):
        idx += self.capacity
        self.tree[idx] = [val, val]

        idx //= 2
        while idx >= 1:
            self.tree[idx] = [self.tree[2 * idx][0] + self.tree[2 * idx + 1][0],
                              min(self.tree[2 * idx][1], self.tree[2 * idx + 1][1])]
            idx //= 2

    def __getitem__(self, idx):
        return self.tree[self.capacity + idx][0]


if __name__ == '__main__':
    segment_tree = SegmentTree(8)
    for i in range(5):
        segment_tree[i] = 1.0
        print(segment_tree.tree)

