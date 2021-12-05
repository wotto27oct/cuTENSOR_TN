import numpy as np
from scipy.stats import unitary_group
import time
import itertools
import cupy as cp
import sys

total_elapsed_start = time.time()

args = sys.argv

node_num = int(args[1])
alpha = float(args[2])
dim = int(args[3])
naive = int(args[4])
rnum = int(args[5])

cp_naive_list = []
cp_optimal_list = []

stream = cp.cuda.Stream.null

for r in range(rnum):
    print(f"{r}th calc")

    tensor_list = []
    cp_tensor_list = []
    index_list = [[] for i in range(node_num)]
    cp.random.seed(r)
    np.random.seed(r)
    str_idx = 97

    upper_bound = (node_num - 1) * alpha

    for i in range(node_num):
        for j in range(i, node_num):
            if i == j or len(index_list[i]) > upper_bound or len(index_list[j]) > upper_bound:
                continue
            if alpha > np.random.rand():
                index_list[i].append(chr(str_idx))
                index_list[j].append(chr(str_idx))
                str_idx += 1

        # at least 1 couple
        if len(index_list[i]) == 0:
            couple = (np.random.randint(1, node_num) + i) % node_num
            index_list[i].append(chr(str_idx))
            index_list[couple].append(chr(str_idx))
            str_idx += 1

    print(index_list)

    for i in range(node_num):
        tensor_list.append(np.random.normal(0.0, np.sqrt(10.0/node_num), tuple([dim for j in range(len(index_list[i]))])))
        tensor_list[i] = tensor_list[i].astype(np.float64)
        cp_tensor_list.append(cp.asarray(tensor_list[i]))

    cont_str = ""
    for i in range(node_num):
        if i != 0:
            cont_str += ","
        for j in index_list[i]:
            cont_str += j

    if naive == 1:
        start = stream.record()
        print(cont_str)
        output = cp.einsum(cont_str, *cp_tensor_list)
        end = stream.record()
        end.synchronize()
        elapsed_time = cp.cuda.get_elapsed_time(start, end) / 1000
        print(f"elapsed time by naive contraction: {elapsed_time}[s]")
        print(f"output by naive contraction: {output}")
        cp_naive_list.append(elapsed_time)

    class Node:
        def __init__(self, left, right, tensor, index, cost, name):
            self.parent = -1
            self.left = left
            self.right = right
            self.tensor = tensor
            self.index = index
            self.cost = cost
            self.name = name
            self.cont_str = None
        
        def calc_tensor(self):
            if self.tensor is not None:
                return self.tensor
            self.left.calc_tensor()
            self.right.calc_tensor()
            left_str = "".join([str(i) for i in self.left.index])
            right_str = "".join([str(i) for i in self.right.index])
            node_str = "".join([str(i) for i in self.index])
            self.cont_str = left_str + "," + right_str + "->" + node_str
            self.tensor = cp.einsum(self.cont_str, self.left.tensor, self.right.tensor, dtype=np.float64)
            return self.tensor
        
        def calc_expression(self):
            if self.left is None:
                return str(self.name)
            left = self.left.calc_expression()
            right = self.right.calc_expression()
            return "(" + left + "," + right + ")"

    mu_best = {}
    Q_best = {}

    def dp(N):
        if N in mu_best:
            return mu_best[N], Q_best[N]
        if len(N) == 1:
            mu_best[N] = 0
            N_idx = N[0]
            node = Node(None, None, cp_tensor_list[N_idx], index_list[N_idx], 0, str(N_idx))
            Q_best[N] = node
            return 0, node
        for i in range(1, len(N)//2 + 1):
            left_div_list = tuple(itertools.combinations(list(N), i))
            for left_div in left_div_list:
                right_div = tuple([j for j in list(N) if j not in left_div])
                L_cost, LQ = dp(left_div)
                R_cost, RQ = dp(right_div)
                common_index = set(LQ.index) & set(RQ.index)
                contract_cost = 2 ** (len(LQ.index) + len(RQ.index) - len(common_index))
                if N not in mu_best or mu_best[N] > L_cost + R_cost + contract_cost:
                    mu_best[N] = L_cost + R_cost + contract_cost
                    index = [x for x in dict.fromkeys(LQ.index + RQ.index) if (LQ.index + RQ.index).count(x) == 1]
                    Q_best[N] = Node(LQ, RQ, None, index, mu_best[N], None)
        return mu_best[N], Q_best[N]

    all = tuple([i for i in range(node_num)])
    start = stream.record()
    dp(all)
    end = stream.record()
    end.synchronize()
    print(f"calculatoin time for optimal contraction tree: {cp.cuda.get_elapsed_time(start, end) / 1000}[s]")
    top = Q_best[all]
    start = stream.record()
    top.calc_tensor()
    end = stream.record()
    end.synchronize()
    elapsed_time = cp.cuda.get_elapsed_time(start, end) / 1000
    print(f"elapsed time by optimal contraction: {elapsed_time}[s]")
    print(f"output by optimal contraction: {top.tensor}")
    print(f"contraction order: {top.calc_expression()}")
    cp_optimal_list.append(elapsed_time)

    del cp_tensor_list
    cp._default_memory_pool.free_all_blocks()

if naive == 1:
    cp_naive_list = cp.array(cp_naive_list)
    print(f"cupy naive: ave {cp.average(cp_naive_list)}, std {cp.std(cp_naive_list)}")

cp_optimal_list = cp.array(cp_optimal_list)
print(f"cupy optimal: ave {cp.average(cp_optimal_list)}, std {cp.std(cp_optimal_list)}")

total_elapsed_end = time.time()
print(f"total elapsed time: {total_elapsed_end - total_elapsed_start}")