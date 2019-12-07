import torch
import random
from copy import deepcopy
from typing import Tuple
from torch_geometric.data import Data
from paraller import ParallerParser


class RandomWalk(object):
    def __init__(self, data: Data, edge_weight: torch.FloatTensor = None,
                 is_sorted: bool = False, is_parallel: bool = True):
        self.is_unweighted = True
        self.is_parallel = is_parallel

        if not is_sorted:
            edge_list = data.edge_index.tolist()
            if not edge_weight is None:
                self.is_unweighted = False
                edge_list.append(edge_weight)
            sorted(edge_list, key=lambda x: x[0])
            data.edge_index = torch.LongTensor(edge_list[:2])
            if not edge_weight is None:
                edge_weight = torch.FloatTensor(edge_list[-1])

        self.data = data
        self.edge_weight = edge_weight
        self.deg = self._degree()
        self.cum_deg = torch.cat(
            (torch.zeros(1, dtype=torch.long), self.deg.cumsum(0)), 0)

    def _degree(self) -> torch.LongTensor:
        zero = torch.zeros(self.data.num_nodes, dtype=torch.long)
        one = torch.ones(self.data.num_edges, dtype=torch.long)
        return zero.scatter_add_(0, self.data.edge_index[0], one)

    def _get_neighbors(self, cur: int) -> Tuple[torch.LongTensor, torch.FloatTensor]:
        nl, nr = self.cum_deg[cur], self.deg[cur]
        cur_neighbor = self.data.edge_index[1, nl:nl+nr]
        cur_edge_weight = None
        if not self.is_unweighted:
            cur_edge_weight = deepcopy(self.edge_weight[nl:nl+nr])
        return cur_neighbor, cur_edge_weight

    def _sample(self, ll, rr, start: torch.LongTensor, walk_length: int = 20,
                p: float = 1.0, q: float = 1.0, out: torch.LongTensor = None):
        # full tensor with 0(_end flag)
        out = torch.full((rr-ll, walk_length+1), 0, dtype=torch.long)
        for n in range(0, rr-ll):
            cur = start[n]
            out[n, 0] = cur

            for l in range(1, walk_length+1):
                cur_neighbor, cur_edge_weight = self._get_neighbors(cur)
                if cur_neighbor.size(0) == 0:
                    break
                if self.is_unweighted:
                    cur = random.choice(cur_neighbor.tolist())
                else:
                    # update edge_weight following p/q
                    if l > 1:
                        prev_cur = out[n, l-2]
                        for idx, item in enumerate(cur_neighbor):
                            if item == prev_cur:
                                cur_edge_weight[idx] /= p
                            else:
                                prev_cur_neighbor, prev_cur_edge_weight = self._get_neighbors(
                                    prev_cur)
                                if item in prev_cur_neighbor:
                                    continue
                                else:
                                    cur_edge_weight[idx] /= q

                    cur = random.choices(
                        cur_neighbor.tolist(), weights=cur_edge_weight.tolist())[0]
                out[n, l] = cur
            if not self.is_unweighted:
                del cur_edge_weight
        return out

    def walk(self, start: torch.LongTensor, walk_length: int = 20,
             p: float = 1.0, q: float = 1.0) -> torch.LongTensor:
        out = None

        if self.is_parallel:
            paraller = ParallerParser(total=start.size(0),
                                      start=start, walk_length=walk_length, p=p, q=q)
            paraller.parser = self._sample
            for item in paraller.run():
                if out is None:
                    out = item.get()
                else:
                    out = torch.cat((out, item.get()), 0)
        else:
            out = self._sample(0, start.size(0), start=start,
                               walk_length=walk_length, p=p, q=q)
        return out


if __name__ == "__main__":
    edge_index = torch.tensor([[0, 1, 1, 2],
                               [1, 0, 2, 1]], dtype=torch.long)
    x = torch.tensor([[-1], [0], [1]], dtype=torch.float)
    edge_weight = torch.tensor([1, 0.2, 0.8, 1])
    start = torch.tensor([0, 1, 2])
    data = Data(x=x, edge_index=edge_index)
    rw = RandomWalk(data, edge_weight=edge_weight, is_parallel=True)
    print(rw.walk(start, walk_length=5, p=0.2, q=0.5))
