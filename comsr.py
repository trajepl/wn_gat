import torch
from typing import Dict

## graphsage
node2vec_save_prefix = './model_save/WNNode2vec_1582008016.856677'
graphsage_save_prefix = './model_save/WNGraphSage_1582121713.6465182'
gat_save_prefix = './model_save/WNGat_1582122126.2792394'

def load_params(save_prefix: str) -> Dict:
    params = torch.load(f'{save_prefix}/max.m5')
    return params

if __name__ == "__main__":
    print(load_params(node2vec_save_prefix)['sr'])
    print(load_params(graphsage_save_prefix)['sr'])
    print(load_params(gat_save_prefix)['sr'])


