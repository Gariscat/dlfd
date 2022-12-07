from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


Delta_i = 100 * 1000  # 100ms


def parse_trace(
    trace_path,
    source_id,
    obs_ord=2,
    scale=1e8,
    reduced=True,
    lim=100000,
) -> np.ndarray:
    rece_times = []
    emit_times = []
    with open(trace_path, 'r') as f:
        log_items = f.readlines()
    for i, line in enumerate(tqdm(log_items, desc='initialize env - collecting log items')):
        # if i > 10000: break
        log_item = [int(x) for x in line.strip().split()]
        try:
            sid, _, emit_time, rece_time, _ = log_item
            if sid != source_id:
                continue
        except ValueError:
            ### erroneous log item (e.g. not enough items to unpack)
            continue
        rece_times.append(rece_time)
        # emit_times.append(emit_time)

    """plt.plot(emit_times[:32])
    plt.savefig("emissions.jpg")
    plt.close()"""

    del log_items
        
    if len(rece_times) == 0:
        raise KeyError(f'Current log file does not include pulses sent from node {source_id}')
        
    rece_times = np.array(rece_times)
    gaps = rece_times[1:] - rece_times[:-1]
    del rece_times

    ret = np.zeros((gaps.shape[0], obs_ord))
    ret[:, 0] = gaps
    for i in range(obs_ord-1):
        ret[i+1:, i+1] = ret[i+1:, i] - ret[:-i-1, i]

    if reduced:
        ret = ret[:lim, :]
        print(ret[:5, 0], '......')
        plt.plot(ret[:32, 0]-Delta_i)
        plt.savefig(f'pulses (unnormalized).jpg')
        plt.close()

    ret /= scale
    return ret
    """
    def __getitem__(self, idx):
        # from [idx-seqlen, idx-1] predict idx 
        target = torch.tensor(data[idx][0])
        
        l, r = max(0, idx-seq_len), idx
        source = torch.from_numpy(data[l:r])
        left_zeros = torch.zeros(seq_len-source.shape[0], obs_ord)
        source = torch.cat((left_zeros, source), dim=0)
        
        return source, target
    """
    
def get_data(trace_path, source_id, obs_ord, scale, test_size=0.1):
    orig_set = parse_trace(
        trace_path=trace_path,
        source_id=source_id,
        obs_ord=obs_ord,
        scale=scale
    )
    delim = int(orig_set.shape[0] * (1 - test_size))
    train_set, eval_set = orig_set[:delim], orig_set[delim:]
    
    return train_set, eval_set
    
    
if __name__ == '__main__':
    pass