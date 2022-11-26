from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
from sklearn.model_selection import train_test_split


class PulseDataset(Dataset):
    def __init__(self,
        trace_path,
        source_id,
        obs_ord=2,
        scale=1e8,
        seq_len=1000,
    ) -> None:
        super().__init__()
        self._trace_path = trace_path
        self._source_id = source_id
        self._obs_ord = obs_ord
        self._scale = scale
        self._seq_len = seq_len


        rece_times = []
        with open(self._trace_path, 'r') as f:
            log_items = f.readlines()
        for i, line in enumerate(tqdm(log_items, desc='initialize env - collecting log items')):
            # if i > 10000: break
            log_item = [int(x) for x in line.strip().split()]
            try:
                source_id, _, _, rece_time, _ = log_item
                if source_id != self._source_id:
                    continue
            except ValueError:
                ### erroneous log item (e.g. not enough items to unpack)
                continue
            rece_times.append(rece_time)

        del log_items
        
        if len(rece_times) == 0:
            raise KeyError(f'Current log file does not include pulses sent from node {self._source_id}')
        
        rece_times = np.array(rece_times) / self._scale
        gaps = rece_times[1:] - rece_times[:-1]
        del rece_times

        self._data = np.zeros((gaps.shape[0], self._obs_ord))
        self._data[:, 0] = gaps
        for i in range(self._obs_ord-1):
            self._data[i+1:, i+1] = self._data[i+1:, i] - self._data[:-i-1, i]
    
    def __len__(self):
        # for a time series of length n, we have n input-output pairs with padding.
        return self._data.shape[0]

    def __getitem__(self, idx):
        # from [idx-seqlen, idx-1] predict idx 
        target = torch.tensor(self._data[idx][0])
        
        l, r = max(0, idx-self._seq_len), idx
        source = torch.from_numpy(self._data[l:r])
        left_zeros = torch.zeros(self._seq_len-source.shape[0], self._obs_ord)
        source = torch.cat((left_zeros, source), dim=0)
        
        return source, target
    
    
def get_data(trace_path, source_id, obs_ord, scale, seq_len, orig_shuffle=False):
    orig_set = PulseDataset(
        trace_path=trace_path,
        source_id=source_id,
        obs_ord=obs_ord,
        scale=scale,
        seq_len=seq_len
    )
    train_set, eval_set = train_test_split(orig_set, test_size=0.1, shuffle=orig_shuffle)
    
    return train_set, eval_set
    
    
if __name__ == '__main__':
    trace_path = './traces/trace.log'
    node_id = 5
    dataset = PulseDataset(trace_path, node_id, obs_ord=3, seq_len=1000)
    print(dataset[3])