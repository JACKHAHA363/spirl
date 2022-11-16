import numpy as np
import itertools
import os
import pickle
import random

from spirl.components.data_loader import Dataset
from spirl.utils.general_utils import AttrDict

ML45PATH = '/home/yuchen/ML45-clean'

class D4RLSequenceSplitDataset(Dataset):
    SPLIT = AttrDict(train=0.99, val=0.01, test=0.0)

    def __init__(self, data_dir, data_conf, phase, resolution=None, shuffle=True, dataset_size=-1):
        self.phase = phase
        self.data_dir = data_dir
        self.spec = data_conf.dataset_spec
        self.subseq_len = self.spec.subseq_len
        self.remove_goal = self.spec.remove_goal if 'remove_goal' in self.spec else False
        self.device = data_conf.device
        self.n_worker = 4
        self.shuffle = shuffle
        self.dataset_size = dataset_size

        # Load metaworld dataset
        self.seqs = []
        self.total_size = 0
        for demo_file in os.listdir(ML45PATH):
            with open(os.path.join(ML45PATH, demo_file), 'rb') as f:
                print('### Load', demo_file)
                data = pickle.load(f)
            for traj in data:
                seq = AttrDict(
                    states=traj['observations'],
                    actions=traj['actions'],
                )
                self.seqs.append(seq)
                self.total_size += traj['observations'].shape[0]

        # 0-pad sequences for skill-conditioned training
        if 'pad_n_steps' in self.spec and self.spec.pad_n_steps > 0:
            for seq in self.seqs:
                seq.states = np.concatenate((np.zeros((self.spec.pad_n_steps, seq.states.shape[1]), dtype=seq.states.dtype), seq.states))
                seq.actions = np.concatenate((np.zeros((self.spec.pad_n_steps, seq.actions.shape[1]), dtype=seq.actions.dtype), seq.actions))

        # filter demonstration sequences
        if 'filter_indices' in self.spec:
            print("!!! Filtering kitchen demos in range {} !!!".format(self.spec.filter_indices))
            if not isinstance(self.spec.filter_indices[0], list):
                self.spec.filter_indices = [self.spec.filter_indices]
            self.seqs = list(itertools.chain.from_iterable([\
                list(itertools.chain.from_iterable(itertools.repeat(x, self.spec.demo_repeats)
                               for x in self.seqs[fi[0] : fi[1]+1])) for fi in self.spec.filter_indices]))
            random.shuffle(self.seqs)

        random.shuffle(self.seqs)
        self.n_seqs = len(self.seqs)

        if self.phase == "train":
            self.start = 0
            self.end = int(self.SPLIT.train * self.n_seqs)
        elif self.phase == "val":
            self.start = int(self.SPLIT.train * self.n_seqs)
            self.end = int((self.SPLIT.train + self.SPLIT.val) * self.n_seqs)
        else:
            self.start = int((self.SPLIT.train + self.SPLIT.val) * self.n_seqs)
            self.end = self.n_seqs
        self.end = min(self.n_seqs, self.end)
        print('#### Average seq length', np.mean([len(seq['states']) for seq in self.seqs]))
        
    def __getitem__(self, index):
        # sample start index in data range
        seq_id = self._sample_seq_id()
        seq = self.seqs[seq_id]
        start_idx = np.random.randint(0, seq.states.shape[0] - self.subseq_len - 1)
        output = AttrDict(
            states=seq.states[start_idx:start_idx+self.subseq_len],
            actions=seq.actions[start_idx:start_idx+self.subseq_len-1],
            pad_mask=np.ones((self.subseq_len,)),
        )
        if self.remove_goal:
            output.states = output.states[..., :int(output.states.shape[-1]/2)]
        return output
    
    def _sample_seq_id(self):
        return random.randint(self.start, self.end - 1)

    def __len__(self):
        if self.dataset_size != -1:
            return self.dataset_size
        return min(int(self.SPLIT[self.phase] * self.total_size / self.subseq_len), 25000)
