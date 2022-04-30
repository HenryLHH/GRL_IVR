import numpy as np
from dmbrl.data import Batch
from .utils import SegmentTree

class Storage:

    def __init__(self, size):
        self._maxsize = size
        self._size = 0
        self._index = 0
        self._keys = []
        self.reset()


    def __len__(self):
        return self._size


    def __del__(self):
        for k in self._keys:
            v = getattr(self, k)
            del v

    def __getitem__(self, index):
        batch_dict = dict(
            zip(self._keys, [getattr(self,k)[index] for k in self._keys]))
        return batch_dict


    def set_placeholder(self, key, value):
        if isinstance(value, np.ndarray):
            setattr(self, key, np.zeros((self._maxsize, *value.shape)))
        elif isinstance(value, dict):
            setattr(self, key, np.array([{} for _ in range(self._maxsize)]))
        elif np.isscalar(value):
            setattr(self, key, np.zeros((self._maxsize,)))

    
    def add(self, data):
        assert isinstance(data, dict)
        for k, v in data.items():
            if v is None:
                continue
            if k not in self._keys:
                self._keys.append(k)
                self.set_placeholder(k, v)
            getattr(self, k)[self._index] = v

        self._size = min(self._size + 1, self._maxsize)
        self._index = (self._index + 1) % self._maxsize


    def add_list(self, data, length):
        assert isinstance(data, dict)

        _tmp_idx = self._index + length

        for k, v in data.items():
            if v is None:
                continue
            if k not in self._keys:
                self._keys.append(k)
                self.set_placeholder(k, v[0])
            
            assert v.shape[0] == length
            
            if _tmp_idx < self._maxsize:
                getattr(self, k)[self._index:_tmp_idx] = v
            else:
                getattr(self, k)[self._index:] = v[:self._maxsize - self._index]
                getattr(self, k)[:_tmp_idx - self._maxsize] = v[self._maxsize - self._index:]
            

        self._size = min(self._size + length, self._maxsize)
        self._index = (self._index + length) % self._maxsize


    def update(self, buffer):
        i = begin = buffer._index % len(buffer)
        to_indices = []
        
        while True:
            to_indices.append(self._index)
            self.add(buffer[i])
            i = (i+1)% len(buffer)
            if i == begin:
                break

        return np.array(to_indices)

    def reset(self):
        self._index = self._size = 0


    def sample(self, batch_size):
        if batch_size > 0:
            indice = np.random.choice(self._size, batch_size)
        else: # sample all available data when batch_size=0
            indice = np.concatenate([
                np.arange(self._index, self._size),
                np.arange(0, self._index),
            ])
        return Batch(**self[indice]), indice


class CacheBuffer(Storage):

    def __init__(self):
        super().__init__(size=0)

    def add(self, data):
        assert isinstance(data, dict)
        for k, v in data.items():
            if v is None:
                continue
            if k not in self._keys:
                self._keys.append(k)
                setattr(self, k, [])
            getattr(self, k).append(v)

        self._index += 1
        self._size += 1

    def reset(self):
        self._index = self._size = 0
        for k in self._keys:
            setattr(self, k, [])


class PrioritizedStorage(Storage):

    def __init__(self, size, alpha, beta, weight_norm):

        super().__init__(size)

        assert alpha > 0.0 and beta >= 0.0
        self._alpha, self._beta = alpha, beta
        self._max_prio = self._min_prio = 1.0

        # save weight directly in this class instead of self._meta
        self.weight = SegmentTree(size)
        self.__eps = np.finfo(np.float32).eps.item()
        self._weight_norm = weight_norm

    def init_weight(self, index):
        self.weight[index] = self._max_prio**self._alpha

    def update(self, buffer):
        indices = super().update(buffer)
        self.init_weight(indices)
        return indices

    def add(self, data):
        self.init_weight(self._index)
        super().add(data)


    def get_weight(self, index):
        """Get the importance sampling weight.
        The "weight" in the returned Batch is the weight on loss function to debias
        the sampling process (some transition tuples are sampled more often so their
        losses are weighted less).
        """
        # important sampling weight calculation
        # original formula: ((p_j/p_sum*N)**(-beta))/((p_min/p_sum*N)**(-beta))
        # simplified formula: (p_j/p_min)**(-beta)
        return (self.weight[index] / self._min_prio)**(-self._beta)

    def update_weight(self, index, new_weight):
        """Update priority weight by index in this buffer.
        :param np.ndarray index: index you want to update weight.
        :param np.ndarray new_weight: new priority weight you want to update.
        """
        weight = np.abs(new_weight) + self.__eps
        self.weight[index] = weight**self._alpha
        self._max_prio = max(self._max_prio, weight.max())
        self._min_prio = min(self._min_prio, weight.min())

    def __getitem__(self, index):
        batch_dict = super().__getitem__(index)
        weight = self.get_weight(index)
        # ref: https://github.com/Kaixhin/Rainbow/blob/master/memory.py L154
        batch_dict['weight'] = weight / np.sum(weight) * len(index) if self._weight_norm else weight
        return batch_dict

    def set_beta(self, beta: float) -> None:
        self._beta = beta