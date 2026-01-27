from torch.utils.data import Sampler
import random

class TrainSampler(Sampler):
    def __init__(
        self,
        dataset,
        batch_size: int,
        sample_mode: str,
        max_squared_res: int,
    ):
        self._data = dataset.data
        self._dataset_indices = list(range(len(self._data)))
        self._data["index"] = self._dataset_indices
        self._batch_size = batch_size
        self._sample_mode = sample_mode
        self.sampler_len = len(self._dataset_indices) * self._batch_size
        self.epoch = 0

        self._data["seq_len"] = self._data["seq"].apply(lambda x: len(x))
        self._data["max_batch_examples"] = self._data["seq_len"].apply(
            lambda x: max(int(max_squared_res // x**2), 1)
        )

    def __iter__(self):
        print("=" * 50, f"sampler seed: {self.epoch}", "=" * 50)
        # Each batch contains multiple time steps of the same protein.
        random.seed(self.epoch)
        random.shuffle(self._dataset_indices)
        self.epoch += 1
        if self._sample_mode == "time_batch":
            max_per_batch = self._data.iloc[self._dataset_indices]["max_batch_examples"].tolist()

            # Repeat each index to max batch size and pad until self._batch_size with None as indexes
            repeated_indices = []
            for idx, count in zip(self._dataset_indices, max_per_batch):
                # Repeat the index based on its count
                repeated_indices += [idx] * min(count, self._batch_size)
                repeated_indices += [None] * max(0, self._batch_size - count)

            return iter(repeated_indices)
        else:
            return iter(self._dataset_indices)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.sampler_len