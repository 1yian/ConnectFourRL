from typing import List, Dict

import torch


class ConnectFourDataset(torch.utils.data.dataset.Dataset):
    def __init__(self):
        super().__init__()
        self.examples = list()

    def __getitem__(self, idx):
        return self.examples[idx]

    def __len__(self):
        return len(self.examples)

    def add_record(self, record: List[Dict], discount_factor=1):
        final_score = record[0]['state'].final_score

        # Bellman discount math
        running_return = final_score
        for timestep in record[::-2]:
            timestep['discounted_score'] = running_return
            running_return *= discount_factor

        running_return = final_score
        for timestep in record[-2::-2]:
            timestep['discounted_score'] = running_return
            running_return *= discount_factor

        self.examples += record
