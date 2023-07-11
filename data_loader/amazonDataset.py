import torch
from torch.utils.data import Dataset
import numpy as np

class AmazonDataset(Dataset):
    def __init__(self, file_data, num_negatives = 3):
        self.file_data = file_data
        self.data = np.load(file_data)
        self.num_negatives = num_negatives
        self.user_input, self.item_input, self.labels = self.get_train_instances(self.data)

    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.user_input[index], self.item_input[index], self.labels[index] 
    
    def get_train_instances(self, train):
        #user_input, item_input, labels = [],[],[]
        len_ = train.shape[0]

        user_input_ = train[:,0].astype(int)
        item_input_ = train[:,1].astype(int)
        price = train[:, 3].tolist()

        user_input = user_input_.tolist()
        item_input = item_input_.tolist()
        labels = [1]* len_

        # user_input.append(int(train[i][0]))
        # item_input.append(int(train[i][1]))
        for i in range(len_):
            # positive instance
            label_i = []
            for k in range(len_):
                if train[i][0] == train[k][0]:
                    label_i.append(train[k][1])

            # negative instances
            for t in range(self.num_negatives):
                # j = np.random.randint(num_items)

                # lấy ngẫu nhiên 1 item mà user chưa rating
                j = np.random.choice([x for x in train[:, 1] if x not in label_i])
                user_input.append(int(train[i][0]))
                item_input.append(int(j))
                labels.append(0)
                price.append(0)
        return user_input, item_input, labels, price