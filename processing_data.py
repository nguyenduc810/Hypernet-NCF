import numpy as np
import pandas as pd

path = "/home/ubuntu/duc.nm195858/data/amazon_training.npy"
path_2 = "/home/ubuntu/duc.nm195858/data/products_data_amazon.npy"

data_train = np.load(path)
data_product = np.load(path_2)

nonzero_indices = np.nonzero(data_train)
user_indices = nonzero_indices[0]
item_indices = nonzero_indices[1]
ratings = data_train[nonzero_indices]
prices = data_product[item_indices]

# print(user_indices.shape)
# print(item_indices.shape)
# print(prices.shape)

user_item_rating_price = np.column_stack((user_indices.astype(int), item_indices.astype(int), ratings, prices))



# print(int(user_item_rating_price[1][0]))
# print(user_item_rating_price.shape)


# Xáo trộn các hàng của ma trận
np.random.shuffle(user_item_rating_price)

# Tính số lượng mẫu
num_samples = len(user_item_rating_price)

# Tính số lượng mẫu cho từng tập train, val, test
num_train = int(0.8 * num_samples)
num_val = int(0.1 * num_samples)
num_test = num_samples - num_train - num_val

# Chia ma trận thành các tập train, val, test
train_data = user_item_rating_price[:num_train]
val_data = user_item_rating_price[num_train:num_train+num_val]
test_data = user_item_rating_price[num_train+num_val:]
np.save('/home/ubuntu/duc.nm195858/Hypernet-NCF/data/training_dataset.npy', train_data)
np.save('/home/ubuntu/duc.nm195858/Hypernet-NCF/data/val_dataset.npy', val_data)
np.save('/home/ubuntu/duc.nm195858/Hypernet-NCF/data/test_dataset.npy', test_data)

