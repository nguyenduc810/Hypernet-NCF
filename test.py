from data_loader.amazonDataset import AmazonDataset
from torch.utils.data import DataLoader

train_path = "/home/ubuntu/duc.nm195858/Hypernet-NCF/data/training_dataset.npy"
val_path  = "/home/ubuntu/duc.nm195858/Hypernet-NCF/data/val_dataset.npy"
test_path = "/home/ubuntu/duc.nm195858/Hypernet-NCF/data/test_dataset.npy"

train_dataset = AmazonDataset(train_path)
val_dataset = AmazonDataset(val_path)
test_dataset = AmazonDataset(test_path) 

train_loader = DataLoader(train_dataset, batch_size = bs, shuffle = True)
val_loader = DataLoader(val_dataset, batch_size = bs, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size=bs, shuffle= True)

for batch in train_loader:
    # Trích xuất dữ liệu từ batch
    user_input, item_input, labels = batch

    # In dữ liệu
    print("user data:")
    print(user_input)
    print("item data:")
    print(item_input)
    print("label:")
    print(labels)

