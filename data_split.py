import os
import random

if __name__=="__main__":

    data_list = list(range(1, 50001))
    random.shuffle(data_list)


    split_index = int(0.8 * len(data_list))


    train_data = data_list[:split_index]
    val_data = data_list[split_index:]

    train_split = os.path.join('./dataset',"train.txt")
    with open(train_split, "w") as file:
        for data in train_data:
            file.write(f"{data}\n")

    val_split = os.path.join('./dataset',"val.txt")
    with open(val_split, "w") as file:
        for data in val_data:
            file.write(f"{data}\n")  
