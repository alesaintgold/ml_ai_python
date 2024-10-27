import random
from random import shuffle

def split_array(array, percentage):
    split_index = int(len(array) * percentage)
    shuffle(array)
    train_set = array[:split_index]
    test_set = array[split_index:]
    return train_set, test_set