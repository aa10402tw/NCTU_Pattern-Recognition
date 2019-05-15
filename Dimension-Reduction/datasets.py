from utils import *


def lower(s):
    return s.lower()

def read_dataset(dataset='None'):
    if lower(dataset) == lower('Wine'):
        return read_dataset_Wine()
    if lower(dataset) == lower('ionosphere'):
        return read_dataset_ionosphere()
    if lower(dataset) == lower('Iris'):
        return  read_dataset_Iris()
    elif lower(dataset) == lower('Glass'):
        return  read_dataset_Glass()
    elif lower(dataset) == lower('BreastCancer'):
        return  read_dataset_BreastCancer()
    elif lower(dataset) == lower('Banknote'):
        return  read_dataset_Banknote()
    else:
        raise Excption('No such dataset to load %s'%dataset)

def read_dataset_Wine(file_path='./datasets/wine.data'):
    xs = []
    labels = []
    with open(file_path) as f:
        lines = f.readlines()
        for line in lines:
            datas = line.split('\n')[0].split(',')
            if len(datas) < 3:
                break
            x = [float(feature) for feature in datas[1:]]
            label = datas[0]
            xs.append(x)
            labels.append(label)
    X = np.array(xs)
    y = to_numerical(labels)
    return X, y



def read_dataset_ionosphere(file_path='./datasets/ionosphere.data'):
    xs = []
    labels = []
    with open(file_path) as f:
        lines = f.readlines()
        for line in lines:
            datas = line.split('\n')[0].split(',')
            if len(datas) < 3:
                break
            x = [float(feature) for feature in datas[:-1]]
            label = datas[-1]
            xs.append(x)
            labels.append(label)
    X = np.array(xs)
    y = to_numerical(labels)
    return X, y

# IRIS dataset
def read_dataset_Iris(file_path='./datasets/iris.data'):
    xs = []
    labels = []
    with open(file_path) as f:
        lines = f.readlines()
        for line in lines:
            datas = line.split('\n')[0].split(',')
            if len(datas) < 3:
                break
            x = [float(feature) for feature in datas[:4]]
            label = datas[-1]
            xs.append(x)
            labels.append(label)
    X = np.array(xs)
    y = to_numerical(labels)
    return X, y

# BreastCancer dataset
def read_dataset_BreastCancer(csv_path='./datasets/BreastCancer.csv'):
    import pandas as pd
    df = pd.read_csv(csv_path)
    train_columns = df.columns.values[:-1]
    X = df[train_columns].values
    y = df['Classification'].values -1 
    return X, y

# Glass dataset
def read_dataset_Glass(file_path='datasets/glass.data'):
    xs = []
    labels = []
    with open(file_path) as f:
        lines = f.readlines()
        for line in lines:
            datas = line.split('\n')[0].split(',')
            if len(datas) < 3:
                break
            x = [float(feature) for feature in datas[1:-1]]
            label = datas[-1]
            xs.append(x)
            labels.append(label)
    X = np.array(xs)
    y = to_numerical(labels)
    return X, y

# Bank Note Dataset
def read_dataset_Banknote(file_path='datasets/data_banknote_authentication.txt'):
    xs = []
    labels = []
    with open(file_path) as f:
        lines = f.readlines()
        for line in lines:
            datas = line.split('\n')[0].split(',')
            if len(datas) < 3:
                break
            x = [float(feature) for feature in datas[:-1]]
            label = datas[-1]
            xs.append(x)
            labels.append(label)
    X = np.array(xs)
    y = to_numerical(labels)
    return X, y