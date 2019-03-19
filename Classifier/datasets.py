# IRIS dataset
def read_dataset_IRIS(file_path='./datasets/iris.data'):
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