def split_data(X, y, split_ratio=0.8):
    N = X.shape[0]
    num_train = int(N * split_ratio)
    indices = [i for i in range(N)]
    np.random.shuffle(indices)
    X_train, y_train = X[indices[:num_train]], y[indices[:num_train]]
    X_test, y_test = X[indices[num_train:]], y[indices[num_train:]]
    return (X_train, y_train), (X_test, y_test)

def accuracy(y, y_pred):
    diff = y-y_pred
    num_error = np.count_nonzero(diff)
    return 1-(num_error / len(y))

def to_numerical(labels):
    labels_occured = set(sorted(labels))
    numericals = [i for i in range(len(labels_occured))]
    lab2num = dict(zip(labels_occured, numericals))
    nums = [lab2num[lab] for lab in labels]
    return np.array(nums)

def to_label(labels, numericals):
    labels = list(set(sorted(labels)))
    nums = [i for i in range(len(labels))]
    num2lab = dict(zip(nums, labels))
    labels = [num2lab[num] for num in numericals]
    return labels

## probability density function for Gaussian
def gaussian_pdf(x, mean, std):
    from math import exp, sqrt, log, pi
    variance = std**2
    avg = mean
    exponent = exp(-(pow(x - avg, 2) / (2 * variance)))
    return (1 / (sqrt(2 * pi) * sqrt(variance))) * exponent

def log_gaussian_pdf(x, mean, std):
    from math import exp, sqrt, log, pi
    variance = std**2
    exp_term = -(pow(x - mean, 2) / (2 * variance))
    return log(1 / (sqrt(2 * pi) * std)) + exp_term