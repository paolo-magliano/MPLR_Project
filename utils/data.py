import numpy as np

def load_data(file_path):
    array = np.loadtxt(file_path, dtype=str, delimiter=",")
    data = array[:, :-1].T.astype(float)
    label_mapping = {label: i for i, label in enumerate(np.unique(array[:, -1]))}
    label = np.array([label_mapping[label] for label in array[:, -1]])

    return data, label

def shuffle_data(data, label, seed=0):
    np.random.seed(seed)
    idx = np.random.permutation(data.shape[1])

    return data[:, idx], label[idx]

def split_db_2to1(data, label, seed=0):
    n_samples_train = int(data.shape[1]*2.0/3.0)
    np.random.seed(seed)

    idx = np.random.permutation(data.shape[1])
    idx_train = idx[0:n_samples_train]
    idx_test = idx[n_samples_train:]

    data_train = data[:, idx_train]
    data_test = data[:, idx_test]
    label_train = label[idx_train]
    label_test = label[idx_test]

    return (data_train, label_train), (data_test, label_test)

def split_db(data, label, split_number, seed=0):
    np.random.seed(seed)
    split_sample = int(data.shape[1] / split_number)
    idex_permutation = np.random.permutation(data.shape[1])
    split_data = np.empty((0, data.shape[0], split_sample))
    split_label = np.empty((0, split_sample))

    for n in range(split_number):
        split_index = idex_permutation[n * split_sample : (n + 1) * split_sample]
        data_chunk = data[:, split_index].reshape((1, data.shape[0], split_sample))
        label_chunk = label[split_index].reshape((1, split_sample))

        split_data = np.append(split_data, data_chunk, axis=0)
        split_label = np.append(split_label, label_chunk, axis=0)

    return split_data, split_label

def k_fold_data(k_data, k_label, index):
    data_train, label_train = np.concatenate(np.delete(k_data, index, 0), axis=1), np.concatenate(np.delete(k_label, index, 0), axis=0)
    data_test, label_test = k_data[index], k_label[index]

    return (data_train, label_train), (data_test, label_test)

def k_data(data, label, index, k):
    test_indexes = np.arange(index, data.shape[1], k)
    train_indexes = np.delete(np.arange(data.shape[1]), test_indexes)

    return (data[:, train_indexes], label[train_indexes]), (data[:, test_indexes], label[test_indexes])

def k_data_calibration(data, label, index, k):
    cal_indexes = np.arange(index, data.shape[1], 2*k)
    test_indexes = np.arange(index + k, data.shape[1], 2*k)
    train_indexes = np.delete(np.arange(data.shape[1]), np.concatenate([cal_indexes, test_indexes]))

    return (data[:, train_indexes], label[train_indexes]), (data[:, test_indexes], label[test_indexes]), (data[:, cal_indexes], label[cal_indexes])