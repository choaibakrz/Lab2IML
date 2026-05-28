import argparse
import numpy as np

from src.methods.dummy_methods import DummyClassifier
from src.methods.mlp import MLP
from src.losses import MSE, CrossEntropy
from src.activations import Sigmoid, ReLU , Softmax
from src.methods.kmeans import KMeans
from src.utils import normalize_fn, append_bias_term, accuracy_fn, macrof1_fn, mse_fn, label_to_onehot
import os

np.random.seed(100)


def main(args):
    """
    The main function of the script.

    Arguments:
        args (Namespace): arguments that were parsed from the command line (see at the end
                          of this file). Their value can be accessed as "args.argument".
    """


    dataset_path = args.data_path
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    ## 1. We first load the data.

    feature_data = np.load(dataset_path, allow_pickle=True)
    train_features, test_features, train_labels_reg, test_labels_reg, train_labels_classif, test_labels_classif = (
        feature_data['xtrain'],feature_data['xtest'],feature_data['ytrainreg'],
        feature_data['ytestreg'],feature_data['ytrainclassif'],feature_data['ytestclassif']
    )

    ## 2. Then we must prepare it. This is where you can create a validation set,
    #  normalize, add bias, etc.

    # Make a validation set (it can overwrite xtest, ytest)
    if not args.test:
        ### WRITE YOUR CODE HERE
        N = train_features.shape[0]
        split = int(0.8 * N)
        perm = np.random.permutation(N)

        test_features        = train_features[perm[split:]]
        test_labels_reg      = train_labels_reg[perm[split:]]
        test_labels_classif  = train_labels_classif[perm[split:]]

        train_features       = train_features[perm[:split]]
        train_labels_reg     = train_labels_reg[perm[:split]]
        train_labels_classif = train_labels_classif[perm[:split]]

    ### WRITE YOUR CODE HERE to do any other data processing
    means = train_features.mean(axis=0, keepdims=True)
    stds  = train_features.std(axis=0, keepdims=True)
    stds[stds == 0] = 1  # avoid division by zero

    train_features = normalize_fn(train_features, means, stds)
    test_features  = normalize_fn(test_features,  means, stds)

    ## 3. Initialize the method you want to use.

    # Follow the "DummyClassifier" example for your methods
    if args.method == "dummy_classifier":
        method_obj = DummyClassifier(arg1=1, arg2=2)

    elif args.method == "kmeans":
        ### WRITE YOUR CODE HERE
        method_obj = KMeans(K=args.K, max_iters=args.max_iters)

    elif args.method == "mlp":
        ### WRITE YOUR CODE HERE
        n_features = train_features.shape[1]

        if args.task == "classification":
            n_classes  = len(np.unique(train_labels_classif))
            method_obj = MLP(
                dimensions=(n_features, 64, 32, n_classes),
                activations=(ReLU, ReLU, Sigmoid)
            )
        else:  # regression
            method_obj = MLP(
                dimensions=(n_features, 64, 32, 1),
                activations=(ReLU, ReLU, Sigmoid)
            )
    else:
        raise ValueError(f"Unknown method: {args.method}")

    ## 4. Train and evaluate the method

    if args.task == "classification":

        ### WRITE YOUR CODE HERE
        if args.method == "mlp":
            n_classes = len(np.unique(train_labels_classif))
            y_onehot  = label_to_onehot(train_labels_classif, C=n_classes)
            method_obj.fit(train_features, y_onehot,
                           loss=CrossEntropy, epochs=args.max_iters,
                           batch_size=32, learning_rate=args.lr)
            pred_train = np.argmax(method_obj.predict(train_features), axis=1)
            pred_test  = np.argmax(method_obj.predict(test_features),  axis=1)
        else:
            pred_train = method_obj.fit(train_features, train_labels_classif)
            pred_test  = method_obj.predict(test_features)

        acc_train = accuracy_fn(pred_train, train_labels_classif)
        f1_train  = macrof1_fn(pred_train,  train_labels_classif)
        acc_test  = accuracy_fn(pred_test,  test_labels_classif)
        f1_test   = macrof1_fn(pred_test,   test_labels_classif)

        print(f"\n[{args.method} - classification]")
        print(f"  Train  accuracy={acc_train:.2f}%  macro-F1={f1_train:.4f}")
        print(f"  Test   accuracy={acc_test:.2f}%  macro-F1={f1_test:.4f}")

    elif args.task == "regression":
        assert args.method != "kmeans", f"You should use kmeans as a classification method"

        ### WRITE YOUR CODE HERE
        if args.method == "mlp":
            y_col = train_labels_reg.reshape(-1, 1).astype(float)
            y_min, y_max = y_col.min(), y_col.max()
            y_norm = (y_col - y_min) / (y_max - y_min + 1e-8)

            method_obj.fit(train_features, y_norm,
                           loss=MSE, epochs=args.max_iters,
                           batch_size=32, learning_rate=args.lr)

            pred_train = method_obj.predict(train_features)[:, 0] * (y_max - y_min) + y_min
            pred_test  = method_obj.predict(test_features)[:,  0] * (y_max - y_min) + y_min
        else:
            pred_train = method_obj.fit(train_features, train_labels_reg).astype(float)
            pred_test  = method_obj.predict(test_features).astype(float)

        mse_train = mse_fn(pred_train, train_labels_reg)
        mse_test  = mse_fn(pred_test,  test_labels_reg)

        print(f"\n[{args.method} - regression]")
        print(f"  Train  MSE={mse_train:.4f}")
        print(f"  Test   MSE={mse_test:.4f}")

    ### WRITE YOUR CODE HERE if you want to add other outputs, visualization, etc.


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        default="classification",
        type=str,
        help="classification / regression / clustering",
    )
    parser.add_argument(
        "--method",
        default="dummy_classifier",
        type=str,
        help="dummy_classifier / kmeans / mlp",
    )
    parser.add_argument(
        "--data_path",
        default="data/features.npz",
        type=str,
        help="path to your dataset CSV file",
    )
    parser.add_argument(
        "--K",
        type=int,
        default=1,
        help="number of clusters datapoints used for kmeans",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="learning rate for methods with learning rate",
    )
    parser.add_argument(
        "--max_iters",
        type=int,
        default=100,
        help="max iters for methods which are iterative",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="train on whole training data and evaluate on the test data, "
             "otherwise use a validation set",
    )
    # Feel free to add more arguments here if you need!

    args = parser.parse_args()
    main(args)