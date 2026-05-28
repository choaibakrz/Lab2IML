import argparse
import numpy as np

from src.methods.dummy_methods import DummyClassifier
from src.methods.mlp import MLP
from src.losses import MSE, CrossEntropy
from src.activations import Sigmoid, ReLU, Softmax
from src.methods.kmeans import KMeans
from src.utils import normalize_fn, append_bias_term, accuracy_fn, macrof1_fn, mse_fn, label_to_onehot
import os

np.random.seed(100)


def main(args):
    dataset_path = args.data_path
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    ## 1. Load data
    feature_data = np.load(dataset_path, allow_pickle=True)
    train_features, test_features, train_labels_reg, test_labels_reg, train_labels_classif, test_labels_classif = (
        feature_data['xtrain'], feature_data['xtest'], feature_data['ytrainreg'],
        feature_data['ytestreg'], feature_data['ytrainclassif'], feature_data['ytestclassif']
    )

    ## 2. Prepare data

    # Validation split (80/20)
    if not args.test:
        N = train_features.shape[0]
        split = int(0.8 * N)
        perm = np.random.permutation(N)

        test_features        = train_features[perm[split:]]
        test_labels_reg      = train_labels_reg[perm[split:]]
        test_labels_classif  = train_labels_classif[perm[split:]]

        train_features       = train_features[perm[:split]]
        train_labels_reg     = train_labels_reg[perm[:split]]
        train_labels_classif = train_labels_classif[perm[:split]]

    # Normalize (fit on train only)
    means = train_features.mean(axis=0, keepdims=True)
    stds  = train_features.std(axis=0, keepdims=True)
    stds[stds == 0] = 1

    train_features = normalize_fn(train_features, means, stds)
    test_features  = normalize_fn(test_features,  means, stds)

    ## 3. Initialize method
    if args.method == "dummy_classifier":
        method_obj = DummyClassifier(arg1=1, arg2=2)

    elif args.method == "kmeans":
        method_obj = KMeans(K=args.K, max_iters=args.max_iters)

    elif args.method == "mlp":
        n_features = train_features.shape[1]

        if args.task == "classification":
            n_classes  = len(np.unique(train_labels_classif))
            method_obj = MLP(
                dimensions=(n_features, 64, 32, n_classes),
                activations=(ReLU, ReLU, Softmax)  # Softmax for multi-class output
            )
        else:
            method_obj = MLP(
                dimensions=(n_features, 64, 32, 1),
                activations=(ReLU, ReLU, Sigmoid)
            )
    else:
        raise ValueError(f"Unknown method: {args.method}")

    ## 4. Train and evaluate

    if args.task == "classification":
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
        assert args.method != "kmeans", "K-Means is for classification only."

        if args.method == "mlp":
            y_col  = train_labels_reg.reshape(-1, 1).astype(float)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",      default="classification", type=str)
    parser.add_argument("--method",    default="dummy_classifier", type=str)
    parser.add_argument("--data_path", default="data/features.npz", type=str)
    parser.add_argument("--K",         type=int,   default=1)
    parser.add_argument("--lr",        type=float, default=1e-5)
    parser.add_argument("--max_iters", type=int,   default=100)
    parser.add_argument("--test",      action="store_true")

    args = parser.parse_args()
    main(args)