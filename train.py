# IBM Research Singapore, 2022

# main train script

import argparse
import os
import sys
from typing import List, Optional

import numpy as np
import torch
from tqdm import tqdm

# ----------------------------------------------------------
# ARGS
# ----------------------------------------------------------


def common_args():
    import argparse

    parser = argparse.ArgumentParser("Train graphical models for bitcoin data")

    parser.add_argument("model")  # first positional argument
    parser.add_argument("-model_path", default="data/models")
    parser.add_argument("-model_accum_grads", type=int, default=1)
    parser.add_argument("-data_path", default="data/dataset")
    parser.add_argument(
        "-features_dir", default="data/tables", help="""dir where features are from"""
    )
    parser.add_argument("-data_disable_semi_sup", action="store_true", default=True)
    parser.add_argument("-resample_semi_sup", default="random")
    parser.add_argument("-resample_factor_semi_sup", type=int, default=1)
    parser.add_argument("-additional_features", nargs="*")
    parser.add_argument("-train", action="store_false", default=True)
    parser.add_argument("-train_best_metric", default="bacc")
    parser.add_argument("-epochs", type=int, default=10)
    # parser.add_argument('-seed', type=int, default=None) # first positional argument
    parser.add_argument("-tensorboard", action="store_true", default=False)
    parser.add_argument(
        "-debug", action="store_true", default=False
    )  # activate some debug functions
    parser.add_argument(
        "-refresh_cache", action="store_true", default=False
    )  # Refreshes loader's cache
    return parser


def parse_args(
    parser: argparse.ArgumentParser,
    args: Optional[List[str]] = None,
):

    opt = parser.parse_args(args)

    if torch.cuda.is_available():
        opt.device = torch.device("cuda")
    else:
        opt.device = torch.device("cpu")

    return opt


import warnings
from typing import Dict

import sklearn.metrics


# to build the metrics
def metrics(name: str, **kwargs):
    import torchmetrics

    if name == "auroc":
        with warnings.catch_warnings(record=True) as w:
            # this one has severe memory leak problems.
            # do not use
            auroc = torchmetrics.AUROC(pos_label=1, **kwargs)
            return lambda p, t: {"auroc": auroc(p, t).item()}
    elif name == "accuracies":
        conf = torchmetrics.ConfusionMatrix(
            num_classes=2,
            normalize="true",
            threshold=0.5,
        ).to(kwargs.get("device", torch.device("cpu")))

        def _acc(p, t):
            M = conf(p, t)
            tpr = (M[1, 1] / M[1].sum()).item()
            tnr = (M[0, 0] / M[0].sum()).item()
            return {
                "bacc": (tpr + tnr) / 2.0,
                "tpr": tpr,
                "tnr": tnr,
            }

        return _acc
    else:
        raise NotImplementedError


from collections import defaultdict


def train(
    model,
    data: Dict,
    labelled,
    optimizer,
    loss_fn,
    thresh=0.5,
    accum_gradients=1,
    compute_auroc=False,
    device=torch.device("cpu"),
    tqdm=None,
):
    model.train()

    _loss = 0.0
    _metrics = {
        # "auroc": metrics("auroc") # do not use
        "accuracies": metrics("accuracies", device=device),
    }
    _mvals = defaultdict(float)

    # for gradient
    optimizer.zero_grad()
    _n = len(data)

    for cnt, (k, v) in enumerate(data.items()):
        out = model(v.x, v.edge_index)
        _lab = labelled[k]
        loss = loss_fn(out[_lab], v.y[_lab].unsqueeze(dim=-1))

        # y_pred = (torch.sigmoid(out[_lab]) > thresh).cpu()

        loss.backward()

        _cnt = cnt + 1
        if ((_cnt % accum_gradients) == 0) or (_cnt == _n):
            optimizer.step()
            optimizer.zero_grad()

        _loss += loss.item()

        with warnings.catch_warnings(record=True) as w:
            # auroc (use the sklearn version)
            if compute_auroc:
                _mvals["auroc"] += sklearn.metrics.roc_auc_score(
                    v.y[_lab].cpu(),
                    torch.sigmoid(out[_lab]).detach().cpu(),
                )

            for k, m in _metrics.items():
                # each metric when executed will be a dict
                # - (key, val) : metric name - value
                for k2, v in m(
                    torch.sigmoid(out[_lab]).squeeze(),
                    v.y_i[_lab].squeeze(),
                ).items():
                    _mvals[k2] += v

    return {
        "loss": _loss / len(data),
        **{k: v / len(data) for k, v in _mvals.items()},
    }


@torch.no_grad()
def evaluate(
    model,
    data,
    labelled,
    device=torch.device("cpu"),
):

    model.eval()

    _metrics = {
        # "auroc": metrics("auroc") # do not use
        "accuracies": metrics("accuracies", device=device),
    }
    _mvals = defaultdict(float)

    for k, v in data.items():
        out = model(v.x, v.edge_index)
        _lab = labelled[k]

        with warnings.catch_warnings(record=True) as w:

            # auroc (use the sklearn version)
            _mvals["auroc"] += sklearn.metrics.roc_auc_score(
                v.y[_lab].cpu(),
                torch.sigmoid(out[_lab]).detach().cpu(),
            )

            # see comments in train
            for k, m in _metrics.items():
                for k2, v in m(
                    torch.sigmoid(out[_lab]).squeeze(),
                    v.y_i[_lab].squeeze(),
                ).items():
                    _mvals[k2] += v

    return {
        **{k: v / len(data) for k, v in _mvals.items()},
    }, None


# this is for the 3 class
# - 0 vs 1
# - 2 vs 1
def compute_roc_curve(
    out: List,
    y: List,  # this has 3 classes
    # thresholds: List[float],
    quantiles: List[float],
    num_classes=3,
):
    def _find(c):
        (idx,) = np.where(y == c)
        return np.sort(out[idx])

    _out = {c: _find(c) for c in range(num_classes)}
    _n = {c: len(_out[c]) for c in _out}

    from bisect import bisect

    thresholds = [_out[1][int(np.floor((1.0 - q) * _n[1]))] for q in quantiles]

    results = []
    for t in thresholds:
        _ind = {c: bisect(_out[c], t) for c in _out}

        if _ind[1] < _n[1]:
            # if not the trival case
            results.append(
                {
                    "threshold": t,
                    "tpr": 1.0 - _ind[1] / _n[1],
                    **{
                        f"fpr{i}": 1 - _ind[i] / _n[i]
                        for i in range(num_classes)
                        if i != 1
                    },
                }
            )
    return results


from datasets.types import DATA_LABEL_TEST, DATA_LABEL_TRAIN, DATA_LABEL_VAL


# load data into tensors
def load_data(opt, **kwargs):
    from datasets.loader import FEATURE_COLUMNS, load_data

    _add_f = opt.additional_features
    if _add_f is None:
        _add_f = []
    data, labelled, scaler, feature_names = load_data(
        opt.data_path,
        opt.device,
        debug=opt.debug,
        semi_supervised=opt.data_disable_semi_sup,
        semi_supervised_resample_negs=opt.resample_semi_sup,
        semi_supervised_resample_factor=opt.resample_factor_semi_sup,
        feature_column_names=FEATURE_COLUMNS + _add_f,
        features_dir=opt.features_dir,  # used for duckdb queries
        refresh_cache=opt.refresh_cache,
        **kwargs,
    )
    n_features = len(feature_names)
    print(feature_names)
    class_weights = opt.class_weight

    return (data, labelled, scaler, feature_names, n_features, class_weights)


# build the model, optimizer, loss
def build_model_opt_loss(opt, class_weights):
    # get the class weights
    if opt.model == "gcn":
        import models.gcn

        model, optimizer, loss = models.gcn.build_model(
            opt,
            class_weights=class_weights,
        )
    elif opt.model == "gat":
        import models.gat

        model, optimizer, loss = models.gat.build_model(
            opt,
            class_weights=class_weights,
        )
    else:
        raise NotImplementedError

    return model, optimizer, loss


# load model
def load_model(opt, others: List[str] = []):
    if opt.model == "gat":
        import models.gat

        return models.gat.load_model(
            opt,
            model_path=os.path.join(opt.model_path, f"{opt.model}.pt"),
            others=others,
        )
    else:
        raise NotImplementedError


# build the train/eval functions
def build_functions(opt):
    train_args = {
        "accum_gradients": opt.model_accum_grads,
    }
    return (train, evaluate, train_args)


# ----------------------------------------------------------
# MAIN
# ----------------------------------------------------------

from utils import export_args


def main(opt):

    (data, labelled, scaler, feature_names, n_features, class_weights) = load_data(opt)

    opt.input_dim = n_features
    print("n_features", opt.input_dim)

    # build the model, optimizer, loss
    model, optimizer, loss = build_model_opt_loss(opt, class_weights)

    # some printing
    print(model)

    # number of trainable parameters
    _mp = filter(lambda p: p.requires_grad, model.parameters())
    print(f"number of trainable parameters: {sum([np.prod(p.size()) for p in _mp])}")

    # build the functions
    (train, evaluate, train_args) = build_functions(opt)

    if opt.tensorboard:
        print("activate tensorboard")
        from utils import TensorboardWriter

        _external_writer = TensorboardWriter(
            os.path.join(opt.model_path, f"{opt.model}_logs")
        )
    else:
        _external_writer = None

    if opt.train:
        print("training")
        os.makedirs(opt.model_path, exist_ok=True)
        model_save_path = os.path.join(opt.model_path, f"{opt.model}.pt")
        print(f"saving model in '{model_save_path}'")

        best = 0
        best_test = None
        best_epoch = None
        best_metric = opt.train_best_metric
        with tqdm(total=opt.epochs) as pbar:
            for epoch in range(1, 1 + opt.epochs):
                meta = train(
                    model,
                    data[DATA_LABEL_TRAIN],
                    labelled[DATA_LABEL_TRAIN],
                    optimizer,
                    loss,
                    **train_args,
                    device=opt.device,
                )

                held_out_results = {}
                for k in [DATA_LABEL_VAL, DATA_LABEL_TEST]:
                    held_out_results[k], _ = evaluate(
                        model,
                        data[k],
                        labelled[k],
                        device=opt.device,
                    )

                _metric = held_out_results[DATA_LABEL_VAL][best_metric]
                if _metric > best:
                    best = _metric
                    best_test = held_out_results[DATA_LABEL_TEST][best_metric]
                    best_epoch = epoch

                    # print ("saving model", best, _metric)
                    torch.save(
                        {
                            "state_dict": model.state_dict(),
                            "metrics": {
                                "epoch": epoch,
                                "best_metric": best_metric,
                                **held_out_results,
                            },
                            "opt": export_args(opt),
                            "scaler": scaler,  # pickle it
                            "feature_names": feature_names,
                        },
                        model_save_path,
                    )

                pbar.update(1)
                pbar.set_postfix(
                    {
                        "loss": meta["loss"],
                        "best_val": best,
                        "best_test": best_test,
                        "best_epoch": best_epoch,
                    }
                )

                if _external_writer:
                    _external_writer.write("loss/train", meta["loss"], epoch)

                    for k, v in held_out_results.items():
                        for k2 in v:
                            _external_writer.write(f"{k2}/{k}", v[k2], epoch)


# build the parser depending on the model
def embelish_model_args(m: str, parser):
    if m == "gcn":
        import models.gcn

        return models.gcn.args(parser)
    elif m == "gat":
        import models.gat

        return models.gat.args(parser)
    else:
        raise NotImplementedError


if __name__ == "__main__":

    parser = common_args()
    print(parser)
    parser = embelish_model_args(sys.argv[1], parser)

    opt = parse_args(parser)

    print("arguments:")
    print(opt)

    # maybe later add some redundancy here
    main(opt)
