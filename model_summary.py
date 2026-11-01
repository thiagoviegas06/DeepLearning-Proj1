"""Model card for a ResNet config or a finished run: layer table, trainable-parameter count checked
against the competition's 5M limit, and (for a run) the final local test accuracy.

The competition rules ask for clear print statements of the parameter count and the final test
accuracy; this module is the one place that produces them, for the command line and the notebook.

  python model_summary.py                                        # train.py's default config
  python model_summary.py --blocks 4 4 3 --width 64 --kernel 3   # any config, before training it
  python model_summary.py --run runs/baseline_21368912           # config + test accuracy of a run

From a notebook:

  import model_summary
  model_summary.report(run="runs/baseline_21368912")

Only train.py and PyTorch are needed. If `torchinfo` is installed its summary is printed as an
independent cross-check of the parameter count; it is optional.
"""
import argparse
import json
from pathlib import Path

import torch

import train

INPUT_SIZE = (1, 3, 32, 32)


def count_params(model):
    """(trainable, total) parameter counts. The two are equal unless something is frozen."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def layer_table(model, input_size=INPUT_SIZE):
    """One row per top-level piece (stem, each residual block, classifier): name, description,
    output shape and parameter count, captured with a dry forward pass on the CPU."""
    rows, hooks = [], []

    def describe(mod):
        if isinstance(mod, train.BasicBlock):
            k = mod.conv1.kernel_size[0]
            proj = "1x1 projection" if not isinstance(mod.shortcut, torch.nn.Identity) else "identity"
            return (f"BasicBlock {k}x{k}: {mod.conv1.in_channels}->{mod.conv1.out_channels}, "
                    f"stride {mod.conv1.stride[0]}, shortcut {proj}")
        if isinstance(mod, torch.nn.Linear):
            return f"global avg pool + Linear {mod.in_features}->{mod.out_features}"
        conv = next(m for m in mod.modules() if isinstance(m, torch.nn.Conv2d))
        k = conv.kernel_size[0]
        return f"stem: Conv {k}x{k} {conv.in_channels}->{conv.out_channels} + BN + ReLU"

    def hook(name, mod):
        def fn(_, __, out):
            rows.append({"name": name, "layer": describe(mod), "output": list(out.shape),
                         "params": sum(p.numel() for p in mod.parameters())})
        return mod.register_forward_hook(fn)

    hooks.append(hook("stem", model.stem))
    for i, block in enumerate(model.layers):
        hooks.append(hook(f"layers.{i}", block))
    hooks.append(hook("fc", model.fc))
    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(torch.zeros(input_size))
    model.train(was_training)
    for h in hooks:
        h.remove()
    return rows


def build(blocks, width, kernel):
    return train.ResNet(tuple(blocks), width, kernel)


def report(blocks=None, width=None, kernel=None, run=None, torchinfo=True):
    """Print the model card and return its numbers as a dict. With `run`, the config comes from that
    run's summary.json (explicit blocks/width/kernel override it) and the test accuracy is reported."""
    summary = None
    if run is not None:
        run = Path(run)
        summary = json.loads((run / "summary.json").read_text())
        cfg = summary["args"]
        blocks = blocks or cfg["blocks"]
        width = width or cfg["width"]
        kernel = kernel or cfg["kernel"]
    blocks = blocks or [3, 3, 3]
    width = width or 32
    kernel = kernel or 5

    model = build(blocks, width, kernel)
    trainable, total = count_params(model)
    depth = 2 * sum(blocks) + 2
    widths = [width * 2 ** i for i in range(len(blocks))]
    rows = layer_table(model)

    print(f"ResNet-{depth}: {kernel}x{kernel} convs, blocks {list(blocks)}, stage widths {widths}, "
          f"input {list(INPUT_SIZE[1:])}")
    print(f"{'layer':<10}{'description':<62}{'output':<18}{'params':>12}")
    print("-" * 102)
    for r in rows:
        print(f"{r['name']:<10}{r['layer']:<62}{str(r['output']):<18}{r['params']:>12,}")
    print("-" * 102)
    if trainable != total:
        print(f"total parameters:     {total:,}")
    ok = trainable <= train.PARAM_LIMIT
    print(f"TRAINABLE PARAMETERS: {trainable:,}  (limit {train.PARAM_LIMIT:,}: "
          f"{'OK' if ok else 'OVER THE LIMIT'}, {train.PARAM_LIMIT - trainable:+,} to spare)")
    if summary is not None and summary["test_acc"] is None:
        print(f"FINAL TEST ACCURACY:  n/a (trained on all labeled images, no local test split)  [{run}]")
    elif summary is not None:
        print(f"FINAL TEST ACCURACY:  {summary['test_acc']:.4f}  "
              f"({summary['split']['test']:,} held-out local test images, best.pt from epoch "
              f"{summary['best_epoch']}, val acc {summary['best_val_acc']:.4f})  [{run}]")

    if torchinfo:
        try:
            import torchinfo as ti
            info = ti.summary(model, input_size=INPUT_SIZE, verbose=0)
            print(f"torchinfo cross-check: {info.trainable_params:,} trainable params, "
                  f"{info.total_mult_adds / 1e6:,.0f}M mult-adds per image")
        except ImportError:
            print("(torchinfo not installed; `pip install torchinfo` for an independent cross-check)")

    return {"depth": depth, "blocks": list(blocks), "width": width, "kernel": kernel, "widths": widths,
            "trainable_params": trainable, "total_params": total, "under_limit": ok, "layers": rows,
            "test_acc": None if summary is None else summary["test_acc"]}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--blocks", type=int, nargs="+", help="basic blocks per stage (default: run's or 3 3 3)")
    p.add_argument("--width", type=int, help="channels in the first stage (default: run's or 32)")
    p.add_argument("--kernel", type=int, help="conv kernel size (default: run's or 5)")
    p.add_argument("--run", type=Path, help="run directory with summary.json; also prints its test accuracy")
    p.add_argument("--no-torchinfo", action="store_true", help="skip the optional torchinfo cross-check")
    a = p.parse_args()
    report(a.blocks, a.width, a.kernel, a.run, torchinfo=not a.no_torchinfo)


if __name__ == "__main__":
    main()
