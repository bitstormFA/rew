#!/usr/bin/env python3
"""PyTorch equivalents for the Rechenwerk MNIST example benchmarks.

The script emits one JSON object per benchmark row. It intentionally avoids
torchvision and external data files so it can run anywhere PyTorch is present.
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Callable


FRAMEWORK = "PyTorch"
IMAGE_SIDE = 28
IMAGE_PIXELS = IMAGE_SIDE * IMAGE_SIDE
NUM_CLASSES = 10
MLP_HIDDEN_DIM = 64
CNN_CONV1_CHANNELS = 8
CNN_CONV2_CHANNELS = 16
CNN_HIDDEN_DIM = 32
CNN_FEATURE_SIDE = IMAGE_SIDE // 4
CNN_FEATURE_COUNT = CNN_CONV2_CHANNELS * CNN_FEATURE_SIDE * CNN_FEATURE_SIDE
VIT_PATCH_SIZE = 7
VIT_NUM_PATCHES = (IMAGE_SIDE // VIT_PATCH_SIZE) * (IMAGE_SIDE // VIT_PATCH_SIZE)
VIT_PATCH_DIM = VIT_PATCH_SIZE * VIT_PATCH_SIZE
VIT_EMBED_DIM = 16
VIT_NUM_HEADS = 1
VIT_HEAD_DIM = VIT_EMBED_DIM // VIT_NUM_HEADS
VIT_TOKEN_COUNT = VIT_NUM_PATCHES + 1
VIT_MLP_DIM = 32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threads", type=int, default=0)
    return parser.parse_args()


def emit_base(example: str, status: str, args: argparse.Namespace) -> dict:
    return {
        "example": example,
        "framework": FRAMEWORK,
        "device": args.device,
        "status": status,
        "batch_size": args.batch_size,
        "warmup": args.warmup,
        "iterations": args.iterations,
    }


def emit_skipped(example: str, args: argparse.Namespace, reason: str) -> None:
    row = emit_base(example, "skipped", args)
    row.update(
        {
            "reason": reason,
            "first_step_ms": None,
            "mean_step_ms": None,
            "samples_per_s": None,
            "last_loss": None,
        }
    )
    print(json.dumps(row), flush=True)


def emit_failed(example: str, args: argparse.Namespace, reason: str) -> None:
    row = emit_base(example, "failed", args)
    row.update(
        {
            "reason": reason,
            "first_step_ms": None,
            "mean_step_ms": None,
            "samples_per_s": None,
            "last_loss": None,
        }
    )
    print(json.dumps(row), flush=True)


def emit_ok(
    example: str,
    args: argparse.Namespace,
    first_step_ms: float,
    mean_step_ms: float,
    last_loss: float,
) -> None:
    samples_per_s = (
        args.batch_size * 1000.0 / mean_step_ms if mean_step_ms > 0.0 else 0.0
    )
    row = emit_base(example, "ok", args)
    row.update(
        {
            "reason": "",
            "first_step_ms": first_step_ms,
            "mean_step_ms": mean_step_ms,
            "samples_per_s": samples_per_s,
            "last_loss": last_loss,
        }
    )
    print(json.dumps(row), flush=True)


def sync(device: str, torch_module) -> None:
    if device.startswith("cuda"):
        torch_module.cuda.synchronize()


def labels_one_hot(torch_module, batch_size: int, device) -> tuple:
    labels = torch_module.arange(batch_size, device=device) % NUM_CLASSES
    one_hot = torch_module.nn.functional.one_hot(
        labels, num_classes=NUM_CLASSES
    ).to(dtype=torch_module.float32)
    return labels, one_hot


def softmax_cross_entropy(torch_module, logits, one_hot):
    logsumexp = torch_module.logsumexp(logits, dim=1)
    label_logits = (one_hot * logits).sum(dim=1)
    return (logsumexp - label_logits).mean()


def init_linear(torch_module, in_features: int, out_features: int, device):
    bound = (1.0 / in_features) ** 0.5
    weight = torch_module.empty(
        in_features, out_features, device=device
    ).uniform_(-bound, bound)
    bias = torch_module.zeros(out_features, device=device)
    weight.requires_grad_(True)
    bias.requires_grad_(True)
    return weight, bias


def init_conv2d(
    torch_module, in_channels: int, out_channels: int, kernel_size: int, device
):
    fan_in = in_channels * kernel_size * kernel_size
    bound = (1.0 / fan_in) ** 0.5
    weight = torch_module.empty(
        out_channels, in_channels, kernel_size, kernel_size, device=device
    ).uniform_(-bound, bound)
    bias = torch_module.zeros(out_channels, device=device)
    weight.requires_grad_(True)
    bias.requires_grad_(True)
    return weight, bias


def sgd_step(torch_module, params: list, lr: float) -> None:
    with torch_module.no_grad():
        for param in params:
            param -= lr * param.grad
            param.grad = None


def measure(
    step: Callable[[], float], args: argparse.Namespace, device: str, torch_module
) -> tuple[float, float, float]:
    start = time.perf_counter()
    last_loss = step()
    sync(device, torch_module)
    first_step_ms = (time.perf_counter() - start) * 1000.0

    for _ in range(args.warmup):
        last_loss = step()
    sync(device, torch_module)

    total_ms = 0.0
    for _ in range(args.iterations):
        start = time.perf_counter()
        last_loss = step()
        sync(device, torch_module)
        total_ms += (time.perf_counter() - start) * 1000.0
    return first_step_ms, total_ms / args.iterations, last_loss


def bench_mlp(args: argparse.Namespace, torch_module, device) -> None:
    torch_module.manual_seed(0xC0FFEE)
    x = torch_module.rand(args.batch_size, IMAGE_PIXELS, device=device)
    _, y = labels_one_hot(torch_module, args.batch_size, device)
    w1, b1 = init_linear(torch_module, IMAGE_PIXELS, MLP_HIDDEN_DIM, device)
    w2, b2 = init_linear(torch_module, MLP_HIDDEN_DIM, NUM_CLASSES, device)
    params = [w1, b1, w2, b2]

    def step() -> float:
        logits = torch_module.relu(x @ w1 + b1) @ w2 + b2
        loss = softmax_cross_entropy(torch_module, logits, y)
        loss.backward()
        value = float(loss.detach().cpu().item())
        sgd_step(torch_module, params, 0.01)
        return value

    first_ms, mean_ms, last_loss = measure(step, args, args.device, torch_module)
    emit_ok("mnist_mlp", args, first_ms, mean_ms, last_loss)


def bench_cnn(args: argparse.Namespace, torch_module, device) -> None:
    torch_module.manual_seed(0xCAFE)
    x = torch_module.rand(args.batch_size, 1, IMAGE_SIDE, IMAGE_SIDE, device=device)
    _, y = labels_one_hot(torch_module, args.batch_size, device)
    c1w, c1b = init_conv2d(
        torch_module, 1, CNN_CONV1_CHANNELS, 3, device
    )
    c2w, c2b = init_conv2d(
        torch_module, CNN_CONV1_CHANNELS, CNN_CONV2_CHANNELS, 3, device
    )
    fc1w, fc1b = init_linear(
        torch_module, CNN_FEATURE_COUNT, CNN_HIDDEN_DIM, device
    )
    fc2w, fc2b = init_linear(torch_module, CNN_HIDDEN_DIM, NUM_CLASSES, device)
    params = [c1w, c1b, c2w, c2b, fc1w, fc1b, fc2w, fc2b]

    def step() -> float:
        h1 = torch_module.relu(
            torch_module.nn.functional.conv2d(x, c1w, c1b, padding=1)
        )
        p1 = torch_module.nn.functional.max_pool2d(h1, kernel_size=2, stride=2)
        h2 = torch_module.relu(
            torch_module.nn.functional.conv2d(p1, c2w, c2b, padding=1)
        )
        p2 = torch_module.nn.functional.max_pool2d(h2, kernel_size=2, stride=2)
        flat = p2.flatten(start_dim=1)
        h3 = torch_module.relu(flat @ fc1w + fc1b)
        logits = h3 @ fc2w + fc2b
        loss = softmax_cross_entropy(torch_module, logits, y)
        loss.backward()
        value = float(loss.detach().cpu().item())
        sgd_step(torch_module, params, 0.05)
        return value

    first_ms, mean_ms, last_loss = measure(step, args, args.device, torch_module)
    emit_ok("mnist_cnn", args, first_ms, mean_ms, last_loss)


def dense3(torch_module, x, weight, bias):
    batch, tokens, in_features = x.shape
    y = x.reshape(batch * tokens, in_features) @ weight + bias
    return y.reshape(batch, tokens, weight.shape[1])


def forward_vit(torch_module, params: list, x):
    (
        patch_w,
        patch_b,
        class_token,
        position,
        query_w,
        query_b,
        key_w,
        key_b,
        value_w,
        value_b,
        proj_w,
        proj_b,
        mlp1_w,
        mlp1_b,
        mlp2_w,
        mlp2_b,
        head_w,
        head_b,
    ) = params
    batch = x.shape[0]
    patches = x.reshape(batch * VIT_NUM_PATCHES, VIT_PATCH_DIM)
    patch_tokens = (patches @ patch_w + patch_b).reshape(
        batch, VIT_NUM_PATCHES, VIT_EMBED_DIM
    )
    tokens = torch_module.cat([class_token, patch_tokens], dim=1)
    tokens = tokens + position

    q = dense3(torch_module, tokens, query_w, query_b)
    k = dense3(torch_module, tokens, key_w, key_b)
    v = dense3(torch_module, tokens, value_w, value_b)
    q = q.reshape(batch, VIT_TOKEN_COUNT, VIT_NUM_HEADS, VIT_HEAD_DIM).permute(
        0, 2, 1, 3
    )
    k = k.reshape(batch, VIT_TOKEN_COUNT, VIT_NUM_HEADS, VIT_HEAD_DIM).permute(
        0, 2, 1, 3
    )
    v = v.reshape(batch, VIT_TOKEN_COUNT, VIT_NUM_HEADS, VIT_HEAD_DIM).permute(
        0, 2, 1, 3
    )
    scores = (q @ k.transpose(-2, -1)) * (VIT_HEAD_DIM ** -0.5)
    weights = torch_module.softmax(scores, dim=-1)
    context = weights @ v
    merged = context.permute(0, 2, 1, 3).reshape(
        batch, VIT_TOKEN_COUNT, VIT_EMBED_DIM
    )
    tokens = tokens + dense3(torch_module, merged, proj_w, proj_b)
    hidden = torch_module.nn.functional.gelu(
        dense3(torch_module, tokens, mlp1_w, mlp1_b), approximate="tanh"
    )
    tokens = tokens + dense3(torch_module, hidden, mlp2_w, mlp2_b)
    cls = tokens[:, 0:1, :].reshape(batch, VIT_EMBED_DIM)
    return cls @ head_w + head_b


def init_vit_params(torch_module, batch_size: int, device) -> list:
    params = []
    patch_w, patch_b = init_linear(torch_module, VIT_PATCH_DIM, VIT_EMBED_DIM, device)
    query_w, query_b = init_linear(torch_module, VIT_EMBED_DIM, VIT_EMBED_DIM, device)
    key_w, key_b = init_linear(torch_module, VIT_EMBED_DIM, VIT_EMBED_DIM, device)
    value_w, value_b = init_linear(torch_module, VIT_EMBED_DIM, VIT_EMBED_DIM, device)
    proj_w, proj_b = init_linear(torch_module, VIT_EMBED_DIM, VIT_EMBED_DIM, device)
    mlp1_w, mlp1_b = init_linear(torch_module, VIT_EMBED_DIM, VIT_MLP_DIM, device)
    mlp2_w, mlp2_b = init_linear(torch_module, VIT_MLP_DIM, VIT_EMBED_DIM, device)
    head_w, head_b = init_linear(torch_module, VIT_EMBED_DIM, NUM_CLASSES, device)
    class_token = torch_module.empty(
        batch_size, 1, VIT_EMBED_DIM, device=device
    ).uniform_(-0.02, 0.02)
    position = torch_module.empty(
        batch_size, VIT_TOKEN_COUNT, VIT_EMBED_DIM, device=device
    ).uniform_(-0.02, 0.02)
    class_token.requires_grad_(True)
    position.requires_grad_(True)
    params.extend(
        [
            patch_w,
            patch_b,
            class_token,
            position,
            query_w,
            query_b,
            key_w,
            key_b,
            value_w,
            value_b,
            proj_w,
            proj_b,
            mlp1_w,
            mlp1_b,
            mlp2_w,
            mlp2_b,
            head_w,
            head_b,
        ]
    )
    return params


def bench_vit(args: argparse.Namespace, torch_module, device) -> None:
    torch_module.manual_seed(0xB17)
    x = torch_module.rand(args.batch_size, IMAGE_PIXELS, device=device)
    _, y = labels_one_hot(torch_module, args.batch_size, device)
    params = init_vit_params(torch_module, args.batch_size, device)

    def step() -> float:
        logits = forward_vit(torch_module, params, x)
        loss = softmax_cross_entropy(torch_module, logits, y)
        loss.backward()
        value = float(loss.detach().cpu().item())
        sgd_step(torch_module, params, 0.03)
        return value

    first_ms, mean_ms, last_loss = measure(step, args, args.device, torch_module)
    emit_ok("mnist_vit", args, first_ms, mean_ms, last_loss)


def main() -> int:
    args = parse_args()
    if args.warmup < 0 or args.iterations <= 0 or args.batch_size <= 0:
        raise SystemExit(
            "warmup must be non-negative; iterations and batch-size must be positive"
        )

    try:
        import torch
    except ModuleNotFoundError as exc:
        reason = f"PyTorch not installed ({exc})"
        emit_skipped("mnist_mlp", args, reason)
        emit_skipped("mnist_cnn", args, reason)
        emit_skipped("mnist_vit", args, reason)
        return 0

    if args.threads > 0:
        torch.set_num_threads(args.threads)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        emit_skipped("mnist_mlp", args, "CUDA requested but torch.cuda is unavailable")
        emit_skipped("mnist_cnn", args, "CUDA requested but torch.cuda is unavailable")
        emit_skipped("mnist_vit", args, "CUDA requested but torch.cuda is unavailable")
        return 0

    device = torch.device(args.device)
    failures = 0
    for name, fn in (
        ("mnist_mlp", bench_mlp),
        ("mnist_cnn", bench_cnn),
        ("mnist_vit", bench_vit),
    ):
        try:
            fn(args, torch, device)
        except Exception as exc:  # noqa: BLE001 - benchmark rows report failures.
            emit_failed(name, args, repr(exc))
            failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
