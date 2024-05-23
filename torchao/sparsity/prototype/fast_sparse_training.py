# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
import contextlib
import ctypes
import glob
import warnings
from functools import partial
from pathlib import Path
from typing import Any, Callable, Optional, Tuple, TypeVar, cast
import torch
from torch.sparse import SparseSemiStructuredTensor, SparseSemiStructuredTensorCUTLASS, SparseSemiStructuredTensorCUSPARSELT
from collections.abc import Iterable
from torch import nn

# pointwise op support

def semi_sparse_pointwise_op(func, types, args=(), kwargs=None, sparsify_like_args_list=()):
    """
    adds pointwise op support for semi-structured tensors
    """
    reference_sparse_tensor = None
    for tensor in args:
        if isinstance(tensor, SparseSemiStructuredTensor):
            reference_sparse_tensor = tensor
    assert reference_sparse_tensor is not None

    def handle_arg(i, tensor):
        if isinstance(tensor, torch.Tensor):
            if not isinstance(tensor, SparseSemiStructuredTensor):
                if i in sparsify_like_args_list:
                    tensor = semi_sparse_sparsify(tensor, pattern=reference_sparse_tensor)
                else:
                    raise ValueError(
                        f"Operation {func.__module__}.{func.__name__} on {type(reference_sparse_tensor)} requires all operands to "
                        f"be {type(reference_sparse_tensor)}, but operand {i} is a {type(tensor)}"
                    )
            else:
                if (
                    tensor.compressed_swizzled_bitmask is None
                    or reference_sparse_tensor.compressed_swizzled_bitmask is None
                    or tensor.compressed_swizzled_bitmask.data_ptr() != reference_sparse_tensor.compressed_swizzled_bitmask.data_ptr()
                    or tensor.compressed_swizzled_bitmask.stride() != reference_sparse_tensor.compressed_swizzled_bitmask.stride()
                ):
                    raise ValueError(
                        f"Operation {func.__module__}.{func.__name__} on {type(reference_sparse_tensor)} requires all operands to be "
                        f"{type(reference_sparse_tensor)} with the same sparsity pattern"
                    )
        return tensor

    args_updated = [ handle_arg(i, tensor) for i, tensor in enumerate(args) ]

    return reference_sparse_tensor.__class__(
        reference_sparse_tensor.shape,
        func(*[
                x.packed if isinstance(x, SparseSemiStructuredTensor) else x
                for x in args_updated
            ]),
        reference_sparse_tensor.meta,
        func(
            *[
                x.packed_t if isinstance(x, SparseSemiStructuredTensor)
                else x for x in args_updated
            ]
        ),
        reference_sparse_tensor.meta_t,
        reference_sparse_tensor.compressed_swizzled_bitmask,
    )


CUTLASS_POINTWISE_OP_DISPATCH_TABLE = {
    torch.ops.aten.relu: semi_sparse_pointwise_op,
    torch.ops.aten.gelu: semi_sparse_pointwise_op,
    torch.ops.aten.silu: semi_sparse_pointwise_op,
    torch.ops.aten.mul: partial(
        # `mul` BW in swiglu
        semi_sparse_pointwise_op,
        allow_sparsify_args_list=(0, 1),
    ),
    torch.ops.aten.add: semi_sparse_pointwise_op,
    # Note: for these ops, we allow the gradient to come in as a `torch.Tensor`
    # and we will run the sparsification right before calling the BW aten func
    torch.ops.aten.gelu_backward: partial(
        semi_sparse_pointwise_op,
        allow_sparsify_args_list=(0,)
    ),
    torch.ops.aten.silu_backward: partial(
        semi_sparse_pointwise_op,
        allow_sparsify_args_list=(0, 1)
    ),
    torch.ops.aten.threshold_backward: partial(  # relu BW
        semi_sparse_pointwise_op,
        allow_sparsify_args_list=(0,),
    ),
}

SparseSemiStructuredTensorCUTLASS._load_dispatch_table(CUTLASS_POINTWISE_OP_DISPATCH_TABLE)

# autograd support

if torch.__version__ >= "2.1.0":
    torch._dynamo.allow_in_graph(SparseSemiStructuredTensorCUSPARSELT)
    torch._dynamo.allow_in_graph(SparseSemiStructuredTensorCUTLASS)

GRADIENT_STE = "ste"
GRADIENT_DENSE = "dense"
GRADIENT_SPARSE = "sparse"

class _SparsifyFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: torch.Tensor, algo: str, backend: str):  # type: ignore[override]
        use_cutlass = (backend == "cutlass")
        if not isinstance(x, SparseSemiStructuredTensor):
            (packed, meta, packed_t, meta_t, bitmask) = torch._sparse_semi_structured_tile(
                x, algorithm=algo, use_cutlass=use_cutlass
            )
            cls = (
                SparseSemiStructuredTensorCUTLASS if use_cutlass else SparseSemiStructuredTensorCUSPARSELT
            )
            out = cls(
                x.shape,
                packed=packed,
                meta=meta,
                packed_t=packed_t,
                meta_t=meta_t,
                compressed_swizzled_bitmask=bitmask,
                requires_grad=False,
                fuse_transpose_cusparselt=True,
            )
        else:
            out = x.detach()

        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):  # type: ignore[override]
        return grad_out, None, None

class _SparsifyLikeFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: torch.Tensor, pattern: SparseSemiStructuredTensor, gradient=GRADIENT_SPARSE):  # type: ignore[override]
        assert isinstance(pattern, SparseSemiStructuredTensor)

        if not isinstance(pattern, SparseSemiStructuredTensorCUTLASS):
            raise NotImplementedError(
                "`sparsify_like(x, pattern)` is only implemented for CUTLASS backend"
            )
        if not pattern.compressed_swizzled_bitmask.is_contiguous():
            raise NotImplementedError(
                "`sparsify_like(x, pattern)` is not implemented when `bitmask` is transposed"
            )

        packed, packed_t = torch._sparse_semi_structured_apply(x, pattern.compressed_swizzled_bitmask)
        return pattern.__class__(
            x.shape,
            packed,
            pattern.meta,
            packed_t,
            pattern.meta_t,
            pattern.compressed_swizzled_bitmask,
            requires_grad=x.requires_grad,
        )

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):  # type: ignore[override]
        if ctx.gradient == GRADIENT_STE or isinstance(grad_out, SparseSemiStructuredTensor):
            return grad_out, None, None, None
        assert not isinstance(grad_out, SparseSemiStructuredTensor)
        assert grad_out.dtype == ctx.dtype

        if ctx.gradient == GRADIENT_DENSE:
            assert ctx.threads_masks.is_contiguous()
            return (
                torch._sparse_semi_structured_apply_dense(grad_out, ctx.threads_masks),
                None,
                None,
                None,
            )
        assert ctx.gradient == GRADIENT_SPARSE

        packed, _, packed_t, _ = torch._sparse_semi_structured_tile(
            grad_out, ctx.threads_masks, backend="cutlass"
        )
        return (
            SparseSemiStructuredTensorCUTLASS(
                grad_out.shape,
                packed,
                ctx.meta,
                packed_t,
                ctx.meta_t,
                ctx.threads_masks,
                requires_grad=grad_out.requires_grad,
            ),
            None,
            None,
            None,
        )
        return grad_out, None

# We want to use `torch._dynamo.allow_in_graph` as a decorator
# (see https://fburl.com/workplace/uimiz0mf) but it breaks mypy.
# This is a hack to work around this
F = TypeVar("F", bound=Callable[..., Any])
def allow_in_graph(func: F) -> F:
    return cast(F, torch._dynamo.allow_in_graph(func))

@allow_in_graph
def semi_sparse_sparsify(
    x: torch.Tensor,
    pattern: SparseSemiStructuredTensor = None,
    algo: str = "",
    backend: str = "cutlass",
) -> SparseSemiStructuredTensor:
    """
    Sparsifies a dense tensor into a semi-structured tensor.
    When pattern is provided, we will sparsify the tensor according using the same mask as the provided sparse tensor.
    Otherwise, we sparsify using the algorithm/backend specified
    """
    if pattern is None:
        return _SparsifyFunc.apply(x, algo, backend)
    else:
        if not isinstance(pattern, SparseSemiStructuredTensor):
            raise ValueError(
                f"`pattern` must be a `SparseSemiStructuredTensor` but got a {type(pattern)}"
            )
        return _SparsifyLikeFunc.apply(x, pattern)

# user API

class SemiSparseLinear(torch.nn.Linear):
    """
    Replacement nn.Linear that supports runtime weight/activation sparsity
    """

    def forward(self, x):
        if self.weight_sparsity:
            sparse_weight = semi_sparse_sparsify(self.weight, backend="cusparselt")
            return torch.nn.functional.linear(x, sparse_weight, self.bias)
        else:
            sparse_x = semi_sparse_sparsify(x, backend="cusparselt")
            return torch.nn.functional.linear(sparse_x, self.weight, self.bias)

    @classmethod
    def from_dense(cls, linear, weight_sparsity=True):
        mod = cls(linear.in_features, linear.out_features)
        mod.weight = linear.weight
        mod.bias = linear.bias
        mod.weight_sparsity = weight_sparsity
        return mod


def swap_linear_with_semi_sparse_linear_(model, config, current=""):
    name_to_child = dict(model.named_children())
    for name, child in name_to_child.items():
        fqn = f"{current}.{name}" if current else name
        if isinstance(child, torch.nn.Linear):
            if fqn in config:
                setattr(model, name, SemiSparseLinear.from_dense(child))
                del child
        else:
            swap_linear_with_semi_sparse_linear_(child, config, current=fqn)