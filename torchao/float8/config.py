# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import enum
from dataclasses import dataclass
from typing import Optional

import torch


class ScalingType(enum.Enum):
    DELAYED = "delayed"
    DYNAMIC = "dynamic"
    STATIC = "static"

    def short_str(self):
        if self is ScalingType.DELAYED:
            return "del"
        elif self is ScalingType.DYNAMIC:
            return "dyn"
        else:
            assert self is ScalingType.STATIC
            return "sta"


class ScalingGranularity(enum.Enum):
    """
    Defines the granularity of scaling strategies for casting to float8
    """

    # A single scaling factor for the entire tensor
    TENSORWISE = "tensorwise"
    # Scaling factors computed along one axis of the tensor, reducing it to
    # size 1.
    AXISWISE = "axiswise"

    def short_str(self):
        if self is ScalingGranularity.TENSORWISE:
            return "ten"
        else:
            assert self is ScalingGranularity.AXISWISE
            return "axs"


@dataclass(frozen=True)
class CastConfig:
    """
    Configuration for maybe casting a single tensor to float8
    """

    scaling_type: ScalingType = ScalingType.DYNAMIC
    scaling_granularity: ScalingGranularity = ScalingGranularity.TENSORWISE
    static_scale: Optional[torch.Tensor] = None
    # If True, this tensor is not scaled to float8 and left in its original
    # precision.
    # TODO(ideally before this PR lands): a better name for this
    keep_in_original_precision: bool = False

    def __post_init__(self):
        if self.scaling_type is ScalingType.STATIC:
            assert self.static_scale is not None, \
                "static_scale must be specified for static scaling"
        if self.scaling_granularity is ScalingGranularity.AXISWISE:
            assert self.scaling_type is ScalingType.DYNAMIC, \
                "only dynamic scaling type is supported for axiswise scaling granularity"

@dataclass(frozen=True)
class DelayedScalingConfig:
    """
    Configuration for delayed scaling.

    Note: for now, `history_len` values must be the same for all layers in the
    model using delayed scaling.

    TODO(future): serialization for recipes
    """

    # Controls the history length of amax buffers
    history_len: int = 16

    # Controls the way to calculate current scale from amax history
    # TODO(future): add other functions as needed, hardcoded or user defined
    scale_fn_name: str = "max"

    def __post_init__(self):
        assert (
            self.scale_fn_name == "max"
        ), f"{self.scale_fn_name} is not implemented yet. Only max is supported for now."


@dataclass(frozen=True)
class Float8GemmConfig:
    """
    Configuration for a float8 gemm.
    """

    # If True, fast accumulation in lower precision is used.
    # Note: this flag is currently a no-op if emulation is turned on.
    use_fast_accum: bool = False


@dataclass(frozen=False)
class Float8LinearConfig:
    """
    Configuration for converting a `torch.nn.Linear` module to float8
    for training.
    """

    #
    # Per-tensor configuration for `input`, `weight`, `grad_output`
    #
    cast_config_input: CastConfig = CastConfig()
    cast_config_weight: CastConfig = CastConfig()
    cast_config_grad_output: CastConfig = CastConfig()

    #
    # Optional per-tensor configuration for `input`, `weight`, `grad_output` to
    # calculate `grad_weight`, `grad_input`, and `grad_weight` respectively.
    # If not specified, then the configuration from the  is reused.
    # TODO(future PR): maybe rename `cast_config_input` to 
    # `cast_config_input_for_output`, etc, to make the names consistent, 
    # will be BC-breaking.
    #
    cast_config_input_for_grad_weight: Optional[CastConfig] = None
    cast_config_weight_for_grad_input: Optional[CastConfig] = None
    cast_config_grad_output_for_grad_weight: Optional[CastConfig] = None

    #
    # Per-gemm configuration for gemms calculating `output`, `grad_input` and
    # `grad_weight`
    # TODO(this PR): throw warning if fast_accum False is used with axiswise scaling
    #
    gemm_config_output: Float8GemmConfig = Float8GemmConfig(use_fast_accum=True)
    gemm_config_grad_input: Float8GemmConfig = Float8GemmConfig()
    gemm_config_grad_weight: Float8GemmConfig = Float8GemmConfig()

    #
    # Per-linear configuration
    #

    # If True, on the first iteration of Float8Linear the amaxes will be
    # initialized with the incoming data. As of 2023-12-30, this doesn't work
    # with autocast + torch.compile + FSDP. Enabling this option is nice for
    # testing, but this is not necessary for real training jobs.
    enable_amax_init: bool = True

    # If True, pre-forward and post-forward functions are run. As of 2023-12-30,
    # this doesn't work with autocast + torch.compile + FSDP. Enabling this
    # option is useful for safety, but not strictly necessary.
    enable_pre_and_post_forward: bool = True

    # If True, then uses a tensor subclass for the float8 linear module's weight that
    # implements pre/post-all-gather methods to do float8 all-gather with FSDP2.
    enable_fsdp_float8_all_gather: bool = False

    # If True, then prior to performing the fp8 scaled mamtmul we will pad the
    # inner dimension of a (dim 1) and b (dim 2) with 0s. This is needed for matmuls
    # _scaled_mm since it has the strong constraint that for M,N,K  N, K must be a multiple of 16.
    # This can cause a memory spike however so we keep this off by default.
    pad_inner_dim: bool = False

    # If True, emulation is used instead of hardware accelerated gemm
    emulate: bool = False

    # Configuration for delayed scaling
    # Note: this is actually applied per-tensor, but only using the same
    # configuration for all tensors and layers in the model is currently
    # supported. If in the future we add support for a more fine grained
    # configuration, this field may move to per-tensor configs.
    delayed_scaling_config: DelayedScalingConfig = DelayedScalingConfig()

    def __post_init__(self):
        # populate the additional cast overrides, if the user did not specify them
        if self.cast_config_input_for_grad_weight is None:
            self.cast_config_input_for_grad_weight = self.cast_config_input
        if self.cast_config_weight_for_grad_input is None:
            self.cast_config_weight_for_grad_input = self.cast_config_weight
        if self.cast_config_grad_output_for_grad_weight is None:
            self.cast_config_grad_output_for_grad_weight = self.cast_config_grad_output

        # float8 all-gather only supports tensorwise, in the future may support blockwise
        if self.cast_config_weight.scaling_granularity != ScalingGranularity.TENSORWISE:
            assert not self.enable_fsdp_float8_all_gather, \
                f"enable_fsdp_float8_all_gather only supports tensorwise scaling granularity, got {self.cast_config_weight.scaling_granularity}"

        # save some characters in the compatibility checks below
        cc_i = self.cast_config_input
        cc_w = self.cast_config_weight
        cc_go = self.cast_config_grad_output
        cc_i2 = self.cast_config_input_for_grad_weight
        cc_w2 = self.cast_config_weight_for_grad_input
        cc_go2 = self.cast_config_grad_output_for_grad_weight

        # for now, we only have gemm kernels where both operands are scaled with the same
        # granularity. In the future this may be relaxed.
        assert cc_i.scaling_granularity == cc_w.scaling_granularity, \
            "incompatible scaling granularity for output"
        # assert cc_go.scaling_granularity == cc_w2.scaling_granularity, \
        #     "incompatible scaling granularity for grad_input"
        assert cc_i2.scaling_granularity == cc_go2.scaling_granularity, \
            "incompatible scaling granularity for grad_weight"

        # for now, we only have gemm kernels where both operands are either both
        # in high precision, or both in float8. In the future, this may be relaxed.
        # TODO(future): make the float8 check more precise with the specific dtypes.
        assert cc_i.keep_in_original_precision == cc_w.keep_in_original_precision, \
            "incompatible operand precision for output"
        assert cc_go.keep_in_original_precision == cc_w2.keep_in_original_precision, \
            "incompatible operand precision for grad_input"
        assert cc_i2.keep_in_original_precision == cc_go2.keep_in_original_precision, \
            "incompatible operand precision for grad_weight"


# If True, use 'fnuz' float8 types for calculations.
# Currently, ROCm only supports fnuz variants.
# TODO(future PR): move this to Float8LinearConfig
use_fnuz_dtype = False
