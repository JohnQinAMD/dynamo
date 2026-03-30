# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Multimodal utilities for Dynamo components."""

import logging
from collections.abc import Callable

from dynamo.common.constants import EmbeddingTransferMode
from dynamo.common.multimodal.async_encoder_cache import AsyncEncoderCache

try:
    from dynamo.common.multimodal.embedding_transfer import (
        AbstractEmbeddingReceiver,
        AbstractEmbeddingSender,
        LocalEmbeddingReceiver,
        LocalEmbeddingSender,
        NixlReadEmbeddingReceiver,
        NixlReadEmbeddingSender,
        NixlWriteEmbeddingReceiver,
        NixlWriteEmbeddingSender,
        TransferRequest,
    )

    _HAS_NIXL = True
except ImportError:
    logging.warning(
        "nixl/rixl not available — multimodal embedding transfer disabled. "
        "Install nixl or rixl for NIXL-based embedding transfer."
    )
    _HAS_NIXL = False

from dynamo.common.multimodal.image_loader import ImageLoader

if _HAS_NIXL:
    EMBEDDING_SENDER_FACTORIES: dict[
        EmbeddingTransferMode, Callable[[], AbstractEmbeddingSender]
    ] = {
        EmbeddingTransferMode.LOCAL: LocalEmbeddingSender,
        EmbeddingTransferMode.NIXL_WRITE: NixlWriteEmbeddingSender,
        EmbeddingTransferMode.NIXL_READ: NixlReadEmbeddingSender,
    }

    EMBEDDING_RECEIVER_FACTORIES: dict[
        EmbeddingTransferMode, Callable[[], AbstractEmbeddingReceiver]
    ] = {
        EmbeddingTransferMode.LOCAL: LocalEmbeddingReceiver,
        EmbeddingTransferMode.NIXL_WRITE: NixlWriteEmbeddingReceiver,
        EmbeddingTransferMode.NIXL_READ: lambda: NixlReadEmbeddingReceiver(
            max_items=0
        ),
    }
else:
    EMBEDDING_SENDER_FACTORIES = {}
    EMBEDDING_RECEIVER_FACTORIES = {}

__all__ = [
    "AsyncEncoderCache",
    "EMBEDDING_RECEIVER_FACTORIES",
    "EMBEDDING_SENDER_FACTORIES",
    "ImageLoader",
]
