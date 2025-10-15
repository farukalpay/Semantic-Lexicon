# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Callable, Dict, List, Sequence


def build_alias_to_token_ids(
    vocab: Sequence[str],
    aliases: Sequence[str],
    normalize: Callable[[str], str] = lambda s: s.lower(),
) -> Dict[str, List[int]]:
    mapping: Dict[str, List[int]] = {}
    normalised_vocab = [normalize(token) for token in vocab]
    for alias in aliases:
        alias_norm = normalize(alias)
        ids = [index for index, token in enumerate(normalised_vocab) if token == alias_norm]
        if ids:
            mapping.setdefault(alias_norm, []).extend(ids)
    return mapping
