"""Paged KV block pool with reference counting and hash-based prefix caching.

Full blocks are keyed by a chained hash of (parent block hash, block tokens), so two prompts
that share a prefix of N full blocks share those N physical blocks. Freed blocks stay in the
cache (still hashed) and are evicted least-recently-used only when the pool runs dry.
"""

from __future__ import annotations

import hashlib
import struct
from collections import OrderedDict
from collections.abc import Sequence

ROOT_HASH = b"\0" * 16


def block_hash(parent: bytes, tokens: Sequence[int]) -> bytes:
    h = hashlib.blake2b(parent, digest_size=16)
    h.update(struct.pack(f"<{len(tokens)}q", *tokens))
    return h.digest()


class BlockPool:
    def __init__(self, num_blocks: int, block_size: int, enable_prefix_cache: bool = True):
        if num_blocks < 1 or block_size < 1:
            raise ValueError("num_blocks and block_size must be >= 1")
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.enable_prefix_cache = enable_prefix_cache
        self._ref = [0] * num_blocks
        self._hash: list[bytes | None] = [None] * num_blocks
        self._cache: dict[bytes, int] = {}
        # Free blocks, oldest first. Uncached blocks are pushed to the front so they are
        # reused before any block that still holds a reusable prefix.
        self._free: OrderedDict[int, None] = OrderedDict((i, None) for i in range(num_blocks))

    @property
    def num_free(self) -> int:
        return len(self._free)

    def match_prefix(self, tokens: Sequence[int]) -> tuple[list[int], bytes]:
        """Longest cached run of full blocks for ``tokens``; takes a reference on each.

        At least one token is always left uncached so the last position can be computed to
        produce logits. Returns (block ids, hash of the last matched block).
        """
        if not self.enable_prefix_cache:
            return [], ROOT_HASH
        bs = self.block_size
        blocks: list[int] = []
        parent = ROOT_HASH
        for i in range((len(tokens) - 1) // bs):
            h = block_hash(parent, tokens[i * bs : (i + 1) * bs])
            b = self._cache.get(h)
            if b is None:
                break
            self._take(b)
            blocks.append(b)
            parent = h
        return blocks, parent

    def _take(self, b: int) -> None:
        if self._ref[b] == 0:
            del self._free[b]
        self._ref[b] += 1

    def allocate(self) -> int | None:
        if not self._free:
            return None
        b, _ = self._free.popitem(last=False)
        h = self._hash[b]
        if h is not None:  # evict the stale cache entry
            if self._cache.get(h) == b:
                del self._cache[h]
            self._hash[b] = None
        self._ref[b] = 1
        return b

    def commit(self, b: int, h: bytes) -> int:
        """Register full block ``b`` under hash ``h``.

        If an identical block is already cached, the caller should switch to it: returns the
        canonical block id (``b`` itself if it became canonical).
        """
        if not self.enable_prefix_cache:
            return b
        other = self._cache.get(h)
        if other is not None and other != b:
            self._take(other)
            self.free([b])
            return other
        self._hash[b] = h
        self._cache[h] = b
        return b

    def free(self, blocks: Sequence[int]) -> None:
        # Reverse so a sequence's tail is evicted before its head (heads are shared more).
        for b in reversed(blocks):
            self._ref[b] -= 1
            if self._ref[b] == 0:
                self._free[b] = None
                if self._hash[b] is None:
                    self._free.move_to_end(b, last=False)

    def ref(self, b: int) -> int:
        return self._ref[b]
