"""兼容入口：规范实现位于 :mod:`shard_muon`。"""
from shard_muon import *  # noqa: F401,F403
import shard_muon as _impl

for _name in getattr(_impl, "__all__", []):
    globals()[_name] = getattr(_impl, _name)

shard_muon_update = _impl.shard_muon_update
