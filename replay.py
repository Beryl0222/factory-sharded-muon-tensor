"""采集回放脚本：``python replay.py`` 等价于 ``python -m shard_muon replay``。"""
from shard_muon.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["replay"]))
