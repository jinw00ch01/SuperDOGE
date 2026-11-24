"""다른 PC 의 절대경로로 저장된 asset 경로를 이 checkout 기준으로 고친다.

vehicle_v3.usd 의 MDL 재질은 저장한 PC 의 절대경로
(.../a2_isaac/isaac_sim/assets/...)를 들고 있어, checkout 위치가 다르면 재질을 못
읽는다 (물리에는 영향이 없고, 렌더에서 로버가 빨갛게 나온다). 원본 파일은 건드리지
않고 임시 복사본을 만들어 그 안의 경로만 바꾼다.
"""
from __future__ import annotations

import os
import shutil
import tempfile

ISAAC_SIM = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def localize_asset_paths(usd_path: str) -> int:
    """usd_path(쓸 수 있는 복사본) 안의 깨진 절대경로를 치환한다. 치환한 개수를 돌려준다.

    존재하지 않는 절대경로 중 '/isaac_sim/assets/' 를 포함하고, 이 checkout 에 같은
    파일이 있는 것만 바꾼다.
    """
    from pxr import Sdf, UsdUtils

    layer = Sdf.Layer.FindOrOpen(usd_path)
    if layer is None:
        return 0
    marker = "/isaac_sim/assets/"
    count = 0

    def _fix(path: str) -> str:
        nonlocal count
        raw = path[5:] if path.startswith("file:") else path
        if not raw.startswith("/") or os.path.exists(raw):
            return path
        i = raw.find(marker)
        if i < 0:
            return path
        local = ISAAC_SIM + raw[i + len("/isaac_sim"):]
        if not os.path.exists(local):
            return path
        count += 1
        return local

    UsdUtils.ModifyAssetPaths(layer, _fix)
    if count:
        layer.Save()
    return count


def localized_copy(usd_path: str, name: str = "") -> tuple:
    """usd_path 를 임시 폴더에 복사하고 경로를 고친다. return: (복사본 경로, 치환 개수)."""
    out_dir = tempfile.mkdtemp(prefix="a2_isaac_usd_")
    dst = os.path.join(out_dir, name or os.path.basename(usd_path))
    shutil.copyfile(usd_path, dst)
    return dst, localize_asset_paths(dst)
