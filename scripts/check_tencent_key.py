"""Is the Tencent credential shaped like a real SecretId/SecretKey? (never prints values)

`AuthFailure.SecretIdNotFound` is returned after a successful TLS round-trip, so the
network, the signature code and the action names are all fine and the credential
itself is the problem. This script distinguishes the shapes without printing a single
character of the secret: a real SecretId starts with `AKID` and is 36 characters; a
real SecretKey is 32; and an account's *creation-engine* key or a key copied from
another cloud fails both checks.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from pah.app.env import load_env_file  # noqa: E402


def describe(name: str, value: str, *, prefix: str, length: int) -> bool:
    ok = value.startswith(prefix) and len(value) == length
    mark = "OK  " if ok else "FAIL"
    shape = f"starts_with_{prefix}={value.startswith(prefix)} len={len(value)} (want {length})"
    print(f"  {mark} {name:26s} {shape}")
    return ok


def main() -> int:
    load_env_file(REPO / ".env")
    import os

    sid = os.environ.get("TENCENTCLOUD_SECRET_ID", "")
    skey = os.environ.get("TENCENTCLOUD_SECRET_KEY", "")
    print(f"source  {REPO / '.env'}  (values never printed)\n")
    sid_ok = describe("TENCENTCLOUD_SECRET_ID", sid, prefix="AKID", length=36)
    key_ok = describe("TENCENTCLOUD_SECRET_KEY", skey, prefix="", length=32)
    print()
    if not sid_ok:
        print(
            "SecretId 形状不对：\n"
            "  * 真实值以 `AKID` 开头、36 位（新密钥可能是 36 位；少数旧密钥不同）；\n"
            "  * 去 https://console.cloud.tencent.com/cam/capi 重新复制，确认状态=启用；\n"
            "  * 若你填的是「混元3D 创作引擎」页面上的 API Key，那不是 CAM 密钥，不能用于 API。"
        )
    if not key_ok:
        print("SecretKey 形状不对：应为 32 位，去同一页面重新复制（只显示一次）。")
    if sid_ok and key_ok:
        print("形状检查通过；若仍报 SecretIdNotFound，检查密钥是否已被禁用/删除，或用的是子账号但没授权。")
    return 0 if (sid_ok and key_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
