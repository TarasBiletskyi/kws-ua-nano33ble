"""Compile firmware/kws_nano33/frontend.c into a Windows DLL with MSVC, so that the PC runs exactly the code of the board.

Uses a compiler that comes with Visual Studio: the regular VC tools (needs vcvars64.bat) or ScopeCppSDK, which works without
any environment setup. Output: tools/out/frontend_pc/frontend.dll, loaded by tools/frontend_pc.py.

    python tools/build_frontend_pc.py [--vs "C:/Program Files/Microsoft Visual Studio/18/Professional"]
"""
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FIRMWARE = REPO / "firmware" / "kws_nano33"
OUT_DIR = REPO / "tools" / "out" / "frontend_pc"
DLL = OUT_DIR / "frontend.dll"
EXPORTS = ["fe_reset", "fe_push_frame", "fe_window_ready", "fe_finish_window", "fe_process_clip", "fe_power_spectrum",
           "fe_get_logmel", "fe_state_size"]


def find_scope_sdk(vs_arg: str | None) -> Path:
    roots = [Path(vs_arg)] if vs_arg else sorted(Path(os.environ.get("ProgramFiles", r"C:\Program Files"), "Microsoft Visual Studio").glob("*/*"), reverse=True)
    for root in roots:
        sdk = root / "SDK" / "ScopeCppSDK" / "vc15"
        if (sdk / "VC" / "bin" / "cl.exe").is_file():
            return sdk
    raise SystemExit("cl.exe of ScopeCppSDK not found in Visual Studio; pass --vs")


def build(vs_arg: str | None = None) -> Path:
    sdk = find_scope_sdk(vs_arg)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["INCLUDE"] = ";".join(str(sdk / p) for p in ["VC/include", "SDK/include/ucrt", "SDK/include/um", "SDK/include/shared"])
    env["LIB"] = ";".join(str(sdk / p) for p in ["VC/lib", "SDK/lib"])
    env["PATH"] = str(sdk / "VC" / "bin") + os.pathsep + env.get("PATH", "")
    cmd = [str(sdk / "VC" / "bin" / "cl.exe"), "/nologo", "/O2", "/fp:precise", "/MT", "/W3", "/LD", str(FIRMWARE / "frontend.c"),
           f"/Fe{DLL}", f"/Fo{OUT_DIR}{os.sep}", "/link", "/NOLOGO"] + [f"/EXPORT:{e}" for e in EXPORTS]
    r = subprocess.run(cmd, cwd=OUT_DIR, env=env, capture_output=True, text=True)
    if r.returncode != 0 or not DLL.is_file():
        raise SystemExit(f"cl failed ({r.returncode}):\n{r.stdout}\n{r.stderr}")
    return DLL


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vs", default=None)
    args = ap.parse_args()
    print("built", build(args.vs))
