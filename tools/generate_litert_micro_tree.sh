#!/bin/bash
# Generate the LiteRT for Microcontrollers source tree on Windows (Git Bash) from a pinned tensorflow/tflite-micro commit.
# Prerequisites: Git for Windows, Python 3, GNU make ("winget install --id ezwinports.make --exact"; open a new shell afterwards),
# and the Arduino core arduino:mbed_nano 4.6.0.
#
#   tools/generate_litert_micro_tree.sh [WORK_DIR]        default WORK_DIR: /c/lb
#
# Output: WORK_DIR/tree, the input of tools/build_litert_micro_arduino.py. WORK_DIR must have no spaces in its real path: make
# resolves paths to their long form, so a short 8.3 alias of a path with spaces does not help.
#
# The upstream scripts expect a Linux host. Three adaptations stay inside WORK_DIR:
#   * core.autocrlf=false in the clone, because shell scripts with CRLF do not run;
#   * small python3, wget and mv wrappers in WORK_DIR/bin: Windows has python, not python3; wget is absent; a freshly extracted
#     directory can stay locked for a moment, which breaks mv;
#   * TARGET_TOOLCHAIN_ROOT points to the compiler of the Arduino core, otherwise the Makefile downloads another one.
set -euo pipefail

COMMIT=85c6999a1f0032f64f3b0002431a357b451275c4
W="${1:-/c/lb}"
W_WIN="$(cd "$(dirname "$W")" && pwd -W)/$(basename "$W")"
case "$W_WIN" in *" "*) echo "WORK_DIR must have no spaces: $W_WIN" >&2; exit 1 ;; esac
mkdir -p "$W/bin" "$W/tmp"

if [ ! -d "$W/tflite-micro" ]; then
  git -c core.autocrlf=false clone --filter=blob:none https://github.com/tensorflow/tflite-micro.git "$W/tflite-micro"
fi
git -C "$W/tflite-micro" config core.autocrlf false
git -C "$W/tflite-micro" fetch -q origin "$COMMIT" 2>/dev/null || true
git -C "$W/tflite-micro" checkout -q "$COMMIT"

printf '#!/bin/sh\nexec python "$@"\n' > "$W/bin/python3"
cat > "$W/bin/wget" <<'EOF'
#!/bin/sh
# wget URL -O FILE
exec curl -sSL --retry 3 -o "$3" "$1"
EOF
cat > "$W/bin/mv" <<'EOF'
#!/bin/sh
i=0
while [ $i -lt 8 ]; do
  /usr/bin/mv "$@" 2>/dev/null && exit 0
  sleep 2; i=$((i+1))
done
for last; do :; done
n=$#; a=1
for s in "$@"; do
  if [ $a -lt $n ]; then /usr/bin/cp -a "$s" "$last" || exit 1; /usr/bin/rm -rf "$s" 2>/dev/null; fi
  a=$((a+1))
done
EOF
chmod +x "$W/bin/python3" "$W/bin/wget" "$W/bin/mv"

MAKE_DIR="$(ls -d "$LOCALAPPDATA"/Microsoft/WinGet/Packages/ezwinports.make_*/bin 2>/dev/null | head -n 1 || true)"
[ -n "$MAKE_DIR" ] && PATH="$(cygpath -u "$MAKE_DIR"):$PATH"
export PATH="$W/bin:$PATH"
export TEMP="$W_WIN/tmp" TMP="$W_WIN/tmp" TMPDIR="$W/tmp"
command -v make >/dev/null || { echo "make not found" >&2; exit 1; }

# 8.3 form: no spaces, and this path only names the compiler, nothing is run from it while the tree is generated.
GCC_BIN="$(cygpath -d "$LOCALAPPDATA" | tr '\\' '/')/Arduino15/packages/arduino/tools/arm-none-eabi-gcc/7-2017q4/bin/"

rm -rf "$W/tree"
cd "$W/tflite-micro"
python3 tensorflow/lite/micro/tools/project_generation/create_tflm_tree.py \
  --makefile_options="TARGET=cortex_m_generic TARGET_ARCH=cortex-m4+sfp OPTIMIZED_KERNEL_DIR=cmsis_nn TARGET_TOOLCHAIN_ROOT=$GCC_BIN" \
  "$W_WIN/tree"
echo "tree: $W_WIN/tree ($(find "$W/tree" -type f | wc -l) files), commit $COMMIT"
