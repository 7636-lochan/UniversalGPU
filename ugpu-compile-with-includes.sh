#!/bin/bash
# UniversalGPU wrapper: resolves #include directives then compiles

if [ $# -eq 0 ]; then
    echo "Usage: $0 file1.cu file2.cu ... -I<include_dir>" >&2
    exit 1
fi

SOURCE_FILES=()
INCLUDE_DIRS=()

for arg in "$@"; do
    if [[ "$arg" == -I* ]]; then
        INCLUDE_DIRS+=("${arg#-I}")
    else
        SOURCE_FILES+=("$arg")
    fi
done

INCLUDE_FLAGS=""
for dir in "${INCLUDE_DIRS[@]}"; do
    INCLUDE_FLAGS="$INCLUDE_FLAGS -I$dir"
done

PREPROCESSED=""
TEMP_FILES=()
for f in "${SOURCE_FILES[@]}"; do
    TEMP=$(mktemp /tmp/ugpu_pp_XXXXXX.cu)
    clang-22 -E -x cuda -nocudainc -nocudalib $INCLUDE_FLAGS "$f" > "$TEMP" 2>/dev/null
    PREPROCESSED="$PREPROCESSED $TEMP"
    TEMP_FILES+=("$TEMP")
done

~/ugpu/frontend/build/cuda-launch-parser $PREPROCESSED --

for f in "${TEMP_FILES[@]}"; do
    rm -f "$f"
done
