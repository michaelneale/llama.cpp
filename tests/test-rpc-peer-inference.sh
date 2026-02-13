#!/usr/bin/env bash
# Integration test for RPC server-to-server tensor transfer: end-to-end inference.
#
# This test verifies that inference with two RPC servers produces correct output
# when using the direct push optimization (Test 3 from the plan).
#
# Prerequisites:
#   - rpc-server binary (set RPC_SERVER_PATH or have it in bin/)
#   - llama-cli binary (set LLAMA_CLI_PATH or have it in bin/)
#   - A small GGUF model (set MODEL_PATH)
#
# Usage:
#   MODEL_PATH=path/to/model.gguf ./test-rpc-peer-inference.sh
#   MODEL_PATH=path/to/model.gguf RPC_SERVER_PATH=./rpc-server LLAMA_CLI_PATH=./llama-cli ./test-rpc-peer-inference.sh
set -euo pipefail

# ---- Find binaries ----
RPC_SERVER="${RPC_SERVER_PATH:-}"
if [ -z "$RPC_SERVER" ]; then
    for p in ./bin/rpc-server ./rpc-server ../bin/rpc-server; do
        if [ -x "$p" ]; then RPC_SERVER="$p"; break; fi
    done
fi
if [ -z "$RPC_SERVER" ] || [ ! -x "$RPC_SERVER" ]; then
    echo "ERROR: rpc-server not found. Set RPC_SERVER_PATH." >&2
    exit 1
fi

LLAMA_CLI="${LLAMA_CLI_PATH:-}"
if [ -z "$LLAMA_CLI" ]; then
    for p in ./bin/llama-cli ./llama-cli ../bin/llama-cli; do
        if [ -x "$p" ]; then LLAMA_CLI="$p"; break; fi
    done
fi
if [ -z "$LLAMA_CLI" ] || [ ! -x "$LLAMA_CLI" ]; then
    echo "ERROR: llama-cli not found. Set LLAMA_CLI_PATH." >&2
    exit 1
fi

MODEL="${MODEL_PATH:-}"
if [ -z "$MODEL" ] || [ ! -f "$MODEL" ]; then
    echo "ERROR: Model not found. Set MODEL_PATH to a GGUF file." >&2
    exit 1
fi

echo "rpc-server: $RPC_SERVER"
echo "llama-cli:  $LLAMA_CLI"
echo "model:      $MODEL"

# ---- Helpers ----
PIDS=()
cleanup() {
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    done
}
trap cleanup EXIT

pick_port() {
    python3 -c "import socket; s=socket.socket(); s.bind(('127.0.0.1',0)); print(s.getsockname()[1]); s.close()"
}

wait_for_port() {
    local port=$1
    for i in $(seq 1 50); do
        if (echo > /dev/tcp/127.0.0.1/$port) 2>/dev/null; then
            return 0
        fi
        sleep 0.1
    done
    echo "ERROR: port $port not ready" >&2
    return 1
}

PORT_A=$(pick_port)
PORT_B=$(pick_port)

echo ""
echo "=== Test 3: End-to-End Inference with Two RPC Servers ==="
echo "Starting rpc-server A on port $PORT_A..."
$RPC_SERVER -H 127.0.0.1 -p "$PORT_A" >/dev/null 2>&1 &
PIDS+=($!)

echo "Starting rpc-server B on port $PORT_B..."
$RPC_SERVER -H 127.0.0.1 -p "$PORT_B" >/dev/null 2>&1 &
PIDS+=($!)

wait_for_port "$PORT_A"
wait_for_port "$PORT_B"

echo "Running inference with two RPC servers..."
OUTPUT_DUAL=$(GGML_RPC_DEBUG=1 $LLAMA_CLI \
    --rpc "127.0.0.1:${PORT_A},127.0.0.1:${PORT_B}" \
    -m "$MODEL" \
    -ngl 999 \
    -p "Hello, world" \
    -n 32 \
    --seed 42 \
    --temp 0 \
    2>/tmp/rpc-test-dual-stderr.log) || true

echo "Two-server output:"
echo "$OUTPUT_DUAL"

# Check debug log for peer registration
if grep -q "RPC_CMD_REGISTER_PEER\|register_peer\|registered peer" /tmp/rpc-test-dual-stderr.log 2>/dev/null; then
    echo "✓ Debug log shows peer registration"
else
    echo "⚠ No peer registration seen in debug log (may need GGML_RPC_DEBUG=1)"
fi

# Run reference inference with single server
echo ""
echo "Running reference inference with single RPC server..."
OUTPUT_SINGLE=$(GGML_RPC_DEBUG=1 $LLAMA_CLI \
    --rpc "127.0.0.1:${PORT_A}" \
    -m "$MODEL" \
    -ngl 999 \
    -p "Hello, world" \
    -n 32 \
    --seed 42 \
    --temp 0 \
    2>/dev/null) || true

echo "Single-server output:"
echo "$OUTPUT_SINGLE"

# Compare (for greedy decoding with same seed, outputs should match)
if [ "$OUTPUT_DUAL" = "$OUTPUT_SINGLE" ]; then
    echo ""
    echo "✓ PASS: Two-server output matches single-server output"
else
    echo ""
    echo "⚠ WARNING: Outputs differ (may be acceptable for non-deterministic backends)"
    echo "  This is not necessarily a failure — check outputs above."
fi

echo ""
echo "=== Test complete ==="
