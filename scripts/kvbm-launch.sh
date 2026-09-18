#!/usr/bin/env bash
# Launch vLLM (+ optional KVBM tiers) in the official aarch64 image, writing the serve command to a mounted script so the
# --kv-transfer-config JSON survives quoting.
#
#   kvbm-launch.sh <container> <model_path_in_container> <served_name> <gpu_mem_util> <max_model_len> [kvbm 1|0] [extra vllm args...]
#
# Env knobs (all optional):
#   KVBM_CPU_GB   host tier GB (default 8; must be >= the engine's GPU KV pool)      KVBM_DISK_GB  disk tier GB (default 64; 0 = no disk tier)
#   KVBM_POSIX / KVBM_GDS / KVBM_GDS_MT   DYN_KVBM_NIXL_BACKEND_* toggles (default unset/true/true)
#   KVBM_NO_ODIRECT=true                  DYN_KVBM_DISK_DISABLE_O_DIRECT       KVBM_LOG=debug   DYN_LOG
#   KVBM_EXTRA_DOCKER_ARGS                extra `docker run` args (e.g. -e CUFILE_ENV_PATH_JSON=/kvlogs/cufile.json)
#   MODELS_DIR (default $HOME/models -> /models)   IMAGE (default vllm/vllm-openai:v0.27.1-aarch64-ubuntu2404)
set -u
[ $# -ge 5 ] || { echo "usage: $0 <container> <model_path> <served_name> <gpu_mem_util> <max_model_len> [kvbm 1|0] [extra vllm args...]" >&2; exit 2; }
NAME=$1; MODEL=$2; SERVED=$3; GMU=$4; MAXLEN=$5; shift 5
KVBM=1; if [ $# -gt 0 ] && [ "$1" = 0 -o "$1" = 1 ]; then KVBM=$1; shift; fi
MODELS_DIR=${MODELS_DIR:-$HOME/models}; IMAGE=${IMAGE:-vllm/vllm-openai:v0.27.1-aarch64-ubuntu2404}
DISK_DIR=$HOME/kvbm-disk; LOG_DIR=$HOME/kvbm-logs; mkdir -p "$DISK_DIR" "$LOG_DIR"
docker rm -f "$NAME" >/dev/null 2>&1

KVARGS=""
[ "$KVBM" = 1 ] && KVARGS="--kv-transfer-config '{\"kv_connector\":\"DynamoConnector\",\"kv_role\":\"kv_both\",\"kv_connector_module_path\":\"kvbm.vllm_integration.connector\"}'"
EXTRA=""; for x in "$@"; do EXTRA="$EXTRA $(printf '%q' "$x")"; done
{
  echo '#!/usr/bin/env bash'
  echo 'set -e'
  [ "$KVBM" = 1 ] && echo 'pip install -q --no-deps kvbm==1.4.2 2>&1 | grep -v WARNING || true'   # --no-deps: keep the image NIXL
  [ "$KVBM" = 1 ] && echo 'python3 -c "import kvbm.vllm_integration.connector"'                     # fail loudly if the wheel is unusable
  printf 'exec vllm serve %q --served-model-name %q --port 8000 --max-model-len %q --gpu-memory-utilization %q --max-num-seqs 4 --max-num-batched-tokens 8192 --enable-prefix-caching %s%s\n' "$MODEL" "$SERVED" "$MAXLEN" "$GMU" "$KVARGS" "$EXTRA"
} > "$LOG_DIR/serve-$NAME.sh"; chmod +x "$LOG_DIR/serve-$NAME.sh"

DISK_ENV=""; [ "${KVBM_DISK_GB:-64}" != 0 ] && DISK_ENV="-e DYN_KVBM_DISK_CACHE_GB=${KVBM_DISK_GB:-64} -e DYN_KVBM_DISK_CACHE_DIR=/kvbm-disk"
docker run -d --name "$NAME" --gpus all --ipc=host --network host --memory 100g \
  -v "$MODELS_DIR":/models -v "$DISK_DIR":/kvbm-disk -v "$LOG_DIR":/kvlogs \
  -e DYN_KVBM_CPU_CACHE_GB=${KVBM_CPU_GB:-8} $DISK_ENV \
  -e DYN_KVBM_DISABLE_DISK_OFFLOAD_FILTER=true -e DYN_KVBM_METRICS=true -e DYN_KVBM_METRICS_PORT=6880 \
  -e DYN_KVBM_CACHE_STATS_LOG_INTERVAL_SECS=60 \
  -e DYN_KVBM_NIXL_BACKEND_POSIX=${KVBM_POSIX:-false} -e DYN_KVBM_NIXL_BACKEND_GDS_MT=${KVBM_GDS_MT:-true} -e DYN_KVBM_NIXL_BACKEND_GDS=${KVBM_GDS:-true} \
  -e DYN_KVBM_DISK_DISABLE_O_DIRECT=${KVBM_NO_ODIRECT:-false} -e DYN_LOG=${KVBM_LOG:-info} ${KVBM_EXTRA_DOCKER_ARGS:-} \
  --entrypoint bash "$IMAGE" "/kvlogs/serve-$NAME.sh" >/dev/null || { echo "docker run failed" >&2; exit 1; }
sleep 5
docker ps --format '{{.Names}}' | grep -qx "$NAME" || { echo "container exited early:" >&2; docker logs --tail 20 "$NAME" >&2; exit 1; }
echo "started $NAME ($(date +%T)); wait for 'Application startup complete' in: docker logs -f $NAME"
