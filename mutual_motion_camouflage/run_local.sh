#!/usr/bin/env bash
set -eo pipefail
robot="${1:-}"
case "$robot" in
  burger1|burger3) shift ;;
  *) echo 'Usage: bash run_local.sh burger1|burger3 [--ros-args ...]' >&2; exit 2 ;;
esac
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source /opt/ros/humble/setup.bash
source "$HOME/turtlebot3_ws/install/setup.bash"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-30}"
export ROS_LOCALHOST_ONLY=0
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
exec python3 "$script_dir/mmc_${robot}.py" "$@"
