#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROS_SETUP="${ROS_SETUP:-/opt/ros/jazzy/setup.bash}"
if [[ ! -f "$ROS_SETUP" ]]; then
  printf 'ROS setup not found: %s\nSet ROS_SETUP to your ROS installation setup.bash.\n' "$ROS_SETUP" >&2
  exit 1
fi
source "$ROS_SETUP"
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}"
export QT_LINUX_ACCESSIBILITY_ALWAYS_ON=1
exec /usr/bin/python3 "$SCRIPT_DIR/rgbd_workbench.py" "$@"
