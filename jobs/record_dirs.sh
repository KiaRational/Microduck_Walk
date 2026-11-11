#!/bin/bash
#SBATCH --job-name=microduck_dirs_video
#SBATCH --account=aip-mtaylor3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=/scratch/kghasemz/isaaclab/microduck/isaac/logs/slurm_%x_%j.out
# Close-up videos of one robot under fixed commands (forward, backward, left, right, turn left, turn right).
#   sbatch jobs/record_dirs.sh <play task> <checkpoint>      (checkpoint "latest_v4" = newest v4 run)
set -uo pipefail
cd /scratch/kghasemz/isaaclab/microduck/isaac
R=../../v6/run.sh; IL=../../v6/venv/bin/isaaclab; unset DISPLAY
TASK=$1; CK=$2
if [[ "$CK" == latest_v* ]]; then RUN=$(ls -td logs/rsl_rl/microduck_walk_${CK#latest_}/*/ | head -1); CK="$RUN$(ls $RUN | grep -E '^model_[0-9]+\.pt$' | sort -V | tail -1)"; fi
OUT=$(dirname "$CK")/videos/directions; mkdir -p "$OUT"
for spec in "forward 0.3 0 0" "backward -0.3 0 0" "left 0 0.2 0" "right 0 -0.2 0" "turn_left 0 0 0.8" "turn_right 0 0 -0.8"; do
  set -- $spec
  touch "$OUT/.marker"
  $R $IL play --rl_library rsl_rl --task "$TASK" --checkpoint "$CK" --viz kit --video --video_length 400 \
    env.commands.base_velocity.ranges.lin_vel_x="[$2,$2]" env.commands.base_velocity.ranges.lin_vel_y="[$3,$3]" \
    env.commands.base_velocity.ranges.ang_vel_z="[$4,$4]" env.commands.base_velocity.rel_standing_envs=0.0 \
    env.commands.base_velocity.resampling_time_range="[1000.0,1000.0]" \
    env.viewer.origin_type=asset_root env.viewer.asset_name=robot env.viewer.env_index=10 \
    env.viewer.eye="[0.9,-0.9,0.45]" env.viewer.lookat="[0.0,0.0,0.08]" > logs/dirs_video.log 2>&1
  V=$(find "$(dirname "$CK")/videos/play" -name '*.mp4' -newer "$OUT/.marker" 2>/dev/null | head -1)
  [ -n "$V" ] && mv "$V" "$OUT/$1.mp4" && echo "saved $OUT/$1.mp4" || { echo "FAILED $1"; grep -E "Error|Traceback" -A3 logs/dirs_video.log | head; }
done
