#!/bin/bash
#SBATCH --job-name=microduck_walk_v3
#SBATCH --account=aip-mtaylor3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3:30:00
#SBATCH --output=/scratch/kghasemz/isaaclab/microduck/isaac/logs/slurm_%x_%j.out
# Flat-ground walker v3 (rebuilt rewards, command curriculum, mirror augmentation) + corrected fall test.
#   cd /scratch/kghasemz/isaaclab/microduck/isaac && sbatch jobs/train_walk_v3.sh
set -euo pipefail
cd /scratch/kghasemz/isaaclab/microduck/isaac
R=../../v6/run.sh; IL=../../v6/venv/bin/isaaclab; unset DISPLAY
ITERS=${ITERS:-6000}
OLD=logs/rsl_rl/microduck_walk_v2/2026-11-09_18-02-55/model_3999.pt
stamp() { echo "[$(date -u +%H:%M:%S)] $*"; }
latest_run() { ls -td logs/rsl_rl/microduck_walk_v3/*/ | head -1; }
latest_ckpt() { echo "$1$(ls "$1" | grep -E '^model_[0-9]+\.pt$' | sort -V | tail -1)"; }
evalw() { $R scripts/eval_walk.py --task "$1" --checkpoint "$2" --num_envs 256 --seconds 60 2>&1 | grep -E "\[walk-eval\]|Traceback|Error:" | grep -v physicsschema || true; }
video() {  # $1 task $2 ckpt $3 extra viewer args...
  local t=$1 c=$2; shift 2
  $R $IL play --rl_library rsl_rl --task "$t" --checkpoint "$c" --viz kit --video --video_length 750 "$@" > logs/v3_play.log 2>&1 || true
}

stamp "smoke test (v3 + mirroring)"
$R $IL train --rl_library rsl_rl --task Microduck-Walk-V3 --num_envs 64 --max_iterations 3 > logs/v3_smoke.log 2>&1
grep -q "Learning iteration 2/3" logs/v3_smoke.log || { stamp "smoke FAILED"; grep -E "Error|Traceback" -A3 logs/v3_smoke.log | head -40; exit 1; }
SR=$(latest_run); rm -rf "$SR"

stamp "corrected fall test of the previous flat walker (stage A, model_3999)"
evalw Microduck-Walk-V2-Flat-Play "$OLD"

stamp "train v3: $ITERS iterations, 4096 robots"
$R $IL train --rl_library rsl_rl --task Microduck-Walk-V3 --num_envs 4096 --max_iterations "$ITERS" > logs/v3_train.log 2>&1
RUN=$(latest_run); CK=$(latest_ckpt "$RUN"); stamp "v3 done: $CK"
MID="${RUN}model_3000.pt"; [ -f "$MID" ] && { echo "v3 at iteration 3000:"; evalw Microduck-Walk-V3-Play "$MID"; }
echo "v3 final:"; evalw Microduck-Walk-V3-Play "$CK"
$R scripts/gait_check.py --task Microduck-Walk-V3-Play --checkpoint "$CK" --num_envs 49 2>&1 | grep "\[gait\]" || true
stamp "videos"
video Microduck-Walk-V3-Play "$CK" env.viewer.origin_type=asset_root env.viewer.asset_name=robot env.viewer.env_index=10 \
  env.viewer.eye="[0.7,-0.7,0.35]" env.viewer.lookat="[0.0,0.0,0.08]"
mkdir -p "$RUN/videos/closeup" && mv "$RUN"/videos/play/*.mp4 "$RUN/videos/closeup/" || true
video Microduck-Walk-V3-Play "$CK"
stamp "all done: $RUN"
