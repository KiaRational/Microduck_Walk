#!/bin/bash
#SBATCH --job-name=microduck_walk_v6
#SBATCH --account=aip-mtaylor3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --output=/scratch/kghasemz/isaaclab/microduck/isaac/logs/slurm_%x_%j.out
# Walker v6: XL330 servo model (BAM) + reference reward recipe, flat ground.
#   cd /scratch/kghasemz/isaaclab/microduck/isaac && sbatch jobs/train_walk_v6.sh
set -euo pipefail
cd /scratch/kghasemz/isaaclab/microduck/isaac
R=../../v6/run.sh; IL=../../v6/venv/bin/isaaclab; unset DISPLAY
ITERS=${ITERS:-3000}
stamp() { echo "[$(date -u +%H:%M:%S)] $*"; }
latest_run() { ls -td logs/rsl_rl/microduck_walk_v6/*/ | head -1; }
latest_ckpt() { echo "$1$(ls "$1" | grep -E '^model_[0-9]+\.pt$' | sort -V | tail -1)"; }
dirs() { $R scripts/eval_dirs.py --task "$1" --checkpoint "$2" 2>&1 | grep -E "\[dirs\]|Traceback|Error:" | grep -v physicsschema || true; }

stamp "smoke test (v6 + servo model)"
$R $IL train --rl_library rsl_rl --task Microduck-Walk-V6 --num_envs 64 --max_iterations 3 > logs/v6_smoke.log 2>&1
grep -q "Learning iteration 2/3" logs/v6_smoke.log || { stamp "smoke FAILED"; grep -E "Error|Traceback" -A4 logs/v6_smoke.log | head -40; exit 1; }
SR=$(latest_run); dirs Microduck-Walk-V6-Play "$(latest_ckpt "$SR")" | tail -1; rm -rf "$SR"


stamp "previous v5 walker (trained on ovphysx) tested on Isaac Sim PhysX, i.e. what the GUI shows"
V5=logs/rsl_rl/microduck_walk_v5/2026-11-10_07-03-54/model_2999.pt
dirs Microduck-Walk-V5-Play "$V5"
$R scripts/eval_walk.py --task Microduck-Walk-V5-Play --checkpoint "$V5" --num_envs 256 --seconds 60 2>&1 | grep "\[walk-eval\]" || true
echo "same walker on ovphysx (its training engine):"
MICRODUCK_PHYSICS=ovphysx $R scripts/eval_walk.py --task Microduck-Walk-V5-Play --checkpoint "$V5" --num_envs 256 --seconds 60 2>&1 | grep "\[walk-eval\]" || true

stamp "train v6: $ITERS iterations, 4096 robots"
$R $IL train --rl_library rsl_rl --task Microduck-Walk-V6 --num_envs 4096 --max_iterations "$ITERS" > logs/v6_train.log 2>&1
RUN=$(latest_run); CK=$(latest_ckpt "$RUN"); stamp "v6 done: $CK"
for it in 1000 2000; do [ -f "${RUN}model_$it.pt" ] && { echo "v6 at iteration $it:"; dirs Microduck-Walk-V6-Play "${RUN}model_$it.pt" | tail -1; }; done
echo "v6 final:"; dirs Microduck-Walk-V6-Play "$CK"
$R scripts/eval_walk.py --task Microduck-Walk-V6-Play --checkpoint "$CK" --num_envs 256 --seconds 60 2>&1 | grep "\[walk-eval\]" || true
$R scripts/gait_check.py --task Microduck-Walk-V6-Play --checkpoint "$CK" --num_envs 49 2>&1 | grep "\[gait\]" || true
stamp "videos"
$R $IL play --rl_library rsl_rl --task Microduck-Walk-V6-Play --checkpoint "$CK" --viz kit --video --video_length 750 \
  env.viewer.origin_type=asset_root env.viewer.asset_name=robot env.viewer.env_index=10 \
  env.viewer.eye="[0.7,-0.7,0.35]" env.viewer.lookat="[0.0,0.0,0.08]" > logs/v6_play.log 2>&1 || true
mkdir -p "$RUN/videos/closeup" && mv "$RUN"/videos/play/*.mp4 "$RUN/videos/closeup/" || true
$R $IL play --rl_library rsl_rl --task Microduck-Walk-V6-Play --checkpoint "$CK" --viz kit --video --video_length 750 > logs/v6_play.log 2>&1 || true
stamp "all done: $RUN"
