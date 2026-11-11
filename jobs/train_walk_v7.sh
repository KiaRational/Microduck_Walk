#!/bin/bash
#SBATCH --job-name=microduck_walk_v7
#SBATCH --account=aip-mtaylor3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:30:00
#SBATCH --output=/scratch/kghasemz/isaaclab/microduck/isaac/logs/slurm_%x_%j.out
# Walker v7: XL330 servo model (BAM) + reference reward recipe, flat ground.
#   cd /scratch/kghasemz/isaaclab/microduck/isaac && sbatch jobs/train_walk_v7.sh
set -euo pipefail
cd /scratch/kghasemz/isaaclab/microduck/isaac
R=../../v6/run.sh; IL=../../v6/venv/bin/isaaclab; unset DISPLAY
ITERS=${ITERS:-4000}
stamp() { echo "[$(date -u +%H:%M:%S)] $*"; }
latest_run() { ls -td logs/rsl_rl/microduck_walk_v7/*/ | head -1; }
latest_ckpt() { echo "$1$(ls "$1" | grep -E '^model_[0-9]+\.pt$' | sort -V | tail -1)"; }
dirs() { $R scripts/eval_dirs.py --task "$1" --checkpoint "$2" 2>&1 | grep -E "\[dirs\]|Traceback|Error:" | grep -v physicsschema || true; }

stamp "smoke test (v7 + servo model)"
$R $IL train --rl_library rsl_rl --task Microduck-Walk-V7 --num_envs 64 --max_iterations 3 env.scene.terrain.terrain_generator.num_rows=4 env.scene.terrain.terrain_generator.num_cols=16 > logs/v7_smoke.log 2>&1
grep -q "Learning iteration 2/3" logs/v7_smoke.log || { stamp "smoke FAILED"; grep -E "Error|Traceback" -A4 logs/v7_smoke.log | head -40; exit 1; }
SR=$(latest_run); dirs Microduck-Walk-V7-Play "$(latest_ckpt "$SR")" | tail -1; rm -rf "$SR"


stamp "train v7: $ITERS iterations, 4096 robots"
$R $IL train --rl_library rsl_rl --task Microduck-Walk-V7 --num_envs 4096 --max_iterations "$ITERS" > logs/v7_train.log 2>&1
RUN=$(latest_run); CK=$(latest_ckpt "$RUN"); stamp "v7 done: $CK"
for it in 1000 2000 3000; do [ -f "${RUN}model_$it.pt" ] && { echo "v7 at iteration $it:"; dirs Microduck-Walk-V7-Play "${RUN}model_$it.pt" | tail -1; }; done
echo "v7 final:"; dirs Microduck-Walk-V7-Play "$CK"
$R scripts/eval_walk.py --task Microduck-Walk-V7-Play --checkpoint "$CK" --num_envs 256 --seconds 60 2>&1 | grep "\[walk-eval\]" || true
$R scripts/gait_check.py --task Microduck-Walk-V7-Play --checkpoint "$CK" --num_envs 49 2>&1 | grep "\[gait\]" || true
echo "terrain level reached in training (0-9):"; sed "s/\x1b\[[0-9;]*m//g" logs/v7_train.log | grep "Curriculum/terrain_levels" | awk 'NR%500==1 || 0 {print}' | tail -9
stamp "videos"
$R $IL play --rl_library rsl_rl --task Microduck-Walk-V7-Play --checkpoint "$CK" --viz kit --video --video_length 750 \
  env.viewer.origin_type=asset_root env.viewer.asset_name=robot env.viewer.env_index=10 \
  env.viewer.eye="[0.7,-0.7,0.35]" env.viewer.lookat="[0.0,0.0,0.08]" > logs/v7_play.log 2>&1 || true
mkdir -p "$RUN/videos/closeup" && mv "$RUN"/videos/play/*.mp4 "$RUN/videos/closeup/" || true
$R $IL play --rl_library rsl_rl --task Microduck-Walk-V7-Play --checkpoint "$CK" --viz kit --video --video_length 750 > logs/v7_play.log 2>&1 || true
stamp "all done: $RUN"
