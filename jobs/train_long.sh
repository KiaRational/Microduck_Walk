#!/bin/bash
#SBATCH --job-name=microduck_long
#SBATCH --account=aip-mtaylor3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=9:00:00
#SBATCH --output=/scratch/kghasemz/isaaclab/microduck/isaac/logs/slurm_%x_%j.out
# Long Microduck run: walker v2 (10k iters) -> gait check + close-up video -> navigation (3k iters) -> eval + video.
#   cd /scratch/kghasemz/isaaclab/microduck/isaac && sbatch jobs/train_long.sh
set -euo pipefail
cd /scratch/kghasemz/isaaclab/microduck/isaac
R=../../v6/run.sh; IL=../../v6/venv/bin/isaaclab
unset DISPLAY
WALK_ITERS=${WALK_ITERS:-10000}; NAV_ITERS=${NAV_ITERS:-3000}
stamp() { echo "[$(date -u +%H:%M:%S)] $*"; }
latest_run() { ls -td logs/rsl_rl/$1/*/ | head -1; }
latest_ckpt() { ls "$1" | grep -E '^model_[0-9]+\.pt$' | sort -V | tail -1; }

stamp "smoke test (64 robots, small terrain)"
$R $IL train --rl_library rsl_rl --task Microduck-Walk-V2 --num_envs 64 --max_iterations 2 \
  env.scene.terrain.terrain_generator.num_rows=4 env.scene.terrain.terrain_generator.num_cols=16 > logs/long_smoke.log 2>&1
grep -q "Learning iteration 1/2" logs/long_smoke.log || { stamp "smoke test FAILED"; tail -40 logs/long_smoke.log; exit 1; }
rm -rf "$(latest_run microduck_walk_v2)"

stamp "walker: $WALK_ITERS iterations, 4096 robots"
$R $IL train --rl_library rsl_rl --task Microduck-Walk-V2 --num_envs 4096 --max_iterations "$WALK_ITERS" > logs/long_walk.log 2>&1
WRUN=$(latest_run microduck_walk_v2); WCK=$(latest_ckpt "$WRUN"); stamp "walker done: $WRUN$WCK"

stamp "walker close-up video + export"
$R $IL play --rl_library rsl_rl --task Microduck-Walk-V2-Play --checkpoint "$WRUN$WCK" --viz kit --video --video_length 500 \
  env.viewer.origin_type=asset_root env.viewer.asset_name=robot env.viewer.env_index=20 \
  env.viewer.eye="[0.7,-0.7,0.35]" env.viewer.lookat="[0.0,0.0,0.08]" > logs/long_walk_play.log 2>&1 || true
[ -f assets/walk_v2_policy.pt ] && cp assets/walk_v2_policy.pt assets/walk_v2_policy_model1100.pt.bak
cp "$WRUN/exported/policy.pt" assets/walk_v2_policy.pt
$R scripts/gait_check.py --task Microduck-Walk-V2-Play --checkpoint "$WRUN$WCK" --num_envs 50 2>&1 | grep "\[gait\]" || true

stamp "navigation: $NAV_ITERS iterations, 2048 robots"
$R $IL train --rl_library rsl_rl --task Microduck-Navigation --num_envs 2048 --max_iterations "$NAV_ITERS" > logs/long_nav.log 2>&1
NRUN=$(latest_run microduck_nav); NCK=$(latest_ckpt "$NRUN"); stamp "navigation done: $NRUN$NCK"

stamp "navigation eval (512 fresh tiles) + video"
$R scripts/eval_nav.py --checkpoint "$NRUN$NCK" --num_envs 512 \
  env.scene.terrain.terrain_generator.num_rows=16 env.scene.terrain.terrain_generator.num_cols=32 2>&1 | grep "\[eval\]" || true
$R $IL play --rl_library rsl_rl --task Microduck-Navigation-Play --checkpoint "$NRUN$NCK" --viz kit --video --video_length 450 \
  env.viewer.eye="[6.5,-6.5,4.5]" > logs/long_nav_play.log 2>&1 || true
stamp "all done"
