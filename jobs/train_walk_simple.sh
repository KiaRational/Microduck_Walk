#!/bin/bash
#SBATCH --job-name=microduck_walk_simple
#SBATCH --account=aip-mtaylor3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:30:00
#SBATCH --output=/scratch/kghasemz/isaaclab/microduck/isaac/logs/slurm_%x_%j.out
# Simplified walker: stage A flat ground -> stage B gentle props with a terrain curriculum (from stage A).
#   cd /scratch/kghasemz/isaaclab/microduck/isaac && sbatch jobs/train_walk_simple.sh
set -euo pipefail
cd /scratch/kghasemz/isaaclab/microduck/isaac
R=../../v6/run.sh; IL=../../v6/venv/bin/isaaclab; unset DISPLAY
A_ITERS=${A_ITERS:-4000}; B_ITERS=${B_ITERS:-3000}
stamp() { echo "[$(date -u +%H:%M:%S)] $*"; }
latest_run() { ls -td logs/rsl_rl/microduck_walk_v2/*/ | head -1; }
latest_ckpt() { echo "$1$(ls "$1" | grep -E '^model_[0-9]+\.pt$' | sort -V | tail -1)"; }
closeup() {  # $1 task, $2 ckpt
  $R $IL play --rl_library rsl_rl --task "$1" --checkpoint "$2" --viz kit --video --video_length 500 \
    env.viewer.origin_type=asset_root env.viewer.asset_name=robot env.viewer.env_index=10 \
    env.viewer.eye="[0.7,-0.7,0.35]" env.viewer.lookat="[0.0,0.0,0.08]" > logs/simple_play.log 2>&1 || true
}
check() {  # $1 play task, $2 ckpt
  $R scripts/eval_walk.py --task "$1" --checkpoint "$2" --num_envs 256 2>&1 | grep "\[walk-eval\]" || true
  $R scripts/gait_check.py --task "$1" --checkpoint "$2" --num_envs 49 2>&1 | grep "\[gait\]" || true
}

stamp "smoke tests"
$R $IL train --rl_library rsl_rl --task Microduck-Walk-V2-Flat --num_envs 64 --max_iterations 2 > logs/simple_smoke.log 2>&1
grep -q "Learning iteration 1/2" logs/simple_smoke.log || { stamp "flat smoke FAILED"; tail -40 logs/simple_smoke.log; exit 1; }
SR=$(latest_run); SCK=$(latest_ckpt "$SR")
$R $IL train --rl_library rsl_rl --task Microduck-Walk-V2-Gentle --num_envs 64 --max_iterations 2 --checkpoint "$SCK" > logs/simple_smoke2.log 2>&1
grep -qE "Learning iteration [0-9]+/3" logs/simple_smoke2.log || { stamp "gentle smoke FAILED"; tail -40 logs/simple_smoke2.log; exit 1; }
grep -iE "loading|resum" logs/simple_smoke2.log | head -3
rm -rf "$SR" "$(latest_run)"

stamp "stage A: flat ground, $A_ITERS iterations, 4096 robots"
$R $IL train --rl_library rsl_rl --task Microduck-Walk-V2-Flat --num_envs 4096 --max_iterations "$A_ITERS" > logs/simple_flat.log 2>&1
ARUN=$(latest_run); ACK=$(latest_ckpt "$ARUN"); stamp "stage A done: $ACK"
check Microduck-Walk-V2-Flat-Play "$ACK"
closeup Microduck-Walk-V2-Flat-Play "$ACK"

stamp "stage B: gentle props + curriculum, $B_ITERS iterations, from stage A"
$R $IL train --rl_library rsl_rl --task Microduck-Walk-V2-Gentle --num_envs 4096 --max_iterations "$B_ITERS" --checkpoint "$ACK" > logs/simple_gentle.log 2>&1
BRUN=$(latest_run); BCK=$(latest_ckpt "$BRUN"); stamp "stage B done: $BCK"
check Microduck-Walk-V2-Gentle-Play "$BCK"
echo "flat ground with the stage-B walker:"; $R scripts/eval_walk.py --task Microduck-Walk-V2-Flat-Play --checkpoint "$BCK" --num_envs 256 2>&1 | grep "\[walk-eval\]" || true
closeup Microduck-Walk-V2-Gentle-Play "$BCK"
stamp "all done (videos in $ARUN/videos/play and $BRUN/videos/play)"
