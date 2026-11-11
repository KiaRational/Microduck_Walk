#!/bin/bash
#SBATCH --job-name=microduck_unilab
#SBATCH --account=aip-mtaylor3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/scratch/kghasemz/isaaclab/microduck/isaac/logs/slurm_%x_%j.out
# Reference pipeline (rocPAI-Forge/microduck_rl_unilab, MuJoCo + BAM servos) on Vulcan:
# install (CUDA profile) -> record the shipped walker -> train microduck_velocity_flat (2048 envs, 500 iters) -> videos.
set -uo pipefail
ROOT=/scratch/kghasemz/isaaclab/microduck; REPO=$ROOT/microduck_rl_unilab; OUT=$ROOT/unilab_runs
mkdir -p $OUT
stamp() { echo "[$(date -u +%H:%M:%S)] $*"; }
module --force purge; module load StdEnv/2023 python/3.12.4 cmake/3.31.0
export UV_CACHE_DIR=/scratch/kghasemz/isaaclab/cache/uv UV_PYTHON_PREFERENCE=only-system UV_PYTHON=$(which python3)
export XDG_CACHE_HOME=/scratch/kghasemz/isaaclab/cache/xdg MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
stamp "install uv + repo (CUDA profile)"
if [ ! -x $ROOT/uvtool/bin/uv ]; then
  python3 -m venv $ROOT/uvtool && $ROOT/uvtool/bin/pip install -q --no-index uv 2>/dev/null || $ROOT/uvtool/bin/pip install -q --index-url https://pypi.org/simple uv
fi
export PATH=$ROOT/uvtool/bin:$PATH; uv --version
cd $REPO && make sync-cuda > $OUT/install.log 2>&1 || { stamp "install FAILED"; tail -40 $OUT/install.log; exit 1; }
uv run python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

play() {  # $1 run dir, $2 checkpoint iteration, $3 name, $4 vx $5 vy $6 wz
  touch $OUT/.marker
  uv run microduck-eval --algo ppo --task microduck_velocity_flat --sim mujoco --load-run "$1" algo.checkpoint=$2 \
    training.play_render_mode=record training.play_env_num=1 training.play_steps=500 \
    training.cam_distance=1.0 training.cam_elevation=-12 training.cam_azimuth=130 \
    training.cam_tracking=true training.cam_tracking_extra_envs=0 training.device=cuda:0 \
    env.commands.twist.rel_standing_envs=0.0 env.commands.twist.rel_forward_envs=1.0 \
    "env.commands.twist.ranges.lin_vel_x=[$4,$4]" "env.commands.twist.ranges.lin_vel_y=[$5,$5]" \
    "env.commands.twist.ranges.ang_vel_z=[$6,$6]" env.curriculum.standing_envs=null > $OUT/play_$3.log 2>&1
  V=$(find $REPO $1 -name '*.mp4' -newer $OUT/.marker 2>/dev/null | head -1)
  [ -n "$V" ] && { mkdir -p $7; mv "$V" "$7/$3.mp4"; echo "saved $7/$3.mp4"; } || { echo "no video for $3"; tail -5 $OUT/play_$3.log; }
}

stamp "record the shipped reference walker (model_499)"
REF=$REPO/examples/velocity_flat_mujoco
play $REF 499 forward 0.4 0.0 0.0 $OUT/reference_videos

stamp "smoke test"
uv run microduck-train --algo ppo --task microduck_velocity_flat --sim mujoco \
  algo.num_envs=4 algo.max_iterations=2 training.no_play=true training.play_env_num=4 > $OUT/smoke.log 2>&1 \
  || { stamp "smoke FAILED"; tail -40 $OUT/smoke.log; exit 1; }

stamp "train microduck_velocity_flat: 2048 envs, 500 iterations (reference budget)"
touch $OUT/.train_marker
uv run microduck-train --algo ppo --task microduck_velocity_flat --sim mujoco training.no_play=true > $OUT/train.log 2>&1
RUN=$(dirname "$(find $REPO -name 'model_499.pt' -newer $OUT/.train_marker | head -1)")
stamp "trained: $RUN"; grep -iE "mean reward|episode length" $OUT/train.log | tail -2
for spec in "forward 0.3 0.0 0.0" "backward -0.3 0.0 0.0" "left 0.0 0.2 0.0" "turn_left 0.0 0.0 0.8" "turn_right 0.0 0.0 -0.8"; do
  set -- $spec; play "$RUN" 499 $1 $2 $3 $4 $OUT/our_run_videos
done
stamp "all done"
