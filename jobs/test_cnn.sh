#!/bin/bash
#SBATCH --job-name=microduck_cnn_test
#SBATCH --account=aip-mtaylor3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0:45:00
#SBATCH --output=/scratch/kghasemz/isaaclab/microduck/isaac/logs/slurm_%x_%j.out
set -uo pipefail
cd /scratch/kghasemz/isaaclab/microduck/isaac
R=../../v6/run.sh; IL=../../v6/venv/bin/isaaclab; unset DISPLAY
SMALL="env.scene.terrain.terrain_generator.num_rows=4 env.scene.terrain.terrain_generator.num_cols=8"
$R $IL train --rl_library rsl_rl --task Microduck-Navigation --num_envs 32 --max_iterations 3 $SMALL > logs/cnn_test_train.log 2>&1
grep -E "Learning iteration 2/3|Traceback|Error" logs/cnn_test_train.log | grep -v physicsschema | head -5
grep -iE "CNN|cnns|Conv2d" logs/cnn_test_train.log | head -12
RUN=$(ls -td logs/rsl_rl/microduck_nav/*/ | head -1); CK=$(ls $RUN | grep -E '^model_[0-9]+\.pt$' | sort -V | tail -1); echo "ckpt $RUN$CK"
$R scripts/eval_nav.py --checkpoint "$RUN$CK" --num_envs 32 $SMALL 2>&1 | grep -E "\[eval\]|Traceback|Error" | grep -v physicsschema | head -5
$R $IL play --rl_library rsl_rl --task Microduck-Navigation-Play --checkpoint "$RUN$CK" --viz kit --video --video_length 20 > logs/cnn_test_play.log 2>&1
grep -E "Traceback|Error:" logs/cnn_test_play.log | grep -v "physicsschema\|get_event_key" | head -5; ls -la $RUN/exported $RUN/videos/play 2>&1
rm -rf "$RUN"; echo "test done"
