#!/usr/bin/env bash
# Runs the experiment plan (doc/experiment_plan.md) end to end on CPU.
# Idempotent: finished data builds, trainings and evaluations are skipped.
# Usage: bash scripts/run_experiment_plan.sh [phase ...]   (default: all phases)
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=. MLFLOW_DISABLE_AGENT_HINT=1
PY=${PY:-.venv/Scripts/python.exe}
[ -x "$PY" ] || PY=python
OUT=outputs/plan
ANALYSIS=$OUT/analysis
mkdir -p "$OUT/logs" "$ANALYSIS"
log() { echo "$(date '+%F %T') $*" | tee -a "$OUT/plan.log"; }

ADM=data/genimage_adm/samples.jsonl
ADM128=data/genimage_adm_128/samples.jsonl
BIGGAN128=data/genimage_biggan_128/samples.jsonl
BIGGAN_TRAIN=data/genimage_biggan_train/samples.jsonl
ADM20K=data/genimage_adm_20k
JOINT=data/joint_adm_biggan/samples.jsonl
PERTURBED=data/perturbed_adm
PERTURBATIONS="border_gray:16 border_gray:32 border_reflect:32 jpeg:90 jpeg:75 jpeg:50 resample:0.75 blur:1 lowpass:0.125 highpass:0.125 shuffle:32 shuffle:64"
BASE_TESTS="--test adm256=$ADM:test_id --test adm128=$ADM128:test_id --test biggan128=$BIGGAN128:test_id"

perturbed_tests() {
    local args=""
    for op in $PERTURBATIONS; do
        local name=${op/:/}
        args="$args --test adm256_${name//./}=$PERTURBED/${name//./}/samples.jsonl:test_id"
    done
    echo "$args"
}

evaluate() {  # evaluate <name> <checkpoint> [extra evaluate args]
    local name=$1 checkpoint=$2
    shift 2
    if [ ! -f "$checkpoint" ]; then log "SKIP eval $name: no checkpoint"; return; fi
    log "eval $name"
    "$PY" -m scripts.experiments evaluate "$ANALYSIS" "$checkpoint" --name "$name" --val $BASE_TESTS "$@" \
        >> "$OUT/logs/eval-$name.log" 2>&1 || log "FAIL eval $name"
}

train() {  # train <name> <hydra overrides...>
    local name=$1
    shift
    local directory=$OUT/runs/$name
    if [ -f "$directory/best.pt" ] && [ -f "$directory/done" ]; then log "skip train $name (done)"; return; fi
    if [ -e "$directory" ]; then log "remove incomplete $directory"; rm -rf "$directory"; fi
    log "train $name: $*"
    if "$PY" -m scripts.train "$@" "hydra.run.dir=$directory" > "$OUT/logs/train-$name.log" 2>&1; then
        touch "$directory/done"
        log "done $name: $(tail -n 1 "$directory/metrics.jsonl" | cut -c1-120)"
    else
        log "FAIL train $name (see $OUT/logs/train-$name.log)"
    fi
}

MAIN="experiment=adm epochs=15 optimizer.schedule=cosine selection.patience=4"
SHORT="experiment=adm epochs=10 optimizer.schedule=cosine selection.patience=4"

phase_perturb() {
    for op in $PERTURBATIONS; do
        local name=${op/:/}
        name=${name//./}
        [ -f "$PERTURBED/$name/samples.jsonl" ] && continue
        log "perturb $op"
        "$PY" -m scripts.experiments perturb "$ADM" test_id "$PERTURBED/$name" "$op" >> "$OUT/logs/data.log" 2>&1 || log "FAIL perturb $op"
    done
}

phase_builds() {
    if [ ! -f "$ADM20K/samples.jsonl" ]; then
        log "build $ADM20K"
        "$PY" -m scripts.prepare_genimage C:/data/ADM/imagenet_ai_0508_adm.zip "$ADM20K" --train 20000 --val 1000 --test 1000 >> "$OUT/logs/data.log" 2>&1 || log "FAIL build adm20k"
    fi
    for n in 2000 4000 10000 20000; do
        [ -f "$ADM20K/samples_$n.jsonl" ] || "$PY" -m scripts.experiments subset "$ADM20K/samples.jsonl" $n "$ADM20K/samples_$n.jsonl" >> "$OUT/logs/data.log" 2>&1
    done
    if [ ! -f "$BIGGAN_TRAIN" ]; then
        log "build biggan train"
        "$PY" -m scripts.prepare_genimage C:/data/BigGAN/imagenet_ai_0419_biggan.zip data/genimage_biggan_train --train 4000 --val 1000 --test 0 --size 128 --allow-missing-volumes >> "$OUT/logs/data.log" 2>&1 || log "FAIL build biggan train"
    fi
    if [ ! -f "$JOINT" ]; then
        log "merge joint ADM+BigGAN"
        "$PY" -m scripts.experiments merge "$JOINT" "$ADM128:2000:500" "$BIGGAN_TRAIN:2000:500" >> "$OUT/logs/data.log" 2>&1 || log "FAIL merge"
    fi
}

phase_diagnostics() {  # experiments 1, 2, 11, 13, 14 on the existing checkpoints
    for model in baseline dual; do
        evaluate "old-adm256-$model" "outputs/adm-$model/best.pt" $(perturbed_tests)
        evaluate "old-adm128-$model" "outputs/adm128-$model/best.pt"
        [ -f "$ANALYSIS/occlusion/old-adm256-$model/occlusion.json" ] || \
            "$PY" -m scripts.experiments occlusion "outputs/adm-$model/best.pt" "$ADM" test_id "$ANALYSIS/occlusion/old-adm256-$model" >> "$OUT/logs/diagnostics.log" 2>&1 || log "FAIL occlusion $model"
    done
    [ -f "$ANALYSIS/spectra/spectra.json" ] || {
        log "spectra"
        "$PY" -m scripts.experiments spectra "$ANALYSIS/spectra" "adm256=$ADM:test_id" "adm128=$ADM128:test_id" "biggan128=$BIGGAN128:test_id" >> "$OUT/logs/diagnostics.log" 2>&1 || log "FAIL spectra"
    }
    "$PY" -m scripts.experiments summarise "$ANALYSIS" >> "$OUT/logs/diagnostics.log" 2>&1
}

main_runs() {  # experiments 3, 4, 5 for one seed
    local seed=$1
    for model in baseline local_only dual_resolution; do
        local name=e3-${model%%_*}-s$seed
        train "$name" $MAIN model=$model seed=$seed
        evaluate "$name" "$OUT/runs/$name/best.pt" $(perturbed_tests)
    done
}

phase_seed17() {
    main_runs 17
    for model in baseline local_only dual_resolution; do
        local name=e3-${model%%_*}-s17
        [ -f "$ANALYSIS/occlusion/$name/occlusion.json" ] || \
            "$PY" -m scripts.experiments occlusion "$OUT/runs/$name/best.pt" "$ADM" test_id "$ANALYSIS/occlusion/$name" >> "$OUT/logs/diagnostics.log" 2>&1 || log "FAIL occlusion $name"
    done
    "$PY" -m scripts.experiments summarise "$ANALYSIS" >> "$OUT/logs/diagnostics.log" 2>&1
}

phase_ablations() {
    # 15: augmentation
    for model in baseline dual_resolution; do
        local name=e15-${model%%_*}-aug
        train "$name" $MAIN model=$model augment.probability=0.5
        evaluate "$name" "$OUT/runs/$name/best.pt" $(perturbed_tests)
    done
    # 6: train on BigGAN, test on ADM
    for model in baseline dual_resolution; do
        local name=e6-${model%%_*}-biggan
        train "$name" $SHORT model=$model data.manifest=$BIGGAN_TRAIN
        evaluate "$name" "$OUT/runs/$name/best.pt"
    done
    # 21: train on ADM + BigGAN (both 128 native)
    for model in baseline dual_resolution; do
        local name=e21-${model%%_*}-joint
        train "$name" $SHORT model=$model data.manifest=$JOINT
        evaluate "$name" "$OUT/runs/$name/best.pt"
    done
    # 16: training-set size
    for n in 2000 4000 10000 20000; do
        local name=e16-baseline-n$n
        train "$name" $SHORT model=baseline data.manifest=$ADM20K/samples_$n.jsonl
        evaluate "$name" "$OUT/runs/$name/best.pt"
    done
    # 17: local crop size
    for size in 32 64; do
        local name=e17-dual-local$size
        train "$name" $SHORT model=dual_resolution views.local_size=$size
        evaluate "$name" "$OUT/runs/$name/best.pt"
    done
    # 18: capacity
    train e18-baseline-wide $SHORT model=baseline "model.encoder.channels=[32,64,128]"
    evaluate e18-baseline-wide "$OUT/runs/e18-baseline-wide/best.pt"
    train e18-dual-wide experiment=adm epochs=6 optimizer.schedule=cosine selection.patience=4 model=dual_resolution model.global_encoder.base_channels=32 model.local_encoder.base_channels=32
    evaluate e18-dual-wide "$OUT/runs/e18-dual-wide/best.pt"
    "$PY" -m scripts.experiments summarise "$ANALYSIS" >> "$OUT/logs/diagnostics.log" 2>&1
}

phase_seeds() {
    main_runs 29
    main_runs 43
    "$PY" -m scripts.experiments summarise "$ANALYSIS" >> "$OUT/logs/diagnostics.log" 2>&1
}

phases=${*:-perturb diagnostics seed17 builds ablations seeds}
for phase in $phases; do
    log "=== phase $phase"
    "phase_$phase"
done
log "=== plan finished"
