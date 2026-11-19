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
# Experiment 1 showed that ADM image borders are a shortcut (generated images are
# whole pictures, real ones are crops). New runs therefore train on the inner
# 224 px of every image (both classes); the 112 px runs use the inner 112 px.
ADM224=data/genimage_adm_224/samples.jsonl
ADM112=data/genimage_adm_112/samples.jsonl
ADM20K224=data/genimage_adm_20k_224
BIGGAN112=data/genimage_biggan_112/samples.jsonl
BIGGAN128B8=data/genimage_biggan_128_border8/samples.jsonl
BIGGAN_TRAIN112=data/genimage_biggan_train_112/samples.jsonl
JOINT=data/joint_adm_biggan_112/samples.jsonl
PERTURBED=data/perturbed_adm
PERTURBED224=data/perturbed_adm224
PERTURBATIONS="border_gray:16 border_gray:32 border_reflect:32 jpeg:90 jpeg:75 jpeg:50 resample:0.75 blur:1 lowpass:0.125 highpass:0.125 shuffle:32 shuffle:64"
PERTURBATIONS224="jpeg:90 jpeg:75 jpeg:50 resample:0.75 blur:1 lowpass:0.125 highpass:0.125 shuffle:32 shuffle:56"
BASE_TESTS="--test adm224=$ADM224:test_id --test adm256=$ADM:test_id --test adm128=$ADM128:test_id --test biggan128=$BIGGAN128:test_id --test biggan128_border8=$BIGGAN128B8:test_id"
TESTS112="--test adm112=$ADM112:test_id --test biggan112=$BIGGAN112:test_id"

op_name() { local name=${1/:/}; echo "${name//./}"; }

perturbed_tests() {  # perturbed_tests <prefix> <directory> <operations>
    local args=""
    for op in $3; do
        args="$args --test $1_$(op_name "$op")=$2/$(op_name "$op")/samples.jsonl:test_id"
    done
    echo "$args"
}

derive() {  # derive <manifest> <split|all> <output_dir> <operation>
    [ -f "$3/samples.jsonl" ] && return
    rm -rf "$3"
    log "derive $3 ($4)"
    "$PY" -m scripts.experiments perturb "$1" "$2" "$3" "$4" >> "$OUT/logs/data.log" 2>&1 || log "FAIL derive $3"
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

occlude() {  # occlude <name> <checkpoint> <manifest>
    [ -f "$ANALYSIS/occlusion/$1/occlusion.json" ] && return
    [ -f "$2" ] || return
    log "occlusion $1"
    "$PY" -m scripts.experiments occlusion "$2" "$3" test_id "$ANALYSIS/occlusion/$1" >> "$OUT/logs/diagnostics.log" 2>&1 || log "FAIL occlusion $1"
}

summarise() { "$PY" -m scripts.experiments summarise "$ANALYSIS" >> "$OUT/logs/diagnostics.log" 2>&1 || log "FAIL summarise"; }

MAIN="experiment=adm data.manifest=$ADM224 epochs=15 optimizer.schedule=cosine selection.patience=4"
SHORT="experiment=adm data.manifest=$ADM224 epochs=10 optimizer.schedule=cosine selection.patience=4"
SHORT112="experiment=adm epochs=10 optimizer.schedule=cosine selection.patience=4 views.global_size=112 views.local_size=112"

phase_perturb() {
    for op in $PERTURBATIONS; do
        derive "$ADM" test_id "$PERTURBED/$(op_name "$op")" "$op"
    done
    derive "$ADM" all "$(dirname "$ADM224")" crop:224
    derive "$ADM224" all "$(dirname "$ADM112")" crop:112
    derive "$BIGGAN128" test_id "$(dirname "$BIGGAN128B8")" border_gray:8
    derive "$BIGGAN128" test_id "$(dirname "$BIGGAN112")" crop:112
    for op in $PERTURBATIONS224; do
        derive "$ADM224" test_id "$PERTURBED224/$(op_name "$op")" "$op"
    done
}

phase_diagnostics() {  # experiments 1, 2, 11, 13, 14 on the existing checkpoints
    for model in baseline dual; do
        evaluate "old-adm256-$model" "outputs/adm-$model/best.pt" $(perturbed_tests adm256 "$PERTURBED" "$PERTURBATIONS")
        evaluate "old-adm128-$model" "outputs/adm128-$model/best.pt"
        occlude "old-adm256-$model" "outputs/adm-$model/best.pt" "$ADM"
    done
    if [ ! -f "$ANALYSIS/spectra/spectra.json" ]; then
        log "spectra"
        "$PY" -m scripts.experiments spectra "$ANALYSIS/spectra" "adm256=$ADM:test_id" "adm224=$ADM224:test_id" "adm128=$ADM128:test_id" "biggan128=$BIGGAN128:test_id" >> "$OUT/logs/diagnostics.log" 2>&1 || log "FAIL spectra"
    fi
    summarise
}

main_runs() {  # experiments 3, 4, 5 for one seed
    local seed=$1
    for model in baseline local_only dual_resolution; do
        local name=e3-${model%%_*}-s$seed
        train "$name" $MAIN model=$model seed=$seed
        evaluate "$name" "$OUT/runs/$name/best.pt" $(perturbed_tests adm224 "$PERTURBED224" "$PERTURBATIONS224")
    done
}

phase_seed17() {
    main_runs 17
    for model in baseline local dual; do
        occlude "e3-$model-s17" "$OUT/runs/e3-$model-s17/best.pt" "$ADM224"
    done
    summarise
}

phase_builds() {
    if [ ! -f "$ADM20K/samples.jsonl" ]; then
        rm -rf "$ADM20K"
        log "build $ADM20K"
        "$PY" -m scripts.prepare_genimage C:/data/ADM/imagenet_ai_0508_adm.zip "$ADM20K" --train 20000 --val 1000 --test 1000 >> "$OUT/logs/data.log" 2>&1 || log "FAIL build adm20k"
    fi
    derive "$ADM20K/samples.jsonl" all "$ADM20K224" crop:224
    for n in 2000 4000 10000 20000; do
        [ -f "$ADM20K224/samples_$n.jsonl" ] || "$PY" -m scripts.experiments subset "$ADM20K224/samples.jsonl" $n "$ADM20K224/samples_$n.jsonl" >> "$OUT/logs/data.log" 2>&1
    done
    if [ ! -f "$BIGGAN_TRAIN" ]; then
        rm -rf "$(dirname "$BIGGAN_TRAIN")"
        log "build biggan train"
        "$PY" -m scripts.prepare_genimage C:/data/BigGAN/imagenet_ai_0419_biggan.zip "$(dirname "$BIGGAN_TRAIN")" --train 4000 --val 1000 --test 0 --size 128 --allow-missing-volumes >> "$OUT/logs/data.log" 2>&1 || log "FAIL build biggan train"
    fi
    derive "$BIGGAN_TRAIN" all "$(dirname "$BIGGAN_TRAIN112")" crop:112
    if [ ! -f "$JOINT" ]; then
        log "merge joint ADM+BigGAN"
        "$PY" -m scripts.experiments merge "$JOINT" "$ADM112:2000:500" "$BIGGAN_TRAIN112:2000:500" >> "$OUT/logs/data.log" 2>&1 || log "FAIL merge"
    fi
}

phase_ablations() {
    # 15: augmentation
    for model in baseline dual_resolution; do
        local name=e15-${model%%_*}-aug
        train "$name" $MAIN model=$model augment.probability=0.5
        evaluate "$name" "$OUT/runs/$name/best.pt" $(perturbed_tests adm224 "$PERTURBED224" "$PERTURBATIONS224")
    done
    # 6: train on BigGAN (inner 112 px), test on ADM; reference: ADM at the same setup
    for model in baseline dual_resolution; do
        local name=e6-${model%%_*}-biggan
        train "$name" $SHORT112 model=$model data.manifest=$BIGGAN_TRAIN112
        evaluate "$name" "$OUT/runs/$name/best.pt" $TESTS112
        name=e6-${model%%_*}-adm112
        train "$name" $SHORT112 model=$model data.manifest=$ADM112
        evaluate "$name" "$OUT/runs/$name/best.pt" $TESTS112
    done
    # 21: train on ADM + BigGAN (both inner 112 px)
    for model in baseline dual_resolution; do
        local name=e21-${model%%_*}-joint
        train "$name" $SHORT112 model=$model data.manifest=$JOINT
        evaluate "$name" "$OUT/runs/$name/best.pt" $TESTS112
    done
    # 16: training-set size
    for n in 2000 4000 10000 20000; do
        local name=e16-baseline-n$n
        train "$name" $SHORT model=baseline data.manifest=$ADM20K224/samples_$n.jsonl
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
    train e18-dual-wide experiment=adm data.manifest=$ADM224 epochs=6 optimizer.schedule=cosine selection.patience=4 model=dual_resolution model.global_encoder.base_channels=32 model.local_encoder.base_channels=32
    evaluate e18-dual-wide "$OUT/runs/e18-dual-wide/best.pt"
    summarise
}

phase_seeds() {
    main_runs 29
    main_runs 43
    summarise
}

phases=${*:-perturb diagnostics seed17 builds ablations seeds}
for phase in $phases; do
    log "=== phase $phase"
    "phase_$phase"
done
log "=== plan finished"
