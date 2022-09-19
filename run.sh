DATA_DIR="/datasets/mnist/"
MODEL="LSTM"
DEVICE="cuda"
EPOCHS=100

OUT_DIR="results/"
mkdir -p "$OUT_DIR/models"
mkdir -p "$OUT_DIR/performances"
mkdir -p "$OUT_DIR/plots"

COMMON_ARGS="--data-path $DATA_DIR --device $DEVICE --model $MODEL --epochs $EPOCHS --output-dir $OUT_DIR"

for LR in 0.1 0.01 0.001
do
    for BATCH_SIZE in 16 32 64 128 256
    do
        for OPTIM in ADAM
        do
            EXPERIMENT_NAME="$MODEL-lr=$LR-bs=$BATCH_SIZE-epochs=$EPOCHS-opt=$OPTIM"
            cmd="python3 base_setup/train.py $COMMON_ARGS --lr $LR --batch-size $BATCH_SIZE --opt $OPTIM --experiment-name $EXPERIMENT_NAME"
            bash -c "$cmd"
            # echo "$cmd"
        done
    done
done