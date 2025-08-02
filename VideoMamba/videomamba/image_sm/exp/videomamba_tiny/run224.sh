export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1
echo "PYTHONPATH: ${PYTHONPATH}"
which_python=$(which python)
echo "which python: ${which_python}"
export PYTHONPATH=${PYTHONPATH}:${which_python}
export PYTHONPATH=${PYTHONPATH}:.
echo "PYTHONPATH: ${PYTHONPATH}"

JOB_NAME='videomamba_tiny_res224'
OUTPUT_DIR="$(dirname $0)"
LOG_DIR="./logs/${JOB_NAME}"
PARTITION='video5'
NNODE=1
NUM_GPUS=8
NUM_CPU=128

srun --mpi=pmi2 \
    -p ${PARTITION} \
    -n${NNODE} \
    --gres=gpu:${NUM_GPUS} \
    --ntasks-per-node=1 \
    --cpus-per-task=${NUM_CPU} \
    python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --use_env main.py \
        --root_dir_train your_imagenet_path/train/ \
        --meta_file_train your_imagenet_path/meta/train.txt \
        --root_dir_val your_imagenet_path/val/ \
        --meta_file_val your_imagenet_path/meta/val.txt \
        --model videomamba_tiny \
        --batch-size 512 \
        --num_workers 16 \
        --num_workers 16 \
        --lr 5e-4 \
        --clip-grad 5.0 \
        --weight-decay 0.1 \
        --drop-path 0 \
        --no-repeated-aug \
        --aa v0 \
        --no-model-ema \
        --output_dir ${OUTPUT_DIR}/ckpt \
        --bf16 \
        --dist-eval