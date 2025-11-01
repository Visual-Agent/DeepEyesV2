# FrozenLake
1. An example for create frozenlake dataset

    create text type dataset: 
    ```bash
    export SIZE=8
    export P=0.8
    python verl/workers/agent/envs/frozenlake/create_dataset.py \
        --env frozenlake \
        --seed 1000 \
        --output data/frozenlake \
        --train_size 3000 \
        --test_size 100 \
        --prefix qwen-instruct
    ```

    create multi-model type dataset:
    ```bash
    export SIZE=8
    export P=0.8
    python verl/workers/agent/envs/frozenlake/create_dataset.py \
        --env frozenlake \
        --seed 1000 \
        --output data/frozenlake \
        --train_size 3000 \
        --test_size 100 \
        --prefix qwen-instruct \
        --use_mm
    ```
# Search(RAG)
