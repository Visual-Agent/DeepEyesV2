set -x
export HYDRA_FULL_ERROR=1


PROJECT_NAME="deepeyesv2"
EXPERIMENT_NAME="deepeyesv2"
export SAVE_CHECKPOINT_DIR=./save_checkpoints


PERCEPTION_TRAIN_PARQUET_1=../data/rl/perception_all_1.parquet
PERCEPTION_TRAIN_PARQUET_2=../data/rl/perception_all_2.parquet
PERCEPTION_TRAIN_PARQUET_3=../data/rl/perception_all_3.parquet
PERCEPTION_TRAIN_PARQUET_4=../data/rl/perception_all_4.parquet
PERCEPTION_TRAIN_PARQUET_5=../data/rl/perception_all_5.parquet
SEARCH_TRAIN_PARQUET=../data/rl/search.parquet
REASON_TRAIN_PARQUET=../data/rl/reason.parquet

VSTAR_TEST_PARQUET=../data/rl/vstar_test.parquet

CUSTOM_STOP='["</code>","</tool_call>"]'
LOSS_AGG_MODE="token-mean"
export WORKING_DIR=${WORKING_DIR:-"${PWD}"}
export RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/verl/trainer/runtime_env.yaml"}

REF_MODEL_PATH=REF_MODEL

python3 -m verl.trainer.main_ppo \
    +debug=False \
    +vs_debug=False \
    algorithm.adv_estimator=grpo \
    data.train_files=[${PERCEPTION_TRAIN_PARQUET_1},${PERCEPTION_TRAIN_PARQUET_2},${PERCEPTION_TRAIN_PARQUET_3},${PERCEPTION_TRAIN_PARQUET_4},${PERCEPTION_TRAIN_PARQUET_5},${SEARCH_TRAIN_PARQUET},${REASON_TRAIN_PARQUET}] \
    data.val_files=[${VSTAR_TEST_PARQUET}] \
    data.train_batch_size=256 \
    data.max_prompt_length=8192 \
    data.max_response_length=16384 \
    data.return_raw_chat=True \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.image_key=images \
    actor_rollout_ref.model.path=${REF_MODEL_PATH} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.checkpoint.save_contents=['model','hf_model','optimizer','extra'] \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.agent.activate_agent=True \
    actor_rollout_ref.rollout.agent.tool_name_key=env_name \
    actor_rollout_ref.rollout.agent.single_response_max_tokens=8192 \
    actor_rollout_ref.rollout.agent.max_turns=9 \
    actor_rollout_ref.rollout.agent.concurrent_workers=2 \
    actor_rollout_ref.rollout.agent.custom_stop=${CUSTOM_STOP} \
    actor_rollout_ref.rollout.agent.show_tqdm=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    reward_model.reward_manager=naive_async \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb','rl_logging_board'] \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=4 \
    trainer.save_freq=4 \
    trainer.test_freq=4 \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.default_local_dir=${SAVE_CHECKPOINT_DIR}/${PROJECT_NAME}/${EXPERIMENT_NAME} \
    +trainer.tensorboard_dir=${SAVE_CHECKPOINT_DIR}/logs/tensorboard \
    +trainer.rl_logging_board_dir=${SAVE_CHECKPOINT_DIR}/logs/rl_logging_board \
    trainer.total_epochs=32 2>&1 | tee ./logs/${EXPERIMENT_NAME}.log
