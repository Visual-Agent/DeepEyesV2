## Evaluation

We adopt [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) to conduct the evaluation.

### Environment Setup
See the instruction of [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) for more details on installation.

### Pretrained Model

Download the pretrained checkpoints from [here](https://huggingface.co/honglyhly/DeepEyesV2_7B_1031).

### Usage

We utilize llm-as-a-judge to conduct the judgement of QA, the judgement operation is similar to that in RL training. You should set the `JUDGE_API_BASE` in `.env`.

After setting the judge server, you should deploy DeepEyesV2 as a server (you can use vLLM) and then modify the `MODEL_CONFIGS` in `eval.sh`.

```bash
MODEL_CONFIGS='{"DeepEyesV2-vllm": {"api_base": "http://10.39.6.41:28000,http://10.39.6.41:18000", "max_tokens": 20480}}'
```

The `api_base` is the IP address and port of the DeepEyesv2 server. You can start multiple servers to accelerate the evaluation, and note that commas should be used to separate different ports. Then, you can use the following command to conduct evaluation.

```bash
bash eval.sh
```




























