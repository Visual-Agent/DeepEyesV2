<div align="center">
  <img src="assets/logo-deepeyes.jpg" alt="logo" height="100">
  <h1 style="font-size: 32px; font-weight: bold;"> DeepEyesV2: Toward Agentic Multimodal Model </h1>

  <br>

  <a href="https://arxiv.org/abs/2505.14362">
    <img src="https://img.shields.io/badge/ArXiv-DeepEyes-brown?logo=arxiv" alt="Paper">
  </a>
  <a href="https://huggingface.co/datasets/honglyhly/DeepEyesV2_SFT">
    <img src="https://img.shields.io/badge/🤗 huggingface-SFT Data-blue" alt="sft dataset">
  </a>
  <a href="https://huggingface.co/datasets/honglyhly/DeepEyesV2_RL">
    <img src="https://img.shields.io/badge/🤗 huggingface-RL Data-blue" alt="rl dataset">
  </a>
  <a href="https://huggingface.co/honglyhly/DeepEyesV2_7B_1031">
    <img src="https://img.shields.io/badge/🤗 huggingface-Model-purple" alt="checkpoint">
  </a>
  <a href="https://visual-agent.github.io/">
    <img src="https://img.shields.io/badge/-HomePage-black?logo=github" alt="homepage">
  </a>
</div>

*\* Logo inspired by oracle bone character "eye".*


## DeepEyesV2
Quote from [https://openai.com/index/thinking-with-images/](https://openai.com/index/thinking-with-images/)
> They don’t just see an image, they can integrate visual information directly into the reasoning chain.


![](assets/fig3.png)

### What's new?
- We introduce ***DeepEyesV2***, an agentic multimodal model that unifies code execution and web search within a single reasoning loop, enabling reliable and complex reasoning. 
- We construct a carefully curated training corpus through rigorous
data filtering and cleaning to build both cold-start SFT data and RL data that complement each other.
![](assets/train_data.png)

- Extensive experiments across real-world understanding, mathematical reasoning, and search-intensive benchmarks demonstrate the strong reasoning and tool-usage ability of DeepEyesV2.
![](assets/result.png)

- We analyze the dynamics of tool-use behavior in DeepEyesV2, revealing task-adaptive patterns. Besides, we also find reinforcement learning can enable more complex tool combinations and adaptive, context-aware tool invocation.
![](assets/dynamic.png)



## Quick Start

### DATA
- Cold Start: [https://huggingface.co/datasets/honglyhly/DeepEyesV2_SFT](DeepEyesV2_SFT)
- RL: [https://huggingface.co/datasets/honglyhly/DeepEyesV2_RL](DeepEyesV2_RL)

Please refer to [data](./data/README.md) for data preparation.


### Cold Start
We use [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) to conduct cold start.

#### Environment Setup
Please refer to [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for more details on installing.

#### Cold Start Training
We use [Qwen-2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) as our foundation model. [Qwen-2.5-VL-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-32B-Instruct) is also supported.


```bash
bash ./cold_start/run_cold_start.sh
```

### Reinforcement Training
We use the same codebase as [DeepEyes](https://github.com/Visual-Agent/DeepEyes/tree/main).

#### Environment Setup

```bash
cd reinforcement_learning
# Follow the VeRL official installation procedure
pip install -e .

# Additional dependencies required by DeepEyes
bash scripts/install_deepeyes.sh
```

#### Code Sandbox
DeepEyesV2 can write code, execute it in a sandbox, and perform subsequent reasoning based on the execution results. Similar to o3, DeepEyesV2' code follows the Jupyter style. We receive DeepEyesV2's code via a server and return the execution results.

You can follow this [repo](https://github.com/ChenShawn/jupyter_sandbox) to deploy the code server. We utilize Redis to store the state of Jupyter Notebook, and you need to install Redis yourself.

During RL training, transmitting a large number of high-resolution images to a single node in a short period may saturate the bandwidth, leading to retransmissions and timeouts. Therefore, we recommend deploying multiple code servers to distribute network pressure and using high-performance machines to run the code servers, ensuring fast response. 

In our practice, we deployed a large number of code server services on each GPU node separately, and you can refer to this [repo](https://github.com/ChenShawn/jupyter_sandbox) for deployment. Through this approach, no timeouts caused by network, CPU, or other hardware issues occurred during our training process.


#### Online Search API
DeepEyesV2 can acquire external knowledge through searching. In our implementation, we use online search instead of RAG. For image search, we use the cache of [MMSearch-R1](https://arxiv.org/abs/2506.20670), and you can download the corresponding cache data from [here](https://huggingface.co/datasets/honglyhly/DeepEyesV2_Search_Cache). For text search, we use an online search service; you can use your own search API and modify the content in `reinforcement_learning/verl/workers/agent/envs/deepeyesv2/search_utils.py`.



#### Judge Server
We utilize Qwen for llm-as-a-judge verification. Please follow [DeepEyes](https://github.com/Visual-Agent/DeepEyes/tree/main) for instruction to deploy a server.

```bash
# download Qwen-2.5-72B-Instruct model
# Ohter models, such as Qwen-3-8B are also ok.
huggingface-cli download --resume-download https://huggingface.co/Qwen/Qwen2.5-72B-Instruct --local-dir /path/to/your/local/filedir --local-dir-use-symlinks False

# start vllm serving
vllm serve /path/to/your/local/filedir \
    --port 18901 \
    --gpu-memory-utilization 0.8 \
    --max-model-len 32768 \
    --tensor-parallel-size 8 \ 
    --served-model-name "judge" \
    --trust-remote-code \
    --disable-log-requests
```

#### Reinforcement Learning Optimization

We recommend using no less than 32 GPUs (4 nodes x 8 GPUs) for 7B training, and no less than 64 GPUs (8 nodes x 8 GPUs) for 32B training. For each node, we recommend using no less than 1200GB CPU RAM, as the high resolution images can consume large amount of memory.

Build a ray cluster for all of the training nodes. Prepare data before starting training. Then, use the following scripts to start training.


```bash
cd reinforcement_learning
# your wandb access key here...
wandb login

# the IP and port for your Qwen-2.5-72B-Instruct vllm serving
export LLM_AS_A_JUDGE_BASE="http://your.vllm.machine.ip:18901/v1"

# config for 7B
bash examples/deepeyesv2/run_qwen2_5_vl-7b_final_allin.sh
```

The training scripts use both [wandb](https://wandb.ai/site/) and [RL Logging Board](https://github.com/HarderThenHarder/RLLoggingBoard) (great work) to visualize the training dynamics.

### Evaluation

Please see [evaluation](./evaluation/VLMEvalKit/README.md) for more details.


## Star Chart

[![Star History Chart](https://api.star-history.com/svg?repos=Visual-Agent/DeepEyes&type=Date)](https://star-history.com/#Visual-Agent/DeepEyes&Date)

## Licence

This project is released under [Apache licence](./LICENSE).

## Citation

```
@article{zheng2025deepeyesincentivizingthinkingimages,
    title={DeepEyes: Incentivizing "Thinking with Images" via Reinforcement Learning}, 
    author={Ziwei Zheng, Michael Yang, Jack Hong, Chenxiao Zhao, Guohai Xu, Le Yang, Chao Shen, Xing Yu},
    year={2025},
    eprint={2505.14362},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/2505.14362}, 
}
```
