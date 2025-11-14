<div align="center">
  <img src="assets/logo-deepeyes.jpg" alt="logo" height="100">
  <h1 style="font-size: 32px; font-weight: bold;"> DeepEyesV2: Toward Agentic Multimodal Model </h1>

  <br>

  <a href="https://arxiv.org/abs/2511.05271">
    <img src="https://img.shields.io/badge/ArXiv-DeepEyes-brown?logo=arxiv" alt="Paper">
  </a>
  <a href="https://huggingface.co/datasets/honglyhly/DeepEyesV2_SFT">
    <img src="https://img.shields.io/badge/🤗 huggingface-SFT Data-blue" alt="sft dataset">
  </a>
  <a href="https://huggingface.co/datasets/honglyhly/DeepEyesV2_RL">
    <img src="https://img.shields.io/badge/🤗 huggingface-RL Data-blue" alt="rl dataset">
  </a>
  <a href="https://huggingface.co/honglyhly/DeepEyesV2_7B_SFT_1031">
    <img src="https://img.shields.io/badge/🤗 huggingface-SFT Model-purple" alt="checkpoint">
  </a>
  <a href="https://huggingface.co/honglyhly/DeepEyesV2_7B_1031">
    <img src="https://img.shields.io/badge/🤗 huggingface-RL Model-purple" alt="checkpoint">
  </a>
  <a href="https://visual-agent.github.io/">
    <img src="https://img.shields.io/badge/-HomePage-black?logo=github" alt="homepage">
  </a>
</div>

*\* Logo inspired by oracle bone character "eye".*

## 🔥 Updates
- [2025/11/14] Since the training time of cold start is relatively long, we have uploaded the cold-start checkpoint. You can download it from [here](https://huggingface.co/honglyhly/DeepEyesV2_7B_SFT_1031).


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

You can follow [this github repo](https://github.com/ChenShawn/MultiModal-Jupyter-Sandbox) to deploy the code server. To guarantee code execution safety, we recommend using docker virtualization to run the code sandbox. The docker image for the code sandbox can be found on [this dockerhub repo](https://hub.docker.com/repository/docker/chenshawn6915/multimodal-ipython-sandbox/general).

During RL training, transmitting a large number of high-resolution images to a single node in a short period may saturate the bandwidth, leading to retransmissions and timeouts. Therefore, we recommend deploying multiple code servers to distribute network pressure and using high-performance machines to run the code servers, ensuring fast response. 

In our practice, we deployed a large number of code server services on each GPU node separately, and each training process should only request the code server on its localhost machine. Through this approach, no timeouts caused by network, CPU, or other hardware issues occurred during our training process.


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

[![Star History Chart](https://api.star-history.com/svg?repos=Visual-Agent/DeepEyesV2&type=Date)](https://star-history.com/#Visual-Agent/DeepEyesV2&Date)

## Licence

This project is released under [Apache licence](./LICENSE).

## Citation

```
# DeepEyesV2
@article{hong2025deepeyesv2,
  title={DeepEyesV2: Toward Agentic Multimodal Model},
  author={Hong, Jack and Zhao, Chenxiao and Zhu, ChengLin and Lu, Weiheng and Xu, Guohai and Yu, Xing},
  journal={arXiv preprint arXiv:2511.05271},
  year={2025}
}
# DeepEyes
@article{zheng2025deepeyes,
  title={DeepEyes: Incentivizing" Thinking with Images" via Reinforcement Learning},
  author={Zheng, Ziwei and Yang, Michael and Hong, Jack and Zhao, Chenxiao and Xu, Guohai and Yang, Le and Shen, Chao and Yu, Xing},
  journal={arXiv preprint arXiv:2505.14362},
  year={2025}
}
```
