# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register

from tqdm import tqdm
from functools import partial
from concurrent.futures import ThreadPoolExecutor

def unit_compute_reward_func(sample, reward_func, pbar=None):
    data_source = sample["data_source"]
    response_str = sample["response_str"]
    ground_truth = sample["ground_truth"]
    extra_info = sample["extra_info"]

    result = reward_func(
        data_source=data_source,
        solution_str=response_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
    )

    if pbar is not None:
        pbar.update(1)
    return result


@register("dapo_async")
class AsyncDAPORewardManager:
    """The reward manager."""

    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key="data_source",
        max_resp_len=None,
        overlong_buffer_cfg=None,
        **kwargs
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        self.overlong_buffer_cfg = overlong_buffer_cfg
        self.max_resp_len = max_resp_len

        if self.overlong_buffer_cfg is not None:
            assert self.max_resp_len is not None, f"max_resp_len must be provided if {overlong_buffer_cfg=}, but got None"

        self.reward_kwargs = kwargs

    def __call__(self, data: DataProto, return_dict: bool = False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        action_or_attn_mask = data.batch['action_mask'] if 'action_mask' in data.batch.keys() else data.batch['attention_mask']
        if 'env_reward' in data.batch.keys():
            reward_tensor += data.batch['env_reward']
            print(f' [DEBUG reward] mean={reward_tensor.mean().item()}, min={reward_tensor.min().item()}, max={reward_tensor.max().item()}')

        input_ds = []
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids)
            response_str = self.tokenizer.decode(valid_response_ids)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]

            data_source = data_item.non_tensor_batch[self.reward_fn_key]

            extra_info = data_item.non_tensor_batch.get("extra_info", None)
            new_input_sample = dict(
                index=i,
                data_source=data_source,
                response_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
                valid_response_length=valid_response_length,
            )
            input_ds.append(new_input_sample)

        print(f" [*] Start computing reward on {len(input_ds)} samples...")
        num_workers = self.reward_kwargs.get("num_workers", 8)
        pbar = tqdm(total=len(input_ds), desc=f'Compute reward on {num_workers} workers')
        partial_reward_func = partial(unit_compute_reward_func, reward_func=self.compute_score, pbar=pbar)
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            reward_result_list = list(executor.map(partial_reward_func, input_ds))
        pbar.close()

        already_print_data_sources = {}
        for i, (sample, result) in enumerate(zip(input_ds, reward_result_list)):
            data_source = sample["data_source"]
            response_str = sample["response_str"]
            ground_truth = sample["ground_truth"]
            extra_info = sample["extra_info"]
            valid_response_length = sample["valid_response_length"]

            score: float
            if isinstance(result, dict):
                score = result["score"]
                # Store the information including original reward
                for key, value in result.items():
                    reward_extra_info[key].append(value)
            else:
                score = result

            reward = score
            reward_tensor[i, valid_response_length - 1] += reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(result, dict):
                    for key, value in result.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
