import re
import os
import json
import base64
import uuid
import requests
import numpy as np
import copy
from typing import Optional, List, Dict, Any
from PIL import Image
from io import BytesIO
from math import ceil, floor
import time
import random
import autopep8
import textwrap

from verl.workers.agent.tool_envs import ToolBase
from verl.workers.agent.envs.deepeyesv2.prompt import RETURN_CODE_PROMPT, RETURN_SEARCH_PROMPT
from verl.workers.agent.envs.deepeyesv2.search_utils import search, image_search


INITIALIZATION_CODE_TEMPLATE = """
from PIL import Image
import base64
from io import BytesIO

_img_base64 = "{base64_image}"
image_1 = Image.open(BytesIO(base64.b64decode(_img_base64)))
"""

CODE_EXECUTION_TEMPLATE = RETURN_CODE_PROMPT
SEARCH_EXECUTION_TEMPLATE = RETURN_SEARCH_PROMPT

def pil_image_to_base64(img: Image.Image, format: str = "PNG") -> str:
    buffer = BytesIO()
    img.save(buffer, format=format)
    buffer.seek(0)
    img_bytes = buffer.read()
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    return img_base64

def base64_to_pil_image(base64_string: str) -> Image.Image:
    image_data = base64.b64decode(base64_string)
    image = Image.open(BytesIO(image_data))
    return image

def generate_session_id():
    salted_str = str(int(time.time())) + str(random.randint(10000, 99999))
    salted_hash_str = str(hex(hash(salted_str.encode('utf-8')))).split('0x')[-1]
    return salted_hash_str

def fix_python_indentation(code):
    try:
        lines = [line.strip() for line in code.split('\n') if line.strip()]
        fixed_lines = []
        indent = 0
        for line in lines:
            if any(line.startswith(kw) for kw in ['except', 'elif', 'else', 'finally']):
                indent = max(0, indent - 1)
            fixed_lines.append('    ' * indent + line)
            if line.endswith(':'):
                indent += 1
        temp_code = '\n'.join(fixed_lines)
        dedented_code = textwrap.dedent(temp_code).strip()
        formatted_code = autopep8.fix_code(dedented_code, options={'aggressive': 2})
        return formatted_code
    except Exception as e:
        print ('Code Format Error:', e)
        return code


class DeepEyesV2_ENV(ToolBase):
    name = "deepeyes_v2"
    code_sandbox_url = "http://127.0.0.1:8000/jupyter_sandbox"
    session_id = generate_session_id()
    max_images_per_round = 10

    def __init__(self, _name, _desc, _params, **kwargs):
        super().__init__(
            name=self.name,
        )
        self.chatml_history = []
        self.multi_modal_data = None  # To store the current image being processed

    def extract_answer(self, action_string: str) -> str:
        answer = re.findall(r'<answer>(.*?)</answer>', action_string, re.DOTALL)
        return answer[-1] if answer else None

    def extract_python_code(self, action_string: str) -> str:
        tool_call_match = re.findall(r'<code>(.*?)</code>', action_string, re.DOTALL)
        if not tool_call_match:
            return None
        
        last_code_block = tool_call_match[-1]
        pattern = r'```python\s*\n(.*?)\n```'
        code_match = re.findall(pattern, last_code_block, re.DOTALL)
        if not code_match:
            return None
        return code_match[-1]
    
    def extract_action(self, action_string: str) -> Dict[str, Any]:
        """
        Extracts the tool call from the action string.
        
        Args:
            action_string: The string containing the tool call in XML tags.
            
        Returns:
            A dictionary with the tool name and arguments.
            
        Raises:
            ValueError: If no tool call is found or JSON is invalid.
        """
        tool_call_match = re.findall(r'<tool_call>(.*?)</tool_call>', action_string, re.DOTALL)
        return tool_call_match[-1] if tool_call_match else None

    def execute(self, action_string: str, **kwargs) -> tuple:
        """
        Execute the tool functionality based on the action string.
        
        Args:
            action_string: The string containing the tool call in XML tags.
            
        Returns:
            observation: The structured observation with the processed image.
            reward: 0.1 if tool call is successful with correct JSON format, 0 otherwise.
            done: Whether the episode is terminated.
            info: Additional info.
        """
        answer = self.extract_answer(action_string)
        if answer:
            return "", 0.0, True, {}

        todo_action = 'code'
        if '<code>' in action_string:
            todo_action = 'code'
        elif '<tool_call>' in action_string:
            todo_action = 'action'
        
        if todo_action == 'code':
            code_string = self.extract_python_code(action_string)
            if not code_string:
                return "", 0.0, True, {}

            code_string = fix_python_indentation(code_string)
            exec_ret = self.request_jupyter_execution(code_string)
            if not exec_ret or exec_ret['status'] != 'success':
                obs = "Code execution error"
                return obs, 0.0, True, {"error": "Code execution failed"}

            image_list = exec_ret.get('images', [])
            image_list = image_list[:self.max_images_per_round]
            code_result_string = CODE_EXECUTION_TEMPLATE.format(
                stdout=exec_ret.get('stdout', ''),
                stderr=exec_ret.get('stderr', ''),
                image="Images:\n" + "<image>" * len(image_list) if len(image_list) > 0 else "",
            ).strip()
            print (f' [DEBUG code] Code Runing Result: {code_result_string=}')

            if len(image_list) == 0:
                obs = "<|im_end|>\n<|im_start|>user\n" + code_result_string + "<|im_end|>\n<|im_start|>assistant\n<think>"
                return obs, 0.0, False, exec_ret
            else:
                obs = {
                    "prompt": "<|im_end|>\n<|im_start|>user\n" + code_result_string + "<|im_end|>\n<|im_start|>assistant\n<think>",
                    "multi_modal_data": {"image": image_list},
                }
                return obs, 0.0, False, exec_ret
        elif todo_action=='action':
            action = self.extract_action(action_string)

            if not action:
                return "", 0.0, True, {}

            try:
                tool_call = json.loads(action.strip())
            except Exception as e:
                error_msg = f"Invalid tool call format: {action.strip()}. Error: {e}"
                obs = "<|im_end|>\n<|im_start|>user\n" + f"Error: {str(error_msg)}" + "<|im_end|>\n<|im_start|>assistant\n<think>"
                info = {"error": str(e), "status": "failed"}
                return obs, 0.0, False, {}
            
            try:
                tool_name = tool_call["name"]
                args = tool_call.get("arguments", None)

                # error process
                if tool_name not in ['search', 'image_search']:
                    error_msg = f"Invalid tool call name: {action.strip()}."
                    obs = "<|im_end|>\n<|im_start|>user\n" + f"Error: {str(error_msg)}" + "<|im_end|>\n<|im_start|>assistant\n<think>"
                    info = {"error": str(error_msg), "status": "failed"}
                    return obs, 0.0, False, {}
                if tool_name == 'image_search' and args is not None:
                    error_msg = f"Invalid tool call parameters for image search: {action.strip()}."
                    obs = "<|im_end|>\n<|im_start|>user\n" + f"Error: {str(error_msg)}" + "<|im_end|>\n<|im_start|>assistant\n<think>"
                    info = {"error": str(error_msg), "status": "failed"}
                    return obs, 0.0, False, {}

                exec_ret = self.request_search(tool_name, args)
                if not exec_ret or exec_ret['status'] != 'success':
                    obs = "Search error"
                    return obs, 0.0, True, {"error": "Search failed"}
                
                image_list = exec_ret.get('images', [])
                image_list = image_list[:self.max_images_per_round]
                search_result_string = SEARCH_EXECUTION_TEMPLATE.format(
                    search_result=exec_ret.get('result', ''),
                ).strip()


                print (f' [DEBUG search] Search Result: {search_result_string=}')

                if len(image_list) == 0:
                    obs = "<|im_end|>\n<|im_start|>user\n" + search_result_string + "<|im_end|>\n<|im_start|>assistant\n<think>"
                    return obs, 0.0, False, exec_ret
                else:
                    obs = {
                        "prompt": "<|im_end|>\n<|im_start|>user\n" + search_result_string + "<|im_end|>\n<|im_start|>assistant\n<think>",
                        "multi_modal_data": {"image": image_list},
                    }
                    return obs, 0.0, False, exec_ret
            except Exception as e:
                obs = "<|im_end|>\n<|im_start|>user\n" + f"Error: {str(e)} for {self.data_index}" + "<|im_end|>\n<|im_start|>assistant\n<think>"
                print (f' [ERROR search] Search Error: {str(e)} for {self.data_index} with input {action_string=}')
                reward = 0.0  # No reward for failed execution
                done = False
                info = {"error": str(e), "status": "error"}
                return obs, reward, done, info
        else:
            obs = "Format Error"
            return obs, 0.0, True, {"error": "Format Error"}



    def reset(self, raw_prompt, multi_modal_data, origin_multi_modal_data, extra_info, **kwargs):
        self.chatml_history = raw_prompt
        self.multi_modal_data = origin_multi_modal_data
        self.data_index = extra_info['index']
        assert 'image' in self.multi_modal_data.keys(), f'[ERROR] {origin_multi_modal_data=}'
        assert len(self.multi_modal_data['image']) > 0, f'[ERROR] {self.multi_modal_data["image"]=}'

        base64_image = pil_image_to_base64(self.multi_modal_data['image'][0])
        init_code_string = INITIALIZATION_CODE_TEMPLATE.format(base64_image=base64_image)
        init_ret = self.request_jupyter_execution(init_code_string)
        
        if not init_ret:
            print(f' [ERROR code] Initialization code execution failed: init_ret is None')
            return None
            
        stdout = init_ret.get('stdout', '')
        stderr = init_ret.get('stderr', '')

        print (f' [DEBUG code] Code Init Result: {stdout=} {stderr=}')

        if init_ret['status'] != 'success':
            print(f' [ERROR code] Initialization code execution failed: {init_ret}')
        return init_ret
    
    def request_search(self, tool_name, tool_args, request_timeout=240):
        if tool_name == 'image_search':
            image_pil_list = []
            result = image_search(tool_args, self.data_index)
            if result == 'Error':
                status = 'error'
                execution_time = -1.0
                content = 'Error'
            else:
                status = 'success'
                execution_time = 1.0
                tool_returned_web_title = result['tool_returned_web_title']
                cached_images_path = result['cached_images_path']
                web_snippets = []
                try:
                    for idx, (title, link) in enumerate(zip(tool_returned_web_title, cached_images_path)):
                        date_published = ""
                        snippet = ""

                        img = Image.open(link)
                        image_pil_list.append(img)
                        
                        redacted_version = f"{idx+1}. <image>\n[{title}] {date_published}\n"
                        redacted_version = redacted_version.replace("Your browser can't play this video.", "")
                        web_snippets.append(redacted_version)
                    content = f"A Google image search for the image found {len(web_snippets)} results:\n\n## Web Results\n" + "\n\n".join(web_snippets)
                except Exception as e:
                    status = 'error'
                    image_pil_list = []
                    content = str(e) + f"No results found for the image. Try with text search or direct output the answer."
        elif tool_name == 'search':
            image_pil_list = []
            time.sleep(random.randint(1,10))
            query = tool_args['query']
            result = search(query)
            if result == 'Error':
                status = 'error'
                execution_time = -1.0
                content = 'Error'
            else:
                status = 'success'
                execution_time = result['elapsed_time']
                search_content = result['data']
                web_snippets = []
                try:
                    for idx, page in enumerate(search_content):

                        date_published = ""
                        if "date" in page and page['date'] is not None:
                            date_published = "\nDate published: " + page["date"]
                        snippet = ""
                        if "snippet" in page and page['snippet'] is not None:
                            snippet = "\n" + page["snippet"]

                        redacted_version = f"{idx+1}. [{page['title']}]({page['link']}){date_published}\n{snippet}"
                        redacted_version = redacted_version.replace("Your browser can't play this video.", "")
                        web_snippets.append(redacted_version)
                    content = f"A Google search for '{query}' found {len(web_snippets)} results:\n\n## Web Results\n" + "\n\n".join(web_snippets)
                except Exception as e:
                    status = 'error'
                    content = str(e) + f"No results found for '{query}'. Try with a more general query."

        return dict(
            status=status,
            execution_time=execution_time,
            result=content,
            images=image_pil_list,
        )



    def request_jupyter_execution(self, code_string, code_timeout=200, request_timeout=240):
        try:
            resjson = requests.post(
                self.code_sandbox_url,
                json={
                    "session_id": self.session_id,
                    "code": code_string,
                    "timeout": code_timeout
                },
                timeout=request_timeout
            ).json()
            result_dict = resjson['output']
        except Exception as err:
            print(f' [ERROR code] Request to Jupyter sandbox failed: {err}')
            return None

        image_pil_list = []
        image_base64_list = result_dict.get("images", [])
        for idx, img in enumerate(image_base64_list):
            try:
                img_pil = base64_to_pil_image(img)
                img_pil = self.maybe_resize_image(img_pil)
                image_pil_list.append(img_pil)
            except Exception as err:
                print(f' [ERROR code] Failed to decode image {idx}: {err}')
                continue

        return dict(
            status=resjson.get("status", "error"),
            execution_time=resjson.get("execution_time", -1.0),
            result=result_dict.get("result", ""),
            stdout=result_dict.get("stdout", ""),
            stderr=result_dict.get("stderr", ""),
            images=image_pil_list,
        )

    def maybe_resize_image(self, image):
        """
        Qwen-VL raises an error for images with height or width less than 32 pixels.
        """
        height, width = image.height, image.width
        if max(height, width) / min(height, width) > 200:
            max_val = max(height, width)
            min_val = min(height, width)

            old_scale = max_val / min_val

            max_ratio = min(150, old_scale / 2)
            target_max = int(min_val * max_ratio)

            if height > width:
                new_height = target_max
                new_width = int(width * old_scale / max_ratio)
            else:
                new_width = target_max
                new_height = int(height * old_scale / max_ratio)
            
            image = image.resize((int(new_width), int(new_height)), Image.LANCZOS)
            height, width = image.height, image.width

        if min(height, width) >= 32:
            return image

        ratio = 32 / min(height, width)
        new_height = ceil(height * ratio)
        new_width = ceil(width * ratio)
        new_image = image.resize((new_width, new_height), Image.LANCZOS)
        return new_image


if __name__ == "__main__":
    debug_action = """<think>The image shows a line graph of the annual inflation rate in China from 1987 to 2023. The graph does not provide data for 2024, so it's not possible to determine the inflation rate for that year based on the information given. Without additional data for 2024, we cannot compare the inflation rate to that of 2023.</think>
<tool_call>
{"name": "image_search"}
</tool_call>
""".strip()

    tool = DeepEyesV2_ENV("deepeyes_v2", 2, 3)
    obs, reward, done, info = tool.execute(debug_action)
    print(f"Result - Reward: {reward}, Info: {info}")
