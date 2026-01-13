import os
from openai import OpenAI
import base64
from io import BytesIO
from PIL import Image
import requests
import time
import re
import random
import json
from math import ceil
from typing import Dict, Any
import textwrap
import autopep8

import math

# 新增：smart resize相关方法
IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28

def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor

def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor

def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor

def smart_resize(
    height: int, width: int, factor: int = IMAGE_FACTOR, min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS
) -> tuple[int, int]:
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar

def maybe_resize_image_v2(image):
    """
    Qwen-VL raises an error for images with height or width less than 32 pixels.
    """
    height, width = image.height, image.width
    new_height, new_width = smart_resize(height, width)
    if new_height == height and new_width == width:
        return image
    new_image = image.resize((new_width, new_height), Image.BILINEAR)
    return new_image


def generate_session_id():
    salted_str = str(int(time.time())) + str(random.randint(10000, 99999))
    salted_hash_str = str(hex(hash(salted_str.encode('utf-8')))).split('0x')[-1]
    return salted_hash_str

def encode_image_to_base64(image_path):
    """将图片转换为base64编码"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def base64_to_pil_image(base64_string: str) -> Image.Image:
    image_data = base64.b64decode(base64_string)
    image = Image.open(BytesIO(image_data))
    return image

def maybe_resize_image(image):
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
            # 高度是最大边，缩小高度到 target_max
            new_height = target_max
            new_width = int(width * old_scale / max_ratio)  # 宽度不变
        else:
            # 宽度是最大边，缩小宽度到 target_max
            new_width = target_max
            new_height = int(height * old_scale / max_ratio)  # 高度不变
        
        image = image.resize((int(new_width), int(new_height)), Image.LANCZOS)
        height, width = image.height, image.width

    if min(height, width) >= 32:
        return image

    ratio = 32 / min(height, width)
    new_height = ceil(height * ratio)
    new_width = ceil(width * ratio)
    new_image = image.resize((new_width, new_height), Image.LANCZOS)

    return new_image

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

def extract_action(action_string: str) -> Dict[str, Any]:
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

def extract_python_code(action_string: str) -> str:
    tool_call_match = re.findall(r'<code>(.*?)</code>', action_string, re.DOTALL)
    if not tool_call_match:
        return None
    
    last_code_block = tool_call_match[-1]
    pattern = r'```python\s*\n(.*?)\n```'
    code_match = re.findall(pattern, last_code_block, re.DOTALL)
    if not code_match:
        return None
    return code_match[-1]

def request_jupyter_execution(code_string, code_sandbox_url,session_id, code_timeout=200, request_timeout=240):
        try:
            resjson = requests.post(
                code_sandbox_url,
                json={
                    "session_id": session_id,
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
                img_pil = maybe_resize_image_v2(img_pil)
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

# This is just a placeholder for the actual search function implementation.
# You need to replace it with the actual implementation.
def request_search(tool_name, tool_args, request_timeout=240):
    if tool_name == 'image_search':
        image_pil_list = []

        result = {'tool_returned_web_title': [], 'cached_images_path': []}
        for i in range(5):
            result["tool_returned_web_title"].append(f"Sample Image Title {i+1}")
            result["cached_images_path"].append(f"sample_image_path_{i+1}.jpg")

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
        query = tool_args['query']

        result = {'elapsed_time': 0.0, 'data': []}
        for i in range(5):
            search_info = {}
            search_info['snippet'] = "This is a placeholder snippet for query: " + query
            search_info['title'] = "Placeholder Title " + str(i)
            search_info['link'] = "http://example.com/" + str(i)
            result['data'].append(search_info)

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


SYSTEM_PROMPT = """You are an agent - please keep going until the user's query is completely resolved, before ending your turn and yielding back to the user. Only terminate your turn when you are sure that the problem is solved.

Solve the following problem step by step. In your reasoning process, if the answer cannot be determined, you can write Python code in a Jupyter Notebook to process the image and extract more information from it. The stdout and stderr content, along with the images generated by `plt.show()` will be returned to better assist with the user query.

You MUST use the python tool to analyze or transform images whenever it could improve your understanding. This includes but is not limited to zooming in, rotating, adjusting contrast, computing statistics, or isolating features.

If you find you sufficient knowledge to confidently answer the question, you MUST conduct search to thoroughly seek the internet for information. No matter how complex the query, you will not give up until you find the corresponding information.

You can conduct image search, which will trigger a Google Lens search using the original image to retrieve relevant information that can help you confirm the visual content, and text search, which will use Google Search to return relevant information based on your query.

You MUST plan extensively before each function call, and reflect extensively on the outcomes of the previous function calls. DO NOT do this entire process by making function calls only, as this can impair your ability to solve the problem and think insightfully.

Additionally, you can combine python tool with search to assist in answering questions. Python tool can help enhance your understanding of images, while search tools can provide the knowledge you lack. Please use python tool and search flexibly. However, you can only call one type of tool in a single round; you cannot use a python tool and perform a search simultaneously.

For all the provided images, in order, the i-th image has already been read into the global variable "image_i" using the "PIL.Image.open()" function. When writing Python code, you can directly use these variables without needing to read them again.

## Tools

## python
Your python code should be enclosed within <code> </code> tag.

Example for calling Python code in Jupyter Notebook:
<code>
```python
# python code here
```
</code>

Note:
1. **python** can be called to analyze the image. **python** will respond with the output of the execution or time out after 300.0 seconds. 
2. Like jupyter notebook, you can use Python code to process the input image and use `plt.show()` to visualize processed images in your code.
3. All python code are running in the same jupyter notebook kernel, which means the functions and variables are automatically stored after code execution.
4. You program should always returns in finite time. Do not write infinite loop in your code.
5. Writing file to disk is not allowed.

## search
You are provided with function signatures within <tools></tools> XML tags:
<tool_call>
{"type":"function", "function":
{
  "name": "image_search",
  "description": "Retrieves top 10 images and descriptions from Google's image search using the original image. Should only be used once.",
},
{
  "name": "search",
  "description": "Performs batched web searches: supply an array 'query'; the tool retrieves the top 10 results for each query in one call.",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "Search query to find relevant information."
      }
    },
    "required": [
      "query"
    ]
    }
}
</tool_call>

Example for calling search:
Return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": "image_search"}
</tool_call>
<tool_call>
{"name": "search", "arguments": {"query": "Does Cloudflare analyze submitted data to block attacks"}}
</tool_call>

Note:
1. You MUST engage in many interactions, delving deeply into the topic to explore all possible aspects until a satisfactory answer is found.
2. Before presenting a Final Answer, you will **cross-check** and **validate the information** you've gathered to confirm its accuracy and reliability.
3. You will carefully analyze each information source to ensure that all data is current, relevant, and from credible origins.
4. Please note that you can **only** call search once at a time. If you need to perform multiple searches, please do so in the next round.
5. You can **only** conduct image search once.
"""

USER_PROMPT = """<image>\n{question}\n\nYou must put your answer inside <answer> </answer> tags, i.e., <answer> answer here </answer>. Please reason step by step. Use Python code to process the image if necessary. You can conduct search to seek the Internet. Format strictly as <think> </think> <code> </code> (if code is neededs) or <think> </think> <tool_call> </tool_call> (if function call is neededs) or <think> <think> <answer> </answer>."""


RETURN_PROMPT = """Code execution result:
stdout:
```
{stdout}
```

stderr:
```
{stderr}
```

{image}
"""

RETURN_SEARCH_PROMPT = """<tool_response>
{search_result}
</tool_response>
"""

INITIALIZATION_CODE_TEMPLATE = """
from PIL import Image
import base64
from io import BytesIO

_img_base64 = "{base64_image}"
image_1 = Image.open(BytesIO(base64.b64decode(_img_base64)))
"""

if __name__ == "__main__":

    openai_api_key = "EMPTY"
    openai_api_base = "http://xxx:xxx/v1"
    code_sandbox_url = "http://127.0.0.1:8000/jupyter_sandbox"

    response = requests.get(f"{openai_api_base}/models")
    models = response.json()
    model_name = models['data'][0]['id']

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    image_path = 'path/to/your/image.jpg'
    question = "What is in this image?"

    session_id = generate_session_id()

    base64_image = encode_image_to_base64(image_path)


    chat_message = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                {"type": "text", "text": USER_PROMPT.format(question=question)},
            ],
        }
    ]

    status = 'success'
    try_count = 0
    turn_idx = 0
    response_message = ""

    init_code_string = INITIALIZATION_CODE_TEMPLATE.format(base64_image=base64_image)
    init_ret = request_jupyter_execution(init_code_string)

    try:
        while '</answer>' not in response_message:
            if '</answer>' in response_message and '<answer>' in response_message:
                break

            if try_count > 10:
                # status = 'error'
                break
            
            params = {
                "model": model_name,  # 或VLLM服务中的任何模型名称
                "messages": chat_message,
                "temperature": 0.0,
                "max_tokens": 20480,
                "stop": ["<|im_end|>\n".strip()],
            }
            # print('print message:   ',print_messages)
            response = client.chat.completions.create(**params)
            response_message = response.choices[0].message.content

            # add assistant response to messages
            chat_message.append(
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": response_message},
                    ],
                }
            )

            todo_action = 'code'
            if '<code>' in response_message:
                todo_action = 'code'
            elif '<tool_call>' in response_message:
                todo_action = 'action'

            if todo_action == 'code':
                code_string = extract_python_code(response_message)
                if not code_string:
                    status = 'error'
                    break
                
                code_string = fix_python_indentation(code_string)
                exec_ret = request_jupyter_execution(code_string, code_sandbox_url, session_id)

                image_list = exec_ret.get('images', [])
                image_list = image_list[:10]
                code_result_string = RETURN_PROMPT.format(
                    stdout=exec_ret.get('stdout', ''),
                    stderr=exec_ret.get('stderr', ''),
                    image="Images:\n" + "<image>" * len(image_list) if len(image_list) > 0 else "",
                ).strip()

                add_message = {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": code_result_string},
                    ],
                }
                if image_list:
                    for idx, img in enumerate(image_list):
                        img_base64 = base64.b64encode(img.tobytes()).decode('utf-8')
                        add_message["content"].append(
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
                        )
                chat_message.append(add_message)

            elif todo_action=='action':
                action = extract_action(response_message)

                if not action:
                    status = 'error'
                    break

                try:
                    tool_call = json.loads(action.strip())  # 或使用 literal_eval
                except Exception as e:
                    error_msg = f"Invalid tool call format: {action.strip()}. Error: {e}"
                    add_message = {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": error_msg},
                        ],
                    }
                    chat_message.append(add_message)
                    continue
                
                try:
                    tool_name = tool_call["name"]
                    args = tool_call.get("arguments", None)

                    # error process
                    if tool_name not in ['search', 'image_search']:
                        error_msg = f"Invalid tool call name: {action.strip()}."
                        add_message = {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": error_msg},
                            ],
                        }
                        chat_message.append(add_message)
                        continue
                    if tool_name == 'image_search' and args is not None:
                        error_msg = f"Invalid tool call parameters for image search: {action.strip()}."
                        add_message = {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": error_msg},
                            ],
                        }
                        chat_message.append(add_message)
                        continue

                    exec_ret = request_search(tool_name, args)
                    if not exec_ret or exec_ret['status'] != 'success':
                        error_msg = "Search error"
                        add_message = {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": error_msg},
                            ],
                        }
                        chat_message.append(add_message)
                        continue
                    
                    image_list = exec_ret.get('images', [])
                    image_list = image_list[:10]
                    search_result_string = RETURN_SEARCH_PROMPT.format(
                        search_result=exec_ret.get('result', ''),
                    ).strip()

                    if len(image_list) == 0:
                        add_message = {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": code_result_string},
                            ],
                        }
                    else:
                        for idx, img in enumerate(image_list):
                            img_base64 = base64.b64encode(img.tobytes()).decode('utf-8')
                            add_message = {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": search_result_string},
                                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
                                ],
                            }
                    chat_message.append(add_message)

                except Exception as e:
                    error_msg = f"Tool call execution error: {str(e)}"
                    add_message = {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": error_msg},
                        ],
                    }
                    chat_message.append(add_message)
            else:
                error_msg = f"Format Error."
                add_message = {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": error_msg},
                    ],
                }
                chat_message.append(add_message)

            turn_idx += 1
        try_count += 1
    except Exception as e:
        status = 'error'
        try_count += 1


    print("Final response message:")
    print(response_message)








