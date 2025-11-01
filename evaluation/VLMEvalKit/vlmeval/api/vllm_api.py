import os
import re
from typing import List, Dict, Any, Tuple
from ..smp import *
from .base import BaseAPI
from io import BytesIO
import threading
import requests
import json
import time
import traceback  # 添加traceback模块用于详细错误信息
import copy
import random

from .prompt import THINKING_CONFIG, AGENT_CONFIG
from .agent_utils import generate_session_id, execute_code, encode_pil_image_to_base64, check_white_image, replace_image_path_resize, maybe_resize_image_v2, return_init_code
from .agent_utils import code_sandbox_url as code_sandbox_url_original
from .agent_utils import ORIGIN_PATH, RESIZE_PATH

def get_image_fmt_from_image_path(image_path):
    _, ext = os.path.splitext(image_path)
    if ext.lower() == '.jpg' or ext.lower() == '.jpeg':
        return 'jpeg'
    elif ext.lower() == '.png':
        return 'png'
    else:
        return 'jpeg'

class VLLMAPIWrapper(BaseAPI):

    is_api: bool = True

    def __init__(self,
                 model: str = None,
                 retry: int = 5,
                 agent_try: int = 10,
                 code_sandbox_url: str = None,
                 wait: int = 5,
                 key: str = None,
                 verbose: bool = False,
                 system_prompt: str = None,
                 temperature: float = 0,
                 timeout: int = 600,
                 api_base: str = None,
                 max_tokens: int = 2048,
                 img_size: int = 512,
                 **kwargs):
        self.cur_idx = 0
        self.fail_msg = 'Failed to obtain answer via API. '
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.inference_mode = 'non-think'  # Default to non-think mode

        self.agent_try = agent_try
        self.code_sandbox_url = code_sandbox_url if code_sandbox_url else code_sandbox_url_original

        env_key = os.environ.get('VLLM_API_KEY', '')
        if key is None:
            key = env_key

        self.key = key
        self.timeout = timeout
        self.cur_idx = 0
        self.cur_idx_lock = threading.Lock()

        super().__init__(wait=wait, retry=retry, system_prompt=system_prompt, verbose=verbose, **kwargs)
        if ',' in api_base:
            api_base = api_base.split(',')
        if isinstance(api_base, str):
            api_base = [api_base]
        self.api_base = api_base
        
        if model is None:
            model = self._get_actual_model_name()
        self.model = model
        assert self.model, f"You must set model name."
        self.logger.info(f'Using API Base: {self.api_base}; Model: {self.model}')

    def set_inference_mode(self, mode: str):
        """Set inference mode: 'think', 'non-think', or 'agent'"""
        if mode not in ['think', 'non-think', 'agent']:
            raise ValueError(f"Invalid inference mode: {mode}. Must be one of ['think', 'non-think', 'agent']")
        self.inference_mode = mode
        self.logger.info(f'Inference mode set to: {mode}')

    def generate(self, message, **kwargs1):
        if self.check_content(message) == 'listdict':
            message = self.preprocess_message_with_role(message)

        assert self.check_content(message) in ['str', 'dict', 'liststr', 'listdict'], f'Invalid input type: {message}'
        message = self.preproc_content(message)
        assert message is not None and self.check_content(message) == 'listdict'
        for item in message:
            assert item['type'] in self.allowed_types, f'Invalid input type: {item["type"]}'

        # merge kwargs
        import copy as cp
        kwargs = cp.deepcopy(self.default_kwargs)
        kwargs.update(kwargs1)

        answer = None
        # a very small random delay [0s - 0.5s]
        import random as rd
        import time
        T = rd.random() * 0.5
        time.sleep(T)
        for i in range(self.retry):
            try:
                ret_code, result, log = self.generate_inner(message, **kwargs)

                # 处理新的result格式
                if isinstance(result, dict):
                    answer = result.get('response', result.get('raw_response', ''))
                    if 'thinking' in result and self.inference_mode == 'think':
                        final_result = {
                            'response': answer,  
                            'thinking': result['thinking'],
                            'raw_response': result['raw_response']
                        }
                        if ret_code == 0 and self.fail_msg not in answer and answer != '':
                            return final_result
                    elif self.inference_mode == 'agent':
                        final_result = {
                            'response': answer,  
                            'raw_response': result['raw_response'],
                            'image_path_list': result.get('image_path_list', [])
                        }
                        if ret_code == 0 and self.fail_msg not in answer and answer != '':
                            return final_result
                    else:
                        if ret_code == 0 and self.fail_msg not in answer and answer != '':
                            return answer
                else:
                    answer = result
                    if ret_code == 0 and self.fail_msg not in answer and answer != '':
                        return answer
                        
                if self.verbose:
                    if not isinstance(log, str):
                        try:
                            log = log.text
                        except Exception as e:
                            self.logger.warning(f'Failed to parse {log} as an http response: {str(e)}. ')
                    self.logger.info(f'RetCode: {ret_code}\nAnswer: {answer}\nLog: {log}')
            except Exception as err:
                if self.verbose:
                    self.logger.error(f'An error occured during try {i}: ')
                    self.logger.error(f'{type(err)}: {err}')
                    
            # delay before each retry
            T = rd.random() * self.wait * 2
            time.sleep(T)

        return self.fail_msg if answer in ['', None] else answer

    def _get_actual_model_name(self):
        try:
            api_base = self.api_base[0] if isinstance(self.api_base, list) else self.api_base
            model_url = f"{api_base}/v1/models"
            print(f"DEBUG: Models url: {model_url}")
            response = requests.get(model_url)
            print(f"DEBUG: Models response: {response.json()}")
            if response.status_code == 200:
                models_data = response.json()
                if 'data' in models_data and len(models_data['data']) > 0:
                    return models_data['data'][0]['id']
        except Exception as e:
            print(f"DEBUG: Error getting model name: {e}")
        return self.model

    def prepare_itlist(self, inputs):
        assert np.all([isinstance(x, dict) for x in inputs])
        
        has_images = np.sum([x['type'] == 'image' for x in inputs])
        image_path_list = []
        
        if has_images:
            content_list = []
            for msg in inputs:
                if msg['type'] == 'text':
                    content_list.append(dict(type='text', text=msg['value']))
                elif msg['type'] == 'image':
                    from PIL import Image
                    fmt=get_image_fmt_from_image_path(msg['value'])
                    image_path_list.append(msg['value'])
                    img = Image.open(msg['value'])
                    b64 = encode_image_to_base64(img, target_size=-1, fmt=fmt)
                    img_struct = dict(url=f'data:image/{fmt};base64,{b64}')
                    content_list.append(dict(type='image_url', image_url=img_struct))
        else:
            assert all([x['type'] == 'text' for x in inputs])
            text = '\n'.join([x['value'] for x in inputs])
            content_list = [dict(type='text', text=text)]
        
        if self.inference_mode == 'agent':
            return content_list, image_path_list
        else:
            return content_list

    def prepare_inputs_thinking(self, inputs):
        input_msgs = []
        input_msgs.append(dict(role='system', content=THINKING_CONFIG['system_prompt']))
        
        assert isinstance(inputs, list) and isinstance(inputs[0], dict)
        assert np.all(['type' in x for x in inputs]) or np.all(['role' in x for x in inputs]), inputs
        
        def add_thinking_prompt(content_list):
            if isinstance(content_list, list) and len(content_list) > 0:
                for content_item in content_list:
                    if content_item.get('type') == 'text':
                        original_text = content_item['text']
                        content_item['text'] = THINKING_CONFIG['user_template'].format(original_text=original_text)
                        break
                else:
                    content_list.append({'type': 'text', 'text': THINKING_CONFIG['fallback_text']})
            else:
                content_list = [{'type': 'text', 'text': THINKING_CONFIG['fallback_text']}]
            return content_list
        
        if 'role' in inputs[0]:
            assert inputs[-1]['role'] == 'user', inputs[-1]
            for item in inputs:
                content = self.prepare_itlist(item['content'])
                if item == inputs[-1] and item['role'] == 'user':
                    content = add_thinking_prompt(content)
                input_msgs.append(dict(role=item['role'], content=content))
        else:
            content = self.prepare_itlist(inputs)
            content = add_thinking_prompt(content)
            input_msgs.append(dict(role='user', content=content))

        return input_msgs
    
    def prepare_itlist_agent(self, inputs):
        assert np.all([isinstance(x, dict) for x in inputs])
        
        has_images = np.sum([x['type'] == 'image' for x in inputs])
        image_path_list = []
        
        if has_images:
            content_list = []
            for msg in inputs:
                if msg['type'] == 'text':
                    content_list.append(dict(type='text', text=msg['value']))
                elif msg['type'] == 'image':
                    from PIL import Image
                    fmt=get_image_fmt_from_image_path(msg['value'])
                    img_path = msg['value']
                    resize_img_path = replace_image_path_resize(img_path)
                    image_path_list.append(resize_img_path)
                    img = Image.open(resize_img_path)
                    b64 = encode_image_to_base64(img, target_size=-1, fmt=fmt)
                    img_struct = dict(url=f'data:image/{fmt};base64,{b64}')
                    content_list.append(dict(type='image_url', image_url=img_struct))
        else:
            assert all([x['type'] == 'text' for x in inputs])
            text = '\n'.join([x['value'] for x in inputs])
            content_list = [dict(type='text', text=text)]
        
        if self.inference_mode == 'agent':
            return content_list, image_path_list
        else:
            return content_list
    
    def prepare_inputs_agent(self, inputs):
        input_msgs = []
        input_msgs.append(dict(role='system', content=AGENT_CONFIG['system_prompt']))
        
        assert isinstance(inputs, list) and isinstance(inputs[0], dict)
        assert np.all(['type' in x for x in inputs]) or np.all(['role' in x for x in inputs]), inputs
        
        def add_agent_prompt(content_list):
            if isinstance(content_list, list) and len(content_list) > 0:
                for content_item in content_list:
                    if content_item.get('type') == 'text':
                        original_text = content_item['text']
                        content_item['text'] = AGENT_CONFIG['user_template'].format(question=original_text)
                        break
                else:
                    content_list.append({'type': 'text', 'text': AGENT_CONFIG['fallback_text']})
            else:
                content_list = [{'type': 'text', 'text': AGENT_CONFIG['fallback_text']}]
            return content_list
        
        if 'role' in inputs[0]:
            assert inputs[-1]['role'] == 'user', inputs[-1]
            for item in inputs:
                content, image_path_list = self.prepare_itlist_agent(item['content'])
                if item == inputs[-1] and item['role'] == 'user':
                    content = add_agent_prompt(content)
                input_msgs.append(dict(role=item['role'], content=content))
        else:
            content, image_path_list = self.prepare_itlist_agent(inputs)
            content = add_agent_prompt(content)
            input_msgs.append(dict(role='user', content=content))

        return input_msgs, image_path_list

    def prepare_inputs_direct(self, inputs):
        input_msgs = []
        direct_system_prompt = 'You are a helpful assistant. Answer the user\'s question directly and concisely.'

        system_content = self.system_prompt if self.system_prompt is not None else direct_system_prompt
        input_msgs.append(dict(role='system', content=system_content))
        
        assert isinstance(inputs, list) and isinstance(inputs[0], dict)
        assert np.all(['type' in x for x in inputs]) or np.all(['role' in x for x in inputs]), inputs
        
        if 'role' in inputs[0]:
            assert inputs[-1]['role'] == 'user', inputs[-1]
            for item in inputs:
                input_msgs.append(dict(role=item['role'], content=self.prepare_itlist(item['content'])))
        else:
            input_msgs.append(dict(role='user', content=self.prepare_itlist(inputs)))

        return input_msgs

    def prepare_inputs(self, inputs):
        if self.inference_mode == 'think':
            return self.prepare_inputs_thinking(inputs)
        elif self.inference_mode == 'agent':
            return self.prepare_inputs_agent(inputs)
        else:  # non-think mode
            return self.prepare_inputs_direct(inputs)

    def _next_api_base(self):
        if isinstance(self.api_base, str):
            return f"{self.api_base}/v1/chat/completions"
        with self.cur_idx_lock:
            _api_base = f"{self.api_base[self.cur_idx]}/v1/chat/completions"
            self.cur_idx = (self.cur_idx + 1) % len(self.api_base)
        return _api_base

    def prepare_payload(self, input_msgs, temperature=0., max_tokens=2048, **kwargs):
        headers = {'Content-Type': 'application/json'}
        payload = dict(
            model=self.model,
            messages=input_msgs,
            temperature=temperature,
            **kwargs)

        payload['max_tokens'] = max_tokens

        return headers, payload
    
    def request_with_retry(self, headers, payload):
        try_times = 0
        while try_times < 3:
            try:
                response = requests.post(
                    self._next_api_base(),
                    headers=headers, data=json.dumps(payload), timeout=self.timeout * 1.1)

                ret_code = 0
                ret_code = response.status_code
                ret_code = 0 if (200 <= int(ret_code) < 300) else ret_code
                answer = self.fail_msg
                
                if ret_code == 0: 
                    try:
                        resp_struct = json.loads(response.text)
                        answer = resp_struct['choices'][0]['message']['content'].strip()

                        thinking_content = None
                        final_answer = answer
                        
                        if self.inference_mode == 'think':
                            think_match = re.search(THINKING_CONFIG['think_regex'], answer, re.DOTALL)
                            if think_match:
                                thinking_content = think_match.group(1).strip()
                            
                            answer_match = re.search(THINKING_CONFIG['answer_regex'], answer, re.DOTALL)
                            if answer_match:
                                final_answer = answer_match.group(1).strip()

                            if not thinking_content and not answer_match and THINKING_CONFIG['retry_on_missing_tags']:
                                if self.verbose:
                                    self.logger.warning(f"No thinking tags found in response, will retry")
                                try_times += 1
                                if try_times < THINKING_CONFIG['max_tag_retries']:
                                    time.sleep(THINKING_CONFIG['tag_retry_delay'])
                                    continue
                                else:
                                    final_answer = answer
                        
                        result = {
                            'response': final_answer,  # 最终答案
                            'raw_response': answer,    # 原始完整回答
                        }
                        
                        if thinking_content:
                            result['thinking'] = thinking_content
                        
                        return ret_code, result, response
                    except Exception as err:
                        if self.verbose:
                            self.logger.error(f'JSON parsing error: {type(err)}: {err}')
                            self.logger.error(f'Response text: {response.text}')
                        answer = f"JSON parsing error: {err}"
                else:
                    if self.verbose:
                        self.logger.error(f'HTTP error {ret_code}: {response.text}')
                    answer = f"HTTP error {ret_code}: {response.text}"

                    if self.inference_mode == 'think':
                        answer = {
                            'response': f"HTTP error {ret_code}: {response.text}",
                            'raw_response': f"HTTP error {ret_code}: {response.text}"
                        }
                
                try_times += 1
                if try_times < 3:
                    self.logger.warning(f"Retry {try_times}/3 after error")
                    time.sleep(1)  # 等待1秒后重试
                    
            except requests.exceptions.RequestException as err:
                if self.verbose:
                    self.logger.error(f'Request exception: {type(err)}: {err}')
                answer = f"Request exception: {err}"

                if self.inference_mode == 'think':
                    answer = {
                        'response': f"Request exception: {err}",
                        'raw_response': f"Request exception: {err}"
                    }
                
                try_times += 1
                if try_times < 3:
                    self.logger.warning(f"Retry {try_times}/3 after exception")
                    time.sleep(1)
        
        print(f"DEBUG: All {try_times} attempts failed")
        if self.verbose:
            self.logger.error(f"All {try_times} attempts failed")

        if self.inference_mode == 'think' and not isinstance(answer, dict):
            answer = {
                'response': answer if answer else self.fail_msg,
                'raw_response': answer if answer else self.fail_msg
            }
        return ret_code, answer, response
    
    def prepare_inputs_agent_turn(self, agent_response, execute_outputs):
        return_image_list = execute_outputs.get('images', [])

        return_content = AGENT_CONFIG['return_template'].format(stdout=execute_outputs['stdout'], stderr=execute_outputs['stderr'])

        return_msgs =[
            {
                "role": "assistant",
                "content": agent_response,
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": return_content},
                ],
            }
        ]
        for r_img in return_image_list:
            if check_white_image(r_img):
                continue
            r_img = maybe_resize_image_v2(r_img)
            r_img = encode_pil_image_to_base64(r_img)
            return_msgs[1]['content'].append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{r_img}"}})

        return return_msgs

    def generate_inner_agent(self, input_msgs, **kwargs) -> str:
        def del_image_base64(input_msgs):
            todo_input_msgs = copy.deepcopy(input_msgs)
            for msg in todo_input_msgs:
                if msg['role'] == 'system' or msg['role'] == 'assistant':
                    continue
                if 'content' in msg:
                    for content in msg['content']:
                        if content['type'] == 'image_url':
                            content['image_url']['url'] = "data:image/jpeg;base64"
            return todo_input_msgs

        temperature = kwargs.pop('temperature', self.temperature)
        max_tokens = kwargs.pop('max_tokens', self.max_tokens)
        input_msgs, image_path_list = input_msgs

        agent_try_times = 0
        agent_status = 'Success'
        agent_response = ''
        agent_total_response = del_image_base64(input_msgs)

        session_id = generate_session_id()

        init_code_part = return_init_code(image_path_list[0].replace(ORIGIN_PATH, RESIZE_PATH))
        execute_outputs = execute_code(session_id, init_code_part)


        # try:
        img_idx = 0
        while AGENT_CONFIG['answer_prefix'] not in agent_response:
            if AGENT_CONFIG['answer_prefix'] in agent_response and AGENT_CONFIG['answer_suffix'] in agent_response:
                break
            
            if agent_try_times > self.agent_try:
                agent_status = 'Out of tries'
                break
            
            headers, payload = self.prepare_payload(input_msgs, temperature=temperature, max_tokens=max_tokens, **kwargs)
            payload['stop'] = AGENT_CONFIG['stop_token']

                
            ret_code, agent_response, response = self.request_with_retry(headers, payload)
            agent_response = agent_response['raw_response']
            if AGENT_CONFIG['code_prefix'] in agent_response:
                if AGENT_CONFIG['code_suffix'] not in agent_response:
                    agent_response += AGENT_CONFIG['code_suffix']  # Ensure code block is closed
                
                code_part = agent_response
                if AGENT_CONFIG['code_prefix'] in code_part:
                    code_part = code_part.split(AGENT_CONFIG['code_prefix'])[1].strip()
                if AGENT_CONFIG['code_suffix'] in code_part:
                    code_part = code_part.split(AGENT_CONFIG['code_suffix'])[0].strip()
                if AGENT_CONFIG['python_prefix'] in code_part:
                    code_part = code_part.split(AGENT_CONFIG['python_prefix'])[1].strip()
                if AGENT_CONFIG['python_suffix'] in code_part:
                    code_part = code_part.split(AGENT_CONFIG['python_suffix'])[0].strip()

                execute_outputs = execute_code(session_id, code_part)

                if not execute_outputs or execute_outputs['status'] != 'success':
                    raise ValueError(f"Code execution failed: {execute_outputs.get('error', 'Unknown error')}")

                return_msgs = self.prepare_inputs_agent_turn(agent_response, execute_outputs)

                input_msgs.extend(return_msgs)
                agent_total_response.extend(del_image_base64(return_msgs))

            agent_try_times += 1

        answer = agent_response
        answer_match = re.search(AGENT_CONFIG['answer_regex'], answer, re.DOTALL)
        if answer_match:
            final_answer = answer_match.group(1).strip()
        else:
            final_answer = answer.strip()
        
        agent_total_response.append({
            "role": "assistant",
            "content": answer,
        })
        
        answer = {
            'response': final_answer,  # 最终答案
            'raw_response': agent_total_response,    # 原始完整回答
            'image_path_list': image_path_list,  # 返回的图片路径列表
        }
        
        return ret_code, answer, agent_total_response    

    def generate_inner(self, inputs, **kwargs) -> str:
        input_msgs = self.prepare_inputs(inputs)

        if self.inference_mode == 'think' or self.inference_mode == 'non-think':
            temperature = kwargs.pop('temperature', self.temperature)
            max_tokens = kwargs.pop('max_tokens', self.max_tokens)

            headers, payload = self.prepare_payload(input_msgs, temperature=temperature, max_tokens=max_tokens, **kwargs)
            ret_code, answer, response = self.request_with_retry(headers, payload)
        elif self.inference_mode == 'agent':
            ret_code, answer, response = self.generate_inner_agent(input_msgs, **kwargs)
        else:
            raise ValueError(f"Invalid inference mode: {self.inference_mode}. Must be one of ['think', 'non-think', 'agent']")
        
        return ret_code, answer, response


class VLLMAPI(VLLMAPIWrapper):
    def generate(self, message, dataset=None):
        return super(VLLMAPI, self).generate(message)



