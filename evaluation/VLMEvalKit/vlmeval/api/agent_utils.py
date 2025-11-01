import time
import autopep8
import textwrap
import random
import requests
import base64
from math import ceil
import re
import os
import math
from io import BytesIO
from PIL import Image
import numpy as np

code_sandbox_url = os.environ.get("CODE_SERVER_BASE", "http://localhost:80/jupyter_sandbox")

ORIGIN_PATH = "/root/LMUData/images"
RESIZE_PATH = "/root/LMUData_Resize/images"
os.makedirs(RESIZE_PATH, exist_ok=True)

init_code = """
from PIL import Image
image_1 = Image.open({img_path!r})
"""

IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28

def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor

def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor

def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor

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

def replace_image_path_resize(image_path: str) -> str:
    resize_image_path = image_path.replace(ORIGIN_PATH, RESIZE_PATH)
    if os.path.exists(resize_image_path):
        return resize_image_path
    else:
        os.makedirs(os.path.dirname(resize_image_path), exist_ok=True)
        img = Image.open(image_path)
        img = maybe_resize_image_v2(img)
        img.save(resize_image_path)
        return resize_image_path

def check_white_image(pil_image):
    img_array = np.array(pil_image)
    if np.all(img_array >= 250):
        return True
    return False

def generate_session_id():
    salted_str = str(int(time.time())) + str(random.randint(10000, 99999))
    salted_hash_str = str(hex(hash(salted_str.encode('utf-8')))).split('0x')[-1]
    return salted_hash_str

def encode_pil_image_to_base64(pil_image):
    if pil_image.mode == 'RGBA':
        pil_image = pil_image.convert('RGB')
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str

def maybe_resize_image(image):
    """
    Qwen-VL raises an error for images with height or width less than 32 pixels.
    """
    height, width = image.height, image.width
    if min(height, width) >= 32:
        return image

    ratio = 32 / min(height, width)
    new_height = ceil(height * ratio)
    new_width = ceil(width * ratio)
    new_image = image.resize((new_width, new_height), Image.LANCZOS)
    return new_image

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

def decode_base64_to_pil_image(base64_string: str) -> Image.Image:
    image_data = base64.b64decode(base64_string)
    image = Image.open(BytesIO(image_data))
    return image

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

def execute_code(session_id, code_part, code_sandbox_url=code_sandbox_url, code_timeout=60.0, request_timeout=120.0):
    try:
        code_part = fix_python_indentation(code_part)
        resjson = requests.post(
            code_sandbox_url,
            json={
                "session_id": session_id,
                "code": code_part,
                "timeout": code_timeout
            },
            timeout=request_timeout
        ).json()
        result_dict = resjson['output']
    except Exception as err:
        print(f' [ERROR code] Request to Jupyter sandbox failed: {err}')
        return None
    
    image_pil_list = []
    if isinstance(result_dict, str):
        print ('[ERROR code] ', result_dict, code_part)

    image_base64_list = result_dict.get("images", [])
    for idx, img in enumerate(image_base64_list):
        try:
            img_pil = decode_base64_to_pil_image(img)
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

def return_init_code(img_path):
    return init_code.format(img_path=img_path)
    