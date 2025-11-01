import os

def build_prompt_judge(question, prediction, groundtruth):
    with open(os.path.join(os.path.dirname(__file__), 'judge_prompt/verify.md'), 'r') as f:
        tmpl = f.read()
    tmpl = tmpl + f"【用户问题】:{question}\n【参考答案】：{groundtruth}\n【模型回答】：{prediction}"
    return tmpl

def judge_result(judge_response):
    if '<最终结果>' in judge_response:
        judge_response = judge_response.split('<最终结果>')[-1].strip().split('<\最终结果>')[0].strip()
    if 'boxed' in judge_response:
        judge_response = judge_response.split('boxed')[-1].strip().split('}')[0].strip()
    
    if 'yes' in judge_response.lower():
        judge_response = True
    else:
        judge_response = False
    return judge_response


