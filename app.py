import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from peft import PeftModel
import gradio as gr
from threading import Thread
import spaces
import os

# 从环境变量中获取 Hugging Face 模型信息
HF_TOKEN = os.environ.get("HF_TOKEN", None)
BASE_MODEL_ID = "Qwen/Qwen2.5-Coder-14B-Instruct"  # 替换为基础模型
LORA_MODEL_PATH = "QLWD/test-7b"  # 替换为 LoRA 模型仓库路径

# 定义界面标题和描述
TITLE = "<h1><center>漏洞检测 微调模型测试</center></h1>"

DESCRIPTION = f"""
<h3>模型: <a href="https://huggingface.co/{LORA_MODEL_PATH}">漏洞检测 微调模型</a></h3>
<center>
<p>测试基础模型 + LoRA 补丁的生成效果。</p>
</center>
"""

CSS = """
.duplicate-button {
margin: auto !important;
color: white !important;
background: black !important;
border-radius: 100vh !important;
}
h3 {
text-align: center;
}
"""

# 加载基础模型和 LoRA 微调权重
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, torch_dtype=torch.float16, device_map="auto", use_auth_token=HF_TOKEN)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, use_auth_token=HF_TOKEN)

# 加载 LoRA 微调权重
model = PeftModel.from_pretrained(base_model, LORA_MODEL_PATH, use_auth_token=HF_TOKEN)
model = model.to("cuda" if torch.cuda.is_available() else "cpu")

# 定义推理函数
@spaces.GPU(duration=50)
def stream_chat(message: str, history: list, temperature: float, max_new_tokens: int, top_p: float, top_k: int, penalty: float):
    conversation = []
    
    # 添加系统提示，定义模型的角色
    conversation.append({"role": "system", "content": '''你是一位代码审计和漏洞修复助手，请仔细分析下面提供的代码，扫描并输出所有存在的漏洞和潜在的风险。每个漏洞或风险之间用分隔符 "--------" 隔开，报告内容左对齐。

从高危到低危的顺序来列出漏洞和风险，每个漏洞或风险的格式如下：
- **类型**：明确描述漏洞的类型（如SQL注入、命令注入、反序列化漏洞等），或潜在的风险类型（如资源泄露、边界条件问题等）。
- **风险等级**：根据漏洞或风险的严重性进行评级（如高危、中危、低危）。
- **漏洞/风险描述**：详细解释漏洞的技术细节和成因，或描述潜在的风险。
- **影响**：说明该漏洞或风险可能对系统、数据或用户造成的具体影响。
- **修复建议**：提供修复该漏洞或风险的具体步骤或建议（不是给出修复代码，而是修复的实现方法）。
- **漏洞所在的代码段**：给出代码中存在漏洞的具体位置和代码段（如适用）。
- **修复的代码段**：给出修复漏洞的替换代码段，以便开发者使用（如适用）。

请确保扫描并**输出所有**漏洞和风险，请确保扫描并**输出所有**漏洞和风险，包括但不限于：命令注入、SQL注入、文件操作不安全、资源泄露、异常处理缺失等。

分隔符 "--------" 用于每个漏洞或风险之间。
'''})

    # 将历史对话内容添加到会话中
    for prompt, answer in history:
        conversation.extend([{"role": "user", "content": prompt}, {"role": "漏洞zhushou", "content": answer}])
    
    # 添加当前用户的输入到对话中
    conversation.append({"role": "user", "content": message})

    # 使用自定义对话模板生成 input_ids
    input_ids = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_ids, return_tensors="pt").to("cuda")

    streamer = TextIteratorStreamer(tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True)
    
    # 设置生成参数
    generate_kwargs = dict(
        inputs,
        streamer=streamer,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=penalty,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        eos_token_id=[151645, 151643],
    )

    # 启动生成线程
    thread = Thread(target=model.generate, kwargs=generate_kwargs)
    thread.start()

    buffer = ""
    for new_text in streamer:
        buffer += new_text
        yield buffer

# 定义 Gradio 界面
chatbot = gr.Chatbot(height=450)

with gr.Blocks(css=CSS) as demo:
    gr.HTML(TITLE)
    gr.HTML(DESCRIPTION)
    gr.DuplicateButton(value="Duplicate Space for private use", elem_classes="duplicate-button")
    
    gr.ChatInterface(
        fn=stream_chat,
        chatbot=chatbot,
        fill_height=True,
        additional_inputs_accordion=gr.Accordion(label="⚙️ 参数设置", open=False, render=False),
        additional_inputs=[
            gr.Slider(minimum=0, maximum=1, step=0.1, value=0.8, label="Temperature", render=False),
            gr.Slider(minimum=128, maximum=4096, step=1, value=1024, label="Max new tokens", render=False),
            gr.Slider(minimum=0.0, maximum=1.0, step=0.1, value=0.8, label="top_p", render=False),
            gr.Slider(minimum=1, maximum=20, step=1, value=20, label="top_k", render=False),
            gr.Slider(minimum=0.0, maximum=2.0, step=0.1, value=1.0, label="Repetition penalty", render=False),
        ],
        cache_examples=False,
    )

# 启动 Gradio 应用
if __name__ == "__main__":
    demo.launch()
