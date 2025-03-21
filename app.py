import gradio as gr
from transformers import AutoProcessor, AutoModelForVision2Seq, TextIteratorStreamer
from transformers.image_utils import load_image
from threading import Thread
import re
import time
import torch
import spaces
import re
import ast
import html
import random
import os
from dotenv import load_dotenv

from PIL import Image, ImageOps

from docling_core.types.doc import DoclingDocument
from docling_core.types.doc.document import DocTagsDocument

# 加载环境变量
load_dotenv()

def add_random_padding(image, min_percent=0.1, max_percent=0.10):
    image = image.convert("RGB")

    width, height = image.size

    pad_w_percent = random.uniform(min_percent, max_percent)
    pad_h_percent = random.uniform(min_percent, max_percent)

    pad_w = int(width * pad_w_percent)
    pad_h = int(height * pad_h_percent)

    corner_pixel = image.getpixel((0, 0))  # Top-left corner
    padded_image = ImageOps.expand(image, border=(pad_w, pad_h, pad_w, pad_h), fill=corner_pixel)

    return padded_image

def normalize_values(text, target_max=500):
    def normalize_list(values):
        max_value = max(values) if values else 1
        return [round((v / max_value) * target_max) for v in values]

    def process_match(match):
        num_list = ast.literal_eval(match.group(0))
        normalized = normalize_list(num_list)
        return "".join([f"<loc_{num}>" for num in normalized])

    pattern = r"\[([\d\.\s,]+)\]"
    normalized_text = re.sub(pattern, process_match, text)
    return normalized_text

# 自动检测设备
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
print(f"MPS available: {torch.backends.mps.is_available()}")

model_path = os.getenv("MODEL_PATH")
if not model_path:
    raise ValueError("MODEL_PATH 环境变量未设置！请在 .env 文件中设置 MODEL_PATH。")

processor = AutoProcessor.from_pretrained(model_path)
model = AutoModelForVision2Seq.from_pretrained(
    model_path, 
    torch_dtype=torch.float16 if device != "cpu" else torch.float32,
    #_attn_implementation="flash_attention_2"
).to(device)

@spaces.GPU
def model_inference(
    input_dict, history
): 
    text = input_dict["text"]
    print(input_dict["files"])
    if len(input_dict["files"]) > 1:
        if "OTSL" in text or "code" in text:
            images = [add_random_padding(load_image(image)) for image in input_dict["files"]]
        else:
            images = [load_image(image) for image in input_dict["files"]]

    elif len(input_dict["files"]) == 1:
        if "OTSL" in text or "code" in text:
            images = [add_random_padding(load_image(input_dict["files"][0]))]
        else:
            images = [load_image(input_dict["files"][0])]   

    else:
        images = []

    if text == "" and not images:
        gr.Error("请输入查询内容和可选的图片。")

    if text == "" and images:
        gr.Error("请输入与图片相关的文本查询。")

    if "OCR at text at" in text or "Identify element" in text or "formula" in text:
        text = normalize_values(text, target_max=500)

    resulting_messages = [
                {
                    "role": "user",
                    "content": [{"type": "image"} for _ in range(len(images))] + [
                        {"type": "text", "text": text}
                    ]
                }
            ]
    prompt = processor.apply_chat_template(resulting_messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[images], return_tensors="pt").to(device)

    generation_args = {
        "input_ids": inputs.input_ids,
        "pixel_values": inputs.pixel_values,
        "attention_mask": inputs.attention_mask,
        "num_return_sequences": 1,
        "max_new_tokens": 8192,
    }

    streamer = TextIteratorStreamer(processor, skip_prompt=True, skip_special_tokens=False)
    generation_args = dict(inputs, streamer=streamer, max_new_tokens=8192)

    thread = Thread(target=model.generate, kwargs=generation_args)
    thread.start()

    yield "..."
    buffer = ""
    full_output = ""

    for new_text in streamer:
        full_output += new_text
        buffer += html.escape(new_text)
        yield buffer

    cleaned_output = full_output.replace("<end_of_utterance>", "").strip()

    if cleaned_output:
        doctag_output = cleaned_output
        yield cleaned_output

    if any(tag in doctag_output for tag in ["<doctag>", "<otsl>", "<code>", "<chart>", "<formula>"]):
        doc = DoclingDocument(name="Document")
        if "<chart>" in doctag_output:
            doctag_output = doctag_output.replace("<chart>", "<otsl>").replace("</chart>", "</otsl>")
            doctag_output = re.sub(r'(<loc_500>)(?!.*<loc_500>)<[^>]+>', r'\1', doctag_output)

        doctags_doc = DocTagsDocument.from_doctags_and_image_pairs([doctag_output], images)
        doc.load_from_doctags(doctags_doc)
        yield f"**MD Output:**\n\n{doc.export_to_markdown()}"

examples=[[{"text": "将此页面转换为docling格式。", "files": ["example_images/2d0fbcc50e88065a040a537b717620e964fb4453314b71d83f3ed3425addcef6.png"]}],
          [{"text": "将此表格转换为OTSL格式。", "files": ["example_images/image-2.jpg"]}],
          [{"text": "将代码转换为文本。", "files": ["example_images/7666.jpg"]}],
          [{"text": "将公式转换为latex格式。", "files": ["example_images/2433.jpg"]}],
          [{"text": "将图表转换为OTSL格式。", "files": ["example_images/06236926002285.png"]}],
          [{"text": "识别位置 [47, 531, 167, 565] 的文本", "files": ["example_images/s2w_example.png"]}],
          [{"text": "提取页面上的所有章节标题。", "files": ["example_images/paper_3.png"]}],
          [{"text": "识别位置 [123, 413, 1059, 1061] 的元素", "files": ["example_images/redhat.png"]}],
          [{"text": "将此页面转换为docling格式。", "files": ["example_images/gazette_de_france.jpg"]}],
          ]

demo = gr.ChatInterface(
    fn=model_inference, 
    title="SmolDocling-256M: 超紧凑的文档转换视觉语言模型 💫", 
    description="体验 [ds4sd/SmolDocling-256M-preview](https://huggingface.co/ds4sd/SmolDocling-256M-preview) 演示。上传图片并输入文本指令，或者试试下面的示例。本演示不使用聊天历史记录，每次对话都是新的会话。",
    examples=examples,
    textbox=gr.MultimodalTextbox(label="查询输入", file_types=["image"], file_count="multiple"), 
    stop_btn="停止生成", 
    multimodal=True,
    cache_examples=False
)

demo.launch(debug=True)