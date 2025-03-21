# SmolDocling-256M-WebUI

本项目是基于 [ds4sd/SmolDocling-256M-preview](https://huggingface.co/ds4sd/SmolDocling-256M-preview) 模型的文档转换 WebUI 应用。这是一个轻量级的视觉语言模型，专门用于文档识别和转换任务。

## 🌟 功能特点

- **支持多种设备运行（CPU/CUDA/MPS），自动检测并使用最适合的设备**
- 📄 文档格式转换：将图片中的文档转换为结构化的 Docling 格式
- 📊 表格识别：将图片中的表格转换为 OTSL 格式
- 💻 代码识别：识别并转换图片中的代码内容
- ➗ 公式转换：将图片中的数学公式转换为 LaTeX 格式
- 📈 图表转换：将图表转换为 OTSL 格式
- 🔍 文本定位：支持精确定位和识别特定区域的文本内容
- 📑 章节提取：自动提取文档中的章节标题

## 🚀 如何运行

1. 克隆项目并安装依赖：
```bash
git clone https://github.com/Airmomo/SmolDocling-256M-WebUI.git
cd SmolDocling-256M-WebUI
pip install -r requirements.txt
```

2. 配置环境变量：
   - 复制 `.env.example` 文件为 `.env`
   - 设置 `MODEL_PATH` 为模型路径（默认使用 HuggingFace 上的模型）

3. 运行应用：
```bash
python app.py
```

WebUI 服务默认端口为`7860`

## 资源

- [SmolDocling-256M 模型](https://huggingface.co/ds4sd/SmolDocling-256M-preview)
