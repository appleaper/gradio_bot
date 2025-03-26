# 大致功能

1. 大模型本地聊天。支持qwen，deepseek本地模型加载
2. 支持多模态模型本地聊天，目前仅支持MiniCPM-2.6
3. 支持用户上传文件后，解析为知识，作为rag内容。
4. 多个知识可以组成知识库
5. 目前可以解析PDF、规定格式的CSV、JPG、png、JPEG、docx、MP4、mp3格式的文件。
6. 提供ocr识别
7. 提供语言转文字
8. rag库支持lancedb,milvus,mysql,es数据库存储

# 重点功能流程解释

**文字聊天**：选定模型参数+知识库名字+提问的内容 --> 从数据库中搜索与问题相关的内容作为rag-->送入大模型-->大模型返回

**MP4解析**：MP4文件上传-->将MP4文件切片（60s）-->语音转文字-->解析的文字存入数据库中-->完成处理！

# 如何安装

## **本地安装**

1. python环境（作者使用3.10）

2. cuda环境

3. 安装requirment.txt中的依赖包

   ```
   pip install -r requirment.txt
   ```

   torch的一定要装pytorch官方的，镜像源的不行

4. 准备好相关的大模型文件

   去huggingface下载，下载好之后放在项目目录中的model文件夹

   ```
   模型名称：
   BAAIbge-m3
   deepseek_r1_distill_qwen_1_5b
   FireRedASR-AED-L
   openbmbMiniCPM-V-2_6-int4
   Qwen25_05B_Instruct
   stepfun-aiGOT-OCR2_0
   ```

5. 启动ui_demo1.py文件

## docker安装

1. 运行dockerfile

   ```
   docker build -t gradio_bot .
   ```

2. 运行容器

   ```
   docker run -d -p 7680:7680 --gpus all -v 项目路径:/app --name gradio_bot 镜像id python ./ui_demo1.py
   
   例如：
   docker run -d -p 7680:7680 --gpus all -v C:\use\code\RapidOcr_small:/app --name gradio_bot 3294f5cc2c01 python ./ui_demo1.py
   ```

   