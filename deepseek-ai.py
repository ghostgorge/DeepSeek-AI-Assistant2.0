import streamlit as st
import requests
import logging
import pdfplumber
import chardet
from typing import List, Dict

# 配置
API_URL = "https://api.deepseek.com/v1/chat/completions"
LOG_FILENAME = "deepseek_dashboard.log"
MAX_CONTEXT_MESSAGES = 8
MAX_FILE_CONTENT = 1000

# 日志配置
def configure_logging():
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    file_handler = logging.FileHandler(LOG_FILENAME)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logging.basicConfig(
        level=logging.INFO,
        handlers=[file_handler, console_handler]
    )

configure_logging()
logger = logging.getLogger("DeepSeekDashboard")

# 会话管理类
class ChatMemory:
    def __init__(self, max_messages: int = MAX_CONTEXT_MESSAGES):
        self.max_messages = max_messages
        self.initialize_session()

    def initialize_session(self):
        if "messages" not in st.session_state:
            st.session_state["messages"] = []
            st.session_state["full_history"] = []

    def add_message(self, role: str, content: str):
        st.session_state.messages.append({"role": role, "content": content})
        st.session_state.full_history.append({"role": role, "content": content})
        while len(st.session_state.full_history) > self.max_messages:
            removed = st.session_state.full_history.pop(0)
            logger.debug(f"Trimming message: {removed['content'][:50]}...")

    def get_context(self, system_prompt: str) -> List[Dict]:
        return [{"role": "system", "content": system_prompt}] + st.session_state.full_history[-self.max_messages:]

    def clear_memory(self):
        st.session_state.messages = []
        st.session_state.full_history = []

# 处理上传文件
def process_uploaded_files(files) -> str:
    processed_content = []
    for file in files:
        try:
            if file.type == "application/pdf":
                # 使用 pdfplumber 处理 PDF 文件
                with pdfplumber.open(file) as pdf:
                    text = ""
                    for page in pdf.pages:
                        text += page.extract_text() or ""
                processed_content.append(f"PDF_CONTENT:{file.name}: {text[:MAX_FILE_CONTENT]}...")
            else:
                # 自动检测文本文件编码
                raw_data = file.read()
                detected_encoding = chardet.detect(raw_data)
                encoding = detected_encoding.get("encoding", "utf-8-sig")  # 默认使用 utf-8-sig
                content = raw_data.decode(encoding, errors="replace")
                processed_content.append(f"FILE_CONTENT:{file.name}: {content[:MAX_FILE_CONTENT]}...")
        except Exception as e:
            logger.error(f"Error processing {file.name}: {str(e)}")
            st.warning(f"处理文件 {file.name} 时出错: {str(e)}")
    return "\n".join(processed_content)

# DeepSeek API 请求
def query_deepseek(prompt: str, system_prompt: str, memory: ChatMemory, model: str = "deepseek-chat",
                   temperature: float = 0.7) -> Dict:
    headers = {
        "Authorization": f"Bearer {st.session_state['DEESEEK_API_KEY']}",
        "Content-Type": "application/json",
    }

    try:
        payload = {
            "model": model,
            "messages": memory.get_context(system_prompt) + [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }

        logger.info(f"发送 {len(payload['messages'])} 条消息到 DeepSeek API...")

        with st.spinner("正在处理中，请稍等..."):
            response = requests.post(API_URL, headers=headers, json=payload)
            response.raise_for_status()

            if response.status_code == 200:
                response_data = response.json()
                assistant_response = response_data.get("choices", [{}])[0].get("message", {}).get("content", "")

                memory.add_message("user", prompt)
                memory.add_message("assistant", assistant_response)

                return response_data

        logger.error(f"API 返回了非 200 状态: {response.status_code}")
        return None

    except requests.exceptions.RequestException as e:
        logger.error(f"API 请求失败: {str(e)}", exc_info=True)
        st.error(f"API 错误: {str(e)}")
        return None

# 登录界面逻辑
def login_page():
    st.set_page_config(page_title="DeepSeek API 登录", page_icon="🔑")
    st.title("欢迎使用 DeepSeek AI 助手")

    st.markdown("请输入您的 DeepSeek API 密钥以开始使用。")

    # 输入框让用户填写 API 密钥
    api_key = st.text_input("API 密钥", type="password")

    # 用户点击登录按钮
    if st.button("进入对话界面"):
        if api_key:
            # 将密钥存储到 session 状态
            st.session_state['DEESEEK_API_KEY'] = api_key
            st.session_state['logged_in'] = True  # 设置已登录标志
            st.session_state['login_successful'] = True  # 登录成功的标志
            # 页面刷新逻辑
            st.session_state['page'] = 'chat'  # 更新页面状态为 'chat' 页面
        else:
            st.error("请输入有效的 API 密钥。")

# 主界面逻辑
def main_interface():
    if 'DEESEEK_API_KEY' not in st.session_state:
        login_page()  # 如果没有 API 密钥，则显示登录界面
        return

    st.set_page_config(page_title="DeepSeek AI 助手", layout="wide", page_icon="🧠")
    st.title("DeepSeek AI 助手")

    with st.sidebar:
        st.title("控制面板")
        model_choice = st.selectbox("选择 AI 模型", ["deepseek-chat", "deepseek-reasoner"], index=0)
        temperature = st.slider("创造力级别", 0.0, 1.0, 0.7, 0.05)
        system_prompt = st.text_area(
            "系统角色",
            value="你是一个专家级 AI 助手，请提供详细且准确的回答。",
            height=150
        )
        uploaded_files = st.file_uploader(
            "上传知识文件",
            accept_multiple_files=True,
            type=None
        )

    memory = ChatMemory()

    # 始终显示对话框（聊天历史）
    for msg in st.session_state.get("messages", []):
        role = "用户" if msg["role"] == "user" else "助手"
        content = msg["content"]

        if "FILE_CONTENT" in content or "PDF_CONTENT" in content:
            parts = content.split(":", 2)
            st.write(f"{role}: {parts[0]}")
            with st.expander("附加文件"):
                st.text(parts[-1])
        else:
            st.write(f"{role}: {content}")

    # 用户输入
    user_input = st.text_input("请输入你的问题...")  # 使用 text_input 代替 chat_input
    if user_input:
        file_context = process_uploaded_files(uploaded_files) if uploaded_files else ""
        full_prompt = f"{user_input}\n{file_context}" if file_context else user_input

        # 显示用户输入
        st.write(f"用户: {user_input}")
        if file_context:
            with st.expander("附加文件"):
                st.text(file_context)

        # 获取并显示助手的回复
        try:
            response = query_deepseek(
                prompt=full_prompt,
                system_prompt=system_prompt,
                memory=memory,
                model=model_choice,
                temperature=temperature
            )

            if response:
                content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
                if content:
                    st.markdown(f"助手: {content}")
                else:
                    st.error("未能从 API 获取有效的回复")

        except Exception as e:
            st.error(f"通信错误: {str(e)}")
            logger.exception("主界面出现意外错误")

    # 添加页面底部的版权信息
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("© 2025 高治中. 版权所有.", unsafe_allow_html=True)

# 启动入口
if __name__ == "__main__":
    if 'page' not in st.session_state:
        st.session_state['page'] = 'login'  # 默认加载登录页

    if st.session_state['page'] == 'login':
        login_page()  # 显示登录界面
    else:
        main_interface()  # 显示主界面
