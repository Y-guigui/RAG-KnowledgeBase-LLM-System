import streamlit as  st
from rag import RagService
# 1. 引入获取 Streamlit 上下文的工具
from streamlit.runtime.scriptrunner import get_script_run_ctx

# 标题
st.title("智能客服")
# 分隔符
st.divider()

# 避免性能压力，session_state 存入对象
if "message" not in st.session_state:
    st.session_state["message"]=[{"role":"assistant","content":"你好，有什么可以帮助你？"}]

if "rag" not in st.session_state:
    st.session_state["rag"]= RagService()

# 动态获取当前浏览器的会话 ID（如果获取不到则降级回 fallback）
ctx = get_script_run_ctx()
current_session_id = ctx.session_id if ctx else "default_fallback_session"

# 动态构建配置
session_config = {
    "configurable": {
        "session_id": current_session_id,
    }
}

#循环 输出历史信息，原本只记录但页面不显示
for message in st.session_state["message"]:
    st.chat_message(message["role"]).write(message["content"])

# 在页面最下方提供用户输入栏
prompt= st.chat_input()


if prompt :
    # 在页面输出用户的提问
    st.chat_message("user").write(prompt)
    st.session_state["message"].append({"role":"user","content":prompt})

    ai_res_list= []
    with st.spinner("AI 思考中......."):
        # 调用 RAG服务

        # 流式输出
        res_stream = st.session_state["rag"].chain.stream({"input": prompt}, session_config)

        def capture(generator, cache_list):
            for chunk in generator:
                cache_list.append(chunk)
                yield chunk
        st.chat_message("assistant").write_stream(capture(res_stream,ai_res_list))
        st.session_state["message"].append({"role": "assistant", "content": "".join(ai_res_list)})