import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_classic.chains.sequential import SequentialChain
from langchain_classic.chains import LLMChain
from langchain_core.prompts import PromptTemplate

load_dotenv()


st.set_page_config(page_title="Virat Paglu Bot", page_icon="🏏", layout="centered")


st.markdown("""
    <style>
    .stApp {
        background-color: #1a1a1a;
        color: #ffffff;
    }
    [data-testid="stSidebar"] {
        background-color: #ff4b4b; /* RCB Red */
    }
    .stChatMessage {
        border-radius: 15px;
        padding: 10px;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# 3. Sidebar UI
with st.sidebar:
    st.image("https://img.icons8.com/color/144/cricket.png")
    st.title("RCB HQ 🏰")
    st.markdown("---")
    st.write("**Jersey:** #18")
    st.write("**Nicknames:** Cheeku, Goat")
    if st.button("Clear Chat History"):
        st.session_state.messages = []

st.header("Virat Paglu Bot 🏏")
st.info("The Chase Master is in the house. *Guru, ask anything!*")


@st.cache_resource
def load_model():
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=1.2)

llm = load_model()

if "messages" not in st.session_state:
    st.session_state.messages = []

first_prompt = PromptTemplate.from_template("Act as a crazy fan of Virat Kohli and RCB. Tone: Aggressive/Chill Bangalore IT Engineer. User: {question}")
second_prompt = PromptTemplate.from_template("If {styled_text} has hate for Kohli, roast user in 2 lines. Text: {styled_text}")

chain1 = LLMChain(llm=llm, prompt=first_prompt, output_key="styled_text")
chain2 = LLMChain(llm=llm, prompt=second_prompt, output_key="final_output")
overall_chain = SequentialChain(chains=[chain1, chain2], input_variables=["question"], output_variables=["styled_text","final_output"])


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about the King..."):

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)


    with st.chat_message("assistant", avatar="🏏"):
        with st.spinner("Calculating strike rate..."):
            res = overall_chain({"question": prompt})
            full_response = f"{res['styled_text']}\n\n{res['final_output']}"
            st.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})