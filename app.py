import os

import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

VECTOR_DIR = "./vectorstore/faiss_index"
DATA_DIR = "./data"

st.set_page_config(
    page_title="Motion Tap RAG 챗봇",
    page_icon="🤖"
)

st.title("🤖 Motion Tap AI 어시스턴트")
st.write("모션탭 매뉴얼과 Q&A 자료를 기반으로 답변하는 RAG 챗봇입니다.")

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

vectorstore = FAISS.load_local(
    VECTOR_DIR,
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 4}
)

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

example_questions = [
    "모션탭 최대 연결 개수는 몇 개인가요?",
    "모션탭 배터리 사용 시간은 얼마나 되나요?",
    "모션탭 앱 설치와 회원가입은 어떻게 하나요?",
    "터치 센서가 잘 안 될 때 어떻게 해결하나요?",
    "레이저 센서 오류가 발생하면 어떻게 대처하나요?",
    "플레이어 추가와 관리는 어떻게 하나요?",
    "시퀀스 동작 방식은 무엇인가요?",
    "문서에 없는 질문에 대해서는 어떻게 답변하나요?"
]

st.subheader("질문 입력")

question = st.text_input(
    "직접 질문을 입력하세요",
    placeholder="예: 모션탭 최대 연결 개수는 몇 개인가요?"
)

selected_question = st.selectbox(
    "또는 테스트 질문을 선택하세요",
    ["선택 안 함"] + example_questions
)

if selected_question != "선택 안 함":
    question = selected_question
    st.info(f"선택한 질문: {question}")

ask_button = st.button("질문하기")

if ask_button:
    if not question.strip():
        st.warning("질문을 입력하거나 테스트 질문을 선택해주세요.")
    else:
        with st.spinner("관련 문서를 검색하고 답변을 생성하는 중입니다..."):
            docs = retriever.invoke(question)

            context = "\n\n".join([
                f"[출처: {doc.metadata.get('source', '알 수 없음')} / 페이지: {doc.metadata.get('page', '-')} / 섹션: {doc.metadata.get('section', '-')} ]\n{doc.page_content}"
                for doc in docs
            ])

            prompt = f"""
너는 Motion Tap 제품 전문 상담 챗봇이다.
아래 제공된 문서 내용만 참고해서 답변해라.
문서에 없는 내용은 절대 추측하지 말고,
"제공된 자료에서 해당 정보를 찾을 수 없습니다."라고 답해라.

[문서 내용]
{context}

[사용자 질문]
{question}

[답변 조건]
- 한국어로 답변
- 초보자도 이해하기 쉽게 설명
- 필요한 경우 단계별로 설명
- 답변 마지막에 참고한 자료명을 간단히 표시
"""

            response = llm.invoke(prompt)

        st.subheader("답변")
        st.write(response.content)

        source_names = []
        for doc in docs:
            source = doc.metadata.get("source", "알 수 없음")
            if source not in source_names:
                source_names.append(source)

        st.subheader("답변 근거")

        if source_names:
            st.success("검색된 자료를 기반으로 답변을 생성했습니다.")
        else:
            st.warning("관련 근거 자료를 찾지 못했습니다.")

        st.markdown("### 참고 자료")

        shown_sources = set()

        for doc in docs:
            source = doc.metadata.get("source", "알 수 없음")

            if source in shown_sources:
                continue

            shown_sources.add(source)

            page = doc.metadata.get("page", "-")
            section = doc.metadata.get("section", "-")

            source_path = os.path.join(DATA_DIR, source)

            col1, col2 = st.columns([4, 1])

            with col1:
                st.markdown(f"📄 **{source}**")
                st.caption(f"페이지: {page} | 섹션: {section}")

            with col2:
                if os.path.exists(source_path):
                    with open(source_path, "rb") as file:
                        st.download_button(
                            label="열기",
                            data=file.read(),
                            file_name=source,
                            mime="application/octet-stream",
                            key=f"download_{source}"
            )