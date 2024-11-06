import os
import pandas as pd
import streamlit as st

from langchain.schema import Document
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings


# Google APIキーが環境変数に設定されていない場合は、デフォルトのキーを設定
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = "AIzaSyDhWd3UYYv7B9m5wXBOenzsYK_o9av_eMQ"


@st.cache_data
def load_data(csv_file_path):
    """
    CSVファイルを読み込み、Pandas DataFrameとして返します。

    Parameters:
    csv_file_path (str): 読み込むCSVファイルのパス

    Returns:
    pd.DataFrame: CSVファイルから読み込んだデータフレーム
    """
    df = pd.read_csv(csv_file_path)
    return df


@st.cache_data
def split_text(df):
    """
    'text_tokenized'列のテキストを分割して、チャンク化したドキュメントのリストを返します。

    Parameters:
    df (pd.DataFrame): 'text_tokenized'列を含むデータフレーム

    Returns:
    list: 分割されたドキュメントのリスト
    """
    # 'text_tokenized' 列からテキストを取得
    text_data = df["text"].dropna().tolist()  # NaNを除外

    # RecursiveCharacterTextSplitterを使ってテキストを分割
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=514,  # 各チャンクの最大サイズ
        chunk_overlap=20,  # チャンク間の重複サイズ
    )

    # チャンクを作成
    documents = []
    for text in text_data:
        if text.strip():  # 空のテキストをスキップ
            documents.append(Document(page_content=text))

    # テキストを分割する
    docs = text_splitter.split_documents(documents)

    return docs

#
def init_messages():
    """
    チャット履歴とメモリを初期化し、セッション状態に保存します。
    また、チャット履歴を消去するボタンを表示します。
    """
    # チャット履歴がセッション状態に保存されていない場合は初期化
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "system", "content": "あなたは日本語で正確に回答する役立つアシスタントです。"}]
        st.session_state.costs = []  # コストの初期化
        st.session_state.memory = ConversationBufferMemory(
            memory_key="chat_history",  # 会話履歴を保存するキー
            input_key='question',       # 質問のキー
            output_key='answer',        # 回答のキー
            return_messages=True,       # メッセージを返すオプション
        )
    else:
        # チャット履歴を消去するボタンを表示
        if st.button("チャットを消去", key="clear"):
            st.session_state.messages.clear()  # メッセージのクリア
            st.session_state.memory.clear()    # メモリのクリア


@st.cache_resource
def create_qa_chain_history(_docs, _memory):
    """
    質問応答チェーンを作成し、キャッシュを使用して最適化します。

    Parameters:
    _docs (list): 分割されたテキストドキュメントのリスト
    _memory (ConversationBufferMemory): 会話の履歴を保持するメモリ

    Returns:
    ConversationalRetrievalChain: 質問応答チェーンオブジェクト
    """
    # Embeddingsとデータベースのキャッシュ
    embeddings = HuggingFaceEmbeddings(model_name="pkshatech/GLuCoSE-base-ja")
    
    # .chroma_dbが存在する場合は再利用、存在しない場合は新規作成
    if os.path.exists(".chroma_db"):
        db = Chroma(persist_directory=".chroma_db", embedding_function=embeddings)
    else:
        db = Chroma.from_documents(_docs, embeddings, persist_directory=".chroma_db")
        db.persist()  # データベースの永続化
    retriever = db.as_retriever()  # 検索機能を設定

    # LLMのセットアップとQAチェーンのキャッシュ
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", convert_system_message_to_human=True, temperature=0)
    
    # 質問応答チェーンの作成
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",  # チェーンタイプの設定
        retriever=retriever,
        memory=_memory  # 会話履歴をメモリに保持
    )

    return qa_chain

# チャット履歴を含む質問応答の処理を行い、ユーザーとアシスタントのやりとりを表示する関数
def ask_chat_history(docs):
    """
    チャット履歴を含む質問応答の処理を行い、ユーザーとアシスタントのやりとりを表示します。

    Parameters:
    docs (list): 分割されたテキストドキュメントのリスト
    """
    memory = st.session_state.memory  # メモリを取得
    qa_chain = create_qa_chain_history(docs, memory)  # QAチェーンを作成

    # ユーザーからの入力を受け付け、質問応答を処理
    if user_input := st.chat_input("質問を入力してください "):
        with st.spinner("Gemini が入力しています ..."):
            result = qa_chain({"question": f"{user_input} 日本語で答えてください。"}, return_only_outputs=True)
            assistant_content = result["answer"]

        # チャット履歴にユーザーとアシスタントのメッセージを追加
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.messages.append({"role": "assistant", "content": assistant_content})

        # チャット履歴を表示
        for message in st.session_state.messages:
            if message["role"] != "system":
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

# メインアプリケーションのフローを制御する関数
def main():
    """
    アプリケーションのメイン関数で、全体のフローを制御します。
    CSVファイルを読み込み、テキストを分割し、チャットインターフェースを表示します。
    """
    st.title("RAG System")

    # CSVファイルを読み込み
    csv_file_path = "yahoo_news_articles_preprocessed.csv"  # CSVファイルのパス
    df = load_data(csv_file_path)

    # テキストを分割
    docs = split_text(df)

    # チャット履歴とメモリを初期化
    init_messages()

    # 質問入力→回答
    ask_chat_history(docs)

# アプリケーションの実行
if __name__ == "__main__":
    main()
