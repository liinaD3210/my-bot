import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import json

# LangChain импорт
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document

# Модель GigaChat (кастомный класс)
from langchain_gigachat.chat_models import GigaChat

# Prompt, инструменты и агенты
from langchain.prompts import PromptTemplate
from langchain.agents import create_react_agent, Tool, AgentExecutor
from langchain.tools import BaseTool
from langchain.memory import ConversationBufferMemory

# (Для примера, если используете gigachat.context)
import gigachat.context
from typing import Any

app = Flask(__name__)
CORS(app)

# --- Ваш кастомный Prompt ---
REACT_PROMPT_TEMPLATE = """
Ты — консультант кофейни, у тебя есть доступ к инструменту:
{tools}

Имена инструментов: {tool_names}.

Каждый твой ответ должен строго соблюдать формат ReAct:
1) Action: retriever_search
2) Action Input: {input}
3) Observation: (этот блок с результатами инструмента вставляет система — НЕ выдумывай его сам!)
4) Final Answer: ... (ответ пользователю на основе Observation)

Важные правила:
- Ты обязан всегда совершить ровно один вызов Action: retriever_search.
- Не пиши Observation сам. Дождись, пока система подставит результаты поиска, и только потом на следующей строке вынеси итог в Final Answer.
- Не вставляй текст Observation дословно в Final Answer. Сформулируй ответ своими словами.
- Если Observation не нашлось (пустое), скажи «К сожалению, я не нашёл подходящего ответа».
- Строка "Final Answer: …" всегда идёт последней, и ответ на этом заканчивается.

Вопрос пользователя: {input}

{agent_scratchpad}
"""

class RetrieverTool(BaseTool):
    """Кастомный инструмент для поиска по векторному хранилищу."""
    retriever: Any
    name: str = "retriever_search"
    description: str = (
        "Используй этот инструмент, когда в истории сообщений нет ответа на вопрос,"
        "и нужно найти дополнительную информацию в JSON базе."
    )

    def _run(self, query: str) -> str:
        docs = self.retriever.get_relevant_documents(query)
        results = []
        for i, doc in enumerate(docs, start=1):
            snippet = doc.page_content.replace("\n", " ")
            results.append(f"[doc {i}] {snippet}")
        return "\n".join(results)

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("Async not implemented")


class LangChainQueryProcessor:
    """
    Класс, инкапсулирующий логику чтения JSON, создания векторного индекса,
    и общения через ReAct-агента (с учётом истории в памяти).
    """
    def __init__(self, json_file):
        # Установим заголовок для GigaChat (при необходимости)
        headers = {
            "X-Session-ID": "8324244b-7133-4d30-a328-31d8466e5503",
        }
        gigachat.context.session_id_cvar.set(headers.get("X-Session-ID"))

        # Токен для GigaChat (пример)
        GIGACHAT_CREDENTIALS = (
            "ZWExMmU1NmUtZjhhNS00M2UxLWJlOGEtNGNhMjIwZWU2Zjc3OmIxMDIyNjZhLThjNmItNDM2NS1hNmI0LTY2ZTkyNTRiZGI1Yw=="
        )

        # Создаём саму модель GigaChat
        self.model = GigaChat(
            credentials=GIGACHAT_CREDENTIALS,
            scope="GIGACHAT_API_PERS",
            model="GigaChat-Pro",
            verify_ssl_certs=False,
        )

        # Инициализируем векторное хранилище (FAISS) по JSON
        self.vectorstore = self._initialize_vectorstore(json_file)

        # Создаём retriever
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})

        # Создаём инструмент для поиска
        self.retriever_tool = RetrieverTool(retriever=self.retriever)
        self.tools = [
            Tool(
                name=self.retriever_tool.name,
                func=self.retriever_tool.run,
                description=self.retriever_tool.description
            )
        ]

        # Создаём память, чтобы агент мог учитывать предыдущие сообщения
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        # Формируем PromptTemplate на основе REACT_PROMPT_TEMPLATE
        self.custom_prompt = PromptTemplate(
            input_variables=["input", "tools", "tool_names", "agent_scratchpad"],
            template=REACT_PROMPT_TEMPLATE
        )

        # Создаём ReAct-агента (без verbose)
        self.agent = create_react_agent(
            llm=self.model,
            tools=self.tools,
            prompt=self.custom_prompt
        )

        # Оборачиваем в AgentExecutor
        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=self.agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True
        )

    def _initialize_vectorstore(self, json_file):
        """Читает JSON, сплитит и создаёт векторное хранилище (FAISS)."""
        # Загружаем JSON
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Формируем список документов
        docs = []
        for item in data:
            # Составим строку из всех ключевых полей (Название, Описание, Цена)
            page_content = (
                f"Название: {item.get('Название', '')}\n"
                f"Описание: {item.get('Описание', '')}\n"
                f"Цена: {item.get('Цена', '')}"
            )
            docs.append(Document(page_content=page_content))

        # При желании можно применять TextSplitter, если тексты большие.
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splitted_docs = text_splitter.split_documents(docs)

        # Создаём векторное хранилище
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(splitted_docs, embedding_model)
        return vectorstore

    def process_query_with_agent(self, user_input: str) -> str:
        """Вызываем нашего собранного агента (с учётом истории в памяти)."""
        result = self.agent_executor({"input": user_input})
        response = result["output"]
        return response

# --- Flask-приложение ---


JSON_PATH = r"C:\Users\Daniil\Projects\my-bot\tea_data.json"  # <-- Вместо PDF

file_search = LangChainQueryProcessor(JSON_PATH)


app = Flask(__name__)
CORS(app)

@app.route('/bot', methods=['POST'])
def bot():
    data = request.json
    user_input = data.get('message', '').strip()
    if not user_input:
        return jsonify({'response': 'Пожалуйста, отправьте текст!'}), 400

    search_result = file_search.process_query_with_agent(user_input)
    return jsonify({'response': search_result})

if __name__ == '__main__':
    app.run(debug=True)
