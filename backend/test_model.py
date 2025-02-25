import json
import chardet
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_gigachat.chat_models import GigaChat
from langchain.chains import RetrievalQA
from langchain_core.messages import HumanMessage, AIMessage
import gigachat.context

def detect_encoding(file_path):
    """
    Определяет кодировку файла, читая его в бинарном режиме.
    """
    with open(file_path, "rb") as f:
        raw_data = f.read()
    result = chardet.detect(raw_data)
    return result["encoding"]

def load_json_file(file_path):
    """
    Загружает JSON из файла. Сначала пытается использовать обнаруженную кодировку,
    при ошибке пробует UTF‑8 с errors="replace".
    """
    encoding = detect_encoding(file_path)
    try:
        with open(file_path, "r", encoding=encoding, errors="replace") as f:
            data = json.load(f)
    except UnicodeDecodeError:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            data = json.load(f)
    return data

class LangChainQueryProcessor:
    """
    Упрощённый класс без агентов и Flask.
    Читает JSON, формирует FAISS-векторное хранилище и позволяет задавать вопросы через RetrievalQA.
    """
    def __init__(self, json_file):
        # Устанавливаем заголовок для GigaChat (при необходимости)
        headers = {
            "X-Session-ID": "8324244b-7133-4d30-a328-31d8466e5503",
        }
        gigachat.context.session_id_cvar.set(headers.get("X-Session-ID"))

        # Токен для GigaChat (пример)
        GIGACHAT_CREDENTIALS = (
            "ZWExMmU1NmUtZjhhNS00M2UxLWJlOGEtNGNhMjIwZWU2Zjc3OmIxMDIyNjZhLThjNmItNDM2NS1hNmI0LTY2ZTkyNTRiZGI1Yw=="
        )

        # Создаём модель GigaChat
        self.model = GigaChat(
            credentials=GIGACHAT_CREDENTIALS,
            scope="GIGACHAT_API_PERS",
            model="GigaChat-Pro",  # Или "GigaChat-Max"
            verify_ssl_certs=False,
        )

        # Создаём векторное хранилище (FAISS) на основе JSON
        self.vectorstore = self._initialize_vectorstore(json_file)

        # Список для хранения истории (вопросы/ответы)
        self.history = []

        # Файл для записи всех диалогов
        self.log_file = "conversation_log.json"

    def _initialize_vectorstore(self, json_file):
        """
        Считывает JSON, формирует список документов, а затем создаёт FAISS-хранилище.
        Если json_file является списком путей, обрабатываются все файлы.
        """
        docs = []
        # Если передан список файлов, обрабатываем каждый, иначе делаем список из одного файла
        files = json_file if isinstance(json_file, list) else [json_file]

        for file_path in files:
            data = load_json_file(file_path)

            # Формируем список документов (Document), где page_content — сочетание полей
            for item in data:
                page_content = (
                    f"Название: {item.get('Название', '')}\n"
                    f"Описание: {item.get('Описание', '')}\n"
                    f"Цена: {item.get('Цена', '')}"
                )
                docs.append(Document(page_content=page_content))

        # Разбиваем большие тексты на чанки
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splitted_docs = text_splitter.split_documents(docs)

        # Создаём эмбеддинги и векторное хранилище
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(splitted_docs, embedding_model)
        return vectorstore

    def process_query(self, question: str) -> str:
        """
        Задаёт вопрос RetrievalQA, используя модель GigaChat и FAISS‑retriever.
        Выводит в консоль релевантные фрагменты для отладки,
        сохраняет результат в self.history и conversation_log.json.
        """
        # Добавляем сообщение пользователя в историю
        self.history.append(HumanMessage(content=question))

        # Получаем retriever из векторного хранилища
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})

        # Отладка: получаем релевантные документы для запроса
        relevant_docs = retriever.get_relevant_documents(question)
        print(f"\n=== Релевантные фрагменты для запроса: '{question}' ===")
        for idx, doc in enumerate(relevant_docs, 1):
            print(f"Фрагмент {idx}: {doc.page_content}")
        print("===============================================\n")

        # Создаём цепочку RetrievalQA
        qa = RetrievalQA.from_chain_type(llm=self.model, retriever=retriever)

        # Получаем ответ от модели
        response = qa.run(question)

        # Добавляем ответ в историю
        self.history.append(AIMessage(content=response))

        # Сохраняем диалог (вопрос+ответ) в JSON‑файл
        self.save_conversation(question, response)

        return response

    def save_conversation(self, question, answer):
        """
        Сохраняет каждый вопрос и ответ в общий JSON‑файл (conversation_log.json).
        """
        conversation_data = {
            "question": question,
            "answer": answer
        }

        # Считываем существующий JSON (если есть)
        try:
            with open(self.log_file, "r", encoding="utf-8", errors="replace") as f:
                conversation_log = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            conversation_log = []

        # Добавляем текущую пару (вопрос-ответ)
        conversation_log.append(conversation_data)

        # Перезаписываем файл
        with open(self.log_file, "w", encoding="utf-8") as f:
            json.dump(conversation_log, f, indent=4, ensure_ascii=False)

    def process_questions_from_json(self, questions_json_file: str):
        """
        Читает список вопросов из JSON, обрабатывает их и сохраняет результаты в answers.json.
        """
        questions_data = load_json_file(questions_json_file)

        responses = []
        for item in questions_data:
            question = item.get('question')
            if question:
                answer = self.process_query(question)
                responses.append({"question": question, "answer": answer})

        # Записываем все ответы в отдельный файл answers.json
        with open('answers.json', 'w', encoding="utf-8") as f:
            json.dump(responses, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    # Путь к вашему JSON с данными (аналог tea_data.json)
    # Если у вас несколько файлов, передайте их как список
    JSON_PATH = r"C:\Users\Daniil\Projects\my-bot\tea_data.json"
    
    # Создаём объект для поиска
    file_search = LangChainQueryProcessor(JSON_PATH)

    # Предположим, что здесь лежит JSON с вопросами для тестирования
    QUESTIONS_JSON = r"C:\Users\Daniil\Projects\my-bot\evaluation_dataset.json"

    # Обрабатываем все вопросы из JSON и записываем ответы в answers.json
    file_search.process_questions_from_json(QUESTIONS_JSON)
