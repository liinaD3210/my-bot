import json
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_gigachat.chat_models import GigaChat
from langchain.chains import RetrievalQA
from langchain_core.messages import HumanMessage, AIMessage
import gigachat.context

class LangChainQueryProcessor:
    def __init__(self, pdf_files):
        self.pdf_files = pdf_files

        headers = {
            "X-Session-ID": "8324244b-7133-4d30-a328-31d8466e5503",
        }
        gigachat.context.session_id_cvar.set(headers.get("X-Session-ID"))

        GIGACHAT_CREDENTIALS = 'ZWExMmU1NmUtZjhhNS00M2UxLWJlOGEtNGNhMjIwZWU2Zjc3OmIxMDIyNjZhLThjNmItNDM2NS1hNmI0LTY2ZTkyNTRiZGI1Yw=='

        self.model = GigaChat(
            credentials=GIGACHAT_CREDENTIALS,
            scope="GIGACHAT_API_PERS",
            model="GigaChat-Max",
            verify_ssl_certs=False,
        )

        self.vectorstore = self._initialize_vectorstore()

        # История сообщений будет храниться в обычном списке
        self.history = []

        # Файл для записи вопросов и ответов
        self.log_file = "conversation_log.json"

    def _initialize_vectorstore(self):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=200
        )

        all_docs = []
        for pdf_file in self.pdf_files:
            loader = UnstructuredPDFLoader(pdf_file)
            splitted_data = loader.load_and_split(text_splitter)
            all_docs.extend(splitted_data)

        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(all_docs, embedding_model)
        return vectorstore

    def process_query(self, question: str) -> str:
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})

        # Добавляем вопрос в историю сообщений
        self.history.append(HumanMessage(content=question))

        # Извлекаем только текст из каждого сообщения и объединяем их в один текст
        conversation_history = "\n".join([msg.content for msg in self.history])  # Объединяем тексты в одну строку

        print(conversation_history)
        # Получаем релевантные документы для поиска
        docs = retriever.get_relevant_documents(question)
        print("Найденные документы для запроса:")
        for doc in docs:
            print(doc)

        # Создаем цепочку RetrievalQA с передачей всей истории как строки
        qa = RetrievalQA.from_chain_type(
            llm=self.model,
            retriever=retriever,
        )
        print(conversation_history)
        # Передаем объединенный текст (весь контекст) в модель
        response = qa.run(question)

        # Добавляем ответ модели в историю сообщений
        self.history.append(AIMessage(content=response))

        # Сохраняем вопрос и ответ в JSON файл
        self.save_conversation(question, response)

        return response

    def save_conversation(self, question, response):
        # Структура для записи
        conversation_data = {
            "question": question,
            "answer": response
        }

        # Чтение существующего файла и добавление нового диалога
        try:
            with open(self.log_file, "r") as f:
                conversation_log = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            conversation_log = []

        # Добавляем новый вопрос и ответ
        conversation_log.append(conversation_data)

        # Запись в JSON файл
        with open(self.log_file, "w") as f:
            json.dump(conversation_log, f, indent=4, ensure_ascii=False)

    def process_questions_from_json(self, questions_json_file):
        with open(questions_json_file, "r", encoding="utf-8") as f:
            questions_data = json.load(f)

        responses = []
        for item in questions_data:  # Обрабатываем данные как список
            question = item.get('question')  # Получаем вопрос
            if question:
                response = self.process_query(question)
                responses.append({
                    "question": question,
                    "answer": response
                })
        
        # Сохраняем ответы в новый JSON файл
        with open('answers.json', 'w', encoding='utf-8') as f:
            json.dump(responses, f, indent=4, ensure_ascii=False)

# Пример использования
file_search = LangChainQueryProcessor([r"pars.pdf"])

# Загружаем вопросы из JSON и обрабатываем их
file_search.process_questions_from_json('C:\\Users\\Daniil\\Projects\\my-bot\\evaluation_dataset.json')