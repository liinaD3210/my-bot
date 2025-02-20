from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_gigachat.chat_models import GigaChat
from langchain.chains import RetrievalQA
import json

app = Flask(__name__)
CORS(app)

class LangChainQueryProcessor:
    def __init__(self, pdf_files):
        """
        Инициализация процессора запросов.
        :param pdf_files: Список PDF-файлов, из которых нужно извлечь текст.
        """
        self.pdf_files = pdf_files

        GIGACHAT_CREDENTIALS = 'ZTkxOGNjNDktYjdkMy00MjY2LTkxZWMtNzY2NGUwZmQ0YThhOmFjYjgzYzU4LTllYzAtNGFkMi05YTFiLWI3YTllY2Y1OTc5Ng=='

        # Инициализируем модель GigaChat
        self.model = GigaChat(
            credentials=GIGACHAT_CREDENTIALS,
            scope="GIGACHAT_API_PERS",
            model="GigaChat-Max",
            verify_ssl_certs=False,
        )

        # Создаём векторное хранилище на основе PDF-файлов
        self.vectorstore = self._initialize_vectorstore()

    def _initialize_vectorstore(self):
        """
        Создает векторное хранилище на основе документов из PDF-файлов.
        """
        # Настраиваем текстовый «сплиттер» для разбиения больших текстовых блоков на чанки
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        all_docs = []

        # Проходимся по каждому PDF-файлу
        for pdf_file in self.pdf_files:
            loader = UnstructuredPDFLoader(pdf_file)
            # Загружаем и сразу разбиваем содержимое PDF на чанки
            splitted_data = loader.load_and_split(text_splitter)
            all_docs.extend(splitted_data)

        # Создаем векторное хранилище из полученных документов
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(all_docs, embedding_model)
        return vectorstore

    def process_query(self, question: str) -> str:
        """
        Выполняет поиск по базе (retriever) и возвращает итоговый ответ от GigaChat.
        :param question: Вопрос для модели.
        :return: Ответ от модели в формате строки.
        """
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})

        # Создаём цепочку RetrievalQA, которая будет искать релевантные документы и формировать ответ
        qa = RetrievalQA.from_chain_type(
            llm=self.model,
            retriever=retriever,
        )

        # Запускаем цепочку с заданным вопросом
        response = qa.run(question)
        return response

# Инициализация объекта для поиска по файлам
file_search = LangChainQueryProcessor([r"C:\Users\Daniil\Projects\my-bot\pars.pdf"])

@app.route('/bot', methods=['POST'])
def bot():
    # Получаем данные от пользователя
    data = request.json
    user_input = data.get('message', '')

    if user_input:
        # Сначала обрабатываем запрос через поиск по файлам
        search_result = file_search.process_query(user_input)
        # Теперь передаем результат поиска в GigaChat для дальнейшего ответа
        response_text = search_result
        return jsonify({'response': response_text})
    else:
        return jsonify({'response': 'Пожалуйста, отправьте текст!'}), 400

def process_user_input(user_input):
    # Здесь можно делать какие-то вычисления или другие операции
    response = file_search.model.ask(user_input)  # Заменили chat на ask
    return response.choiceииs[0].message.content

if __name__ == '__main__':
    app.run(debug=True)
