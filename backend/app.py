import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS

# LangChain импорт
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

import re
from langchain.schema import AgentAction, AgentFinish
from langchain.agents.agent import AgentOutputParser
from typing import Union

app = Flask(__name__)
CORS(app)


# ====================== КАСТОМНЫЕ ИНСТРУМЕНТЫ ======================

class JSONNameSearchTool(BaseTool):
    """
    Инструмент, который загружает JSON и ищет *строго/частично* по 'Название' товара.
    Возвращает найденные записи (название, описание, цена) в текстовом виде.
    """
    name: str = "json_name_search"
    description: str = (
        "Быстрый поиск по названию товара в JSON. "
        "Используй, если пользователь спрашивает детали о товаре."
    )
    json_path: str

    def _run(self, query: str) -> str:
        print(f"[DEBUG] Tool '{self.name}' called with input: {query}")

        try:
            with open(self.json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            result = f"Ошибка при чтении JSON: {e}"
            print(f"[DEBUG] Tool '{self.name}' output: {result}")
            return result

        query_lower = query.strip().lower()
        results = []
        for idx, item in enumerate(data, start=1):
            name = item.get('Название', '')
            desc = item.get('Описание', '')
            price = item.get('Цена', '')

            if query_lower in name.lower():
                snippet = (
                    f"Название: {name}\n"
                    f"Описание: {desc}\n"
                    f"Цена: {price}"
                )
                results.append(f"[doc {idx}] {snippet}")

        if not results:
            result = ""  # Если ничего не найдено
        else:
            result = "\n".join(results)
        
        print(f"[DEBUG] Tool '{self.name}' output: {result}")
        return result

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("Async not implemented")


class JSONOrderSearchTool(BaseTool):
    """
    Инструмент, который загружает JSON и ищет заказы по 'Номер заказа'.
    Возвращает найденные записи (номер заказа, состав, статус, даты) в текстовом виде.
    """
    name: str = "json_order_search"
    description: str = (
        "Поиск заказа по номеру (для статуса, даты доставки и т.п.). "
        "Используй, если пользователь спрашивает про заказ."
    )
    json_path: str

    def _run(self, query: str) -> str:
        print(f"[DEBUG] Tool '{self.name}' called with input: {query}")

        try:
            with open(self.json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            result = f"Ошибка при чтении JSON: {e}"
            print(f"[DEBUG] Tool '{self.name}' output: {result}")
            return result

        query_lower = query.strip().lower()
        results = []
        for idx, item in enumerate(data, start=1):
            order_number = item.get('Номер заказа', '')
            order_items = item.get('Состав заказа', '')
            order_date = item.get('Дата формирования заказа', '')
            delivery_status = item.get('Статус доставки', '')
            delivery_date = item.get('Дата доставки', '')

            if query_lower in order_number.lower():
                snippet = (
                    f"Номер заказа: {order_number}\n"
                    f"Состав заказа: {order_items}\n"
                    f"Дата формирования заказа: {order_date}\n"
                    f"Статус доставки: {delivery_status}\n"
                    f"Дата доставки: {delivery_date}"
                )
                results.append(f"[doc {idx}] {snippet}")

        if not results:
            result = ""
        else:
            result = "\n".join(results)

        print(f"[DEBUG] Tool '{self.name}' output: {result}")
        return result

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("Async not implemented")


class JSONSimilarProductsTool(BaseTool):
    """
    Инструмент, который загружает JSON и ищет товар по 'Название товара',
    чтобы вернуть 'Похожие товары'.
    """
    name: str = "json_similar_products_search"
    description: str = (
        "Поиск похожих товаров по названию. "
        "Используй, если пользователь спросил: 'Похожие товары' или 'Что похожего на ...'"
    )
    json_path: str

    def _run(self, query: str) -> str:
        print(f"[DEBUG] Tool '{self.name}' called with input: {query}")

        try:
            with open(self.json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            result = f"Ошибка при чтении JSON: {e}"
            print(f"[DEBUG] Tool '{self.name}' output: {result}")
            return result

        query_lower = query.strip().lower()
        results = []
        for idx, item in enumerate(data, start=1):
            prod_name = item.get('Название товара', '')
            similar = item.get('Похожие товары', [])

            if query_lower in prod_name.lower():
                similar_str = ", ".join(similar)
                snippet = (
                    f"Название товара: {prod_name}\n"
                    f"Похожие товары: {similar_str}"
                )
                results.append(f"[doc {idx}] {snippet}")

        if not results:
            result = ""
        else:
            result = "\n".join(results)

        print(f"[DEBUG] Tool '{self.name}' output: {result}")
        return result

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("Async not implemented")


# ====================== КАСТОМНЫЙ OUTPUT PARSER ======================

class CustomOutputParser(AgentOutputParser):
    """
    Кастомный OutputParser, который:
      - Удаляет строки, начинающиеся с "Thought:"
      - Если находит блок "Final Answer:", возвращает AgentFinish
      - Если находит шаблон для действия (Action + Action Input на отдельных строках),
        возвращает AgentAction
      - Иначе возвращает AgentFinish с полным текстом.
    """
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Удаляем строки, начинающиеся с "Thought:"
        cleaned_lines = [
            line for line in llm_output.splitlines() 
            if not line.strip().startswith("Thought:")
        ]
        cleaned_output = "\n".join(cleaned_lines).strip()
        
        # Если найден блок Final Answer, возвращаем финальный ответ
        if "Final Answer:" in cleaned_output:
            final_answer = cleaned_output.split("Final Answer:")[-1].strip()
            return AgentFinish(return_values={"output": final_answer}, log=cleaned_output)
        
        # Ожидаем, что действие записано в виде двух строк:
        # Первая: Action: <tool>
        # Вторая: Action Input: <input>
        action_regex = r"Action:\s*([^\n]+)\nAction Input:\s*(.*)"
        match = re.search(action_regex, cleaned_output, re.DOTALL)
        if match:
            tool = match.group(1).strip()
            tool_input = match.group(2).strip()
            return AgentAction(tool=tool, tool_input=tool_input, log=cleaned_output)
        
        # Если ничего не найдено, возвращаем весь текст как финальный ответ
        return AgentFinish(return_values={"output": cleaned_output}, log=cleaned_output)


# ====================== PROMPT TEMPLATE (расширенный) ======================

REACT_PROMPT_TEMPLATE = """
Ты — консультант кофейни. У тебя есть три инструмента: 
1) json_name_search — поиск подробной информации о товаре по названию,
2) json_order_search — поиск информации о заказах (по номеру заказа),
3) json_similar_products_search — поиск похожих товаров (по названию товара).
Имена инструментов: {tool_names},{tools} 

Никогда не выдумывай ничего своего, ты должен выдавать ответы, используя эти инструменты. 
Не выводи лишней информации, только то, что вернёт инструмент.

Как решать, какой инструмент использовать:

- Если пользователь спрашивает про конкретный номер заказа (например, "TCH-250211-02"), 
  то вызывай: json_order_search на двух отдельных строках.  
  Пример: 
  Action: json_order_search
  Action Input: <номер заказа>.

- Если пользователь спрашивает: «Похожие товары на (название товара)» или «Что есть похожего на (название)?», 
  то вызывай: json_similar_products_search на двух отдельных строках.  
  Пример: 
  Action: json_similar_products_search
  Action Input: <название товара>.

- Если пользователь упоминает конкретное название товара (например, "Эрл Грей") и хочет узнать о нём подробнее, 
  то вызывай: json_name_search на двух отдельных строках.  
  Пример: 
  Action: json_name_search
  Action Input: <название>.

- Если пользователь просит порекомендовать чай по вкусу:
1.Проанализируй запрос пользователя и этот список чаев, найди только среди этого списка! соответвтсвие запросу пользователя:
Каскара Коста-Рика - сухофруктовый, шиповниковый, кураговый, черносливый
Чай Ассам - крепкий, ароматный, насыщенный, сладковатый
Чай Цейлон - сухофруктовый, терпкий, медовый, ароматный
Эрл Грей - черный, цитрусовый, ароматный, классический
Чай черный с чабрецом - черный, травяной, чабрец, успокаивающий
Айва с Персиком - фруктовый, айвовый, персиковый, ананасовый
Сенча - традиционный, китайский, травяной, освежающий
Зеленый с жасмином (Моли Хуа Ча) - зеленый, жасминовый, цветочный, китайский
Японская Липа - зеленый, липовый, ромашковый, цитрусовый
Чай Молочный улун (Най Сян Цзинь Сюань) - молочный, улун, китайский, ароматный
Тегуаньинь - улун, китайский, гармонизирующий, нормализующий
Большой красный халат (Да Хун Пао) - улун, легендарный, бодрящий, иммуномодулирующий
Матча Base - зеленый, порошковый, японский, успокаивающий
Голубая Матча - голубой, порошковый, тайский, цветочный
Иван чай листовой - ферментированный, травяной, расслабляющий, натуральный
Чай Каркаде - кислый, сладковатый, гибискус, ароматный
Гречишный чай Ку Цяо - гречишный, травяной, натуральный, китайский
Ройбуш классический - травяной, натуральный, успокаивающий, общеукрепляющий
Малина с мятой - ягодный, мятный, цветочный, травяной
2. Исходя из анализа в первом пункте порекумендуй чай из этого списка чаев. Начни с фразы "Могу порекомендовать вам такой товар:". И закончи фразой "При желании могу описать товар подробнее". Сформулируй из этого всего Final Answer.
3. Всегда добавляй в Final Answer "Если желаете, могу рассказать об этом товаре более подробно"!

Если Final Answer еще не выведен, то строго действуй по шагам:
1. Проанализируй вопрос пользователя.
2. Определи, требуется ли поиск по заказу, или похожим товарам, или конкретное описание товара.
3. Вызови нужный инструмент на двух отдельных строках, например:
   - Action
   - Action Input
4. После того как система подставит Observation (результат работы инструмента), сформулируй **единственный** блок:
   - Final Answer: ... (твоя итоговая формулировка на основе Observation.).
   
Важное правило: **Не выводи одновременно вызов инструмента (Action/Action Input) и финальный ответ (Final Answer) в одном сообщении!**

Если Observation пустое, ответь: «К сожалению, я не нашёл подходящего ответа».

Вопрос пользователя: {input}

{agent_scratchpad}
"""


# ====================== ОСНОВНОЙ КЛАСС ДЛЯ ОБРАБОТКИ ЗАПРОСОВ ======================

class LangChainQueryProcessor:
    """
    Класс, инкапсулирующий логику чтения JSON, создания ReAct-агента и общения через него.
    """
    def __init__(self, json_file_tea, json_file_orders, json_file_similar):
        # Установим заголовок для GigaChat (при необходимости)
        headers = {
            "X-Session-ID": "8324244b-7133-4d30-a328-31d8466e5503",
        }
        gigachat.context.session_id_cvar.set(headers.get("X-Session-ID"))

        # Токен для GigaChat (пример, у вас может быть свой)
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

        # Инициализируем три инструмента
        self.json_name_search_tool = JSONNameSearchTool(json_path=json_file_tea)
        self.json_order_search_tool = JSONOrderSearchTool(json_path=json_file_orders)
        self.json_similar_products_tool = JSONSimilarProductsTool(json_path=json_file_similar)

        # Собираем их в список Tools
        self.tools = [
            Tool(
                name=self.json_name_search_tool.name,
                func=self.json_name_search_tool.run,
                description=self.json_name_search_tool.description
            ),
            Tool(
                name=self.json_order_search_tool.name,
                func=self.json_order_search_tool.run,
                description=self.json_order_search_tool.description
            ),
            Tool(
                name=self.json_similar_products_tool.name,
                func=self.json_similar_products_tool.run,
                description=self.json_similar_products_tool.description
            ),
        ]

        # Память
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        # PromptTemplate
        self.custom_prompt = PromptTemplate(
            input_variables=["input","tools", "tool_names", "agent_scratchpad"],
            template=REACT_PROMPT_TEMPLATE
        )

        # Создаём ReAct-агента
        self.agent = create_react_agent(
            llm=self.model,
            tools=self.tools,
            prompt=self.custom_prompt,
            output_parser=CustomOutputParser()
        )

        # Оборачиваем агента в Executor
        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=self.agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5
        )

    def process_query_with_agent(self, user_input: str) -> str:
        """Вызываем нашего агента (с учётом истории в памяти)."""
        result = self.agent_executor({"input": user_input})
        response = result["output"]
        return response


# ====================== ИНИЦИАЛИЗАЦИЯ НАШЕГО КЛАССА ======================

# Пропишите пути к вашим файлам
JSON_TEA_PATH = r"C:\Users\Daniil\Projects\my-bot\tea_data.json"            # Товары (название, описание, цена)
JSON_ORDERS_PATH = r"C:\Users\Daniil\Projects\my-bot\orders.json"      # Заказы (номер заказа, статус, дата и т.д.)
JSON_SIMILAR_PATH = r"C:\Users\Daniil\Projects\my-bot\similar_products.json" # Похожие товары

file_search = LangChainQueryProcessor(
    json_file_tea=JSON_TEA_PATH,
    json_file_orders=JSON_ORDERS_PATH,
    json_file_similar=JSON_SIMILAR_PATH
)


# ====================== FLASK-СЕРВИС ======================

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
