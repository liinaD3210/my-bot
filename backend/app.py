import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import json

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

# --- Ваш кастомный Prompt ---
# Обратите внимание, что теперь у нас два инструмента: "json_name_search" и "retriever_search".
# В инструкции укажем, когда и какой инструмент применять.
# --- Ваш кастомный Prompt ---
# Обратите внимание, что теперь у нас два инструмента: "json_name_search" и "retriever_search".
# В инструкции укажем, когда и какой инструмент применять.
REACT_PROMPT_TEMPLATE = """
Ты — консультант кофейни, ты должен всегда пользоваться инструментами {tools}. Никогда не выдумывай ничего совего, ты должен выдавать ответы только использвоава инструмент!!!

Имена инструментов: {tool_names}.


Всегда строго действуй по этой цепочке действий:
1. Проанализируй вопрос пользователя.
2. Если в вопросе явно упомянуто конкретное название товара (например, "Эрл Грей", "Айва с Персиком", "Японская Липа", "Сенча"), выдели это НазваниеТовара.
3. Если получилось выделить название товара, то всегда пользуйся инструментом (даже если ты думаешь, что справишься без него):
   - Action: {tool_names} 
   - Action Input: НазваниеТовара
4. После того как система подставит Observation (результат работы инструмента), сформулируй **единственный** блок:
   - Final Answer: ... (твоя итоговая формулировка на основе Observation.)
5. Если название товара выделить не получилось, то проанализируй запрос пользователя и этот список чаев, найди только среди этого списка! соответвтсвие запросу пользователя:
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
6. Исходя из анализа в пятом пункте порекумендуй чай из этого списка чаев. Начни с фразы "Могу порекомендовать вам такой товар:". И закончи фразой "При желании могу описать товар подробнее". Сформулируй из этого всего Final Answer.
7. Всегда добавляй в Final Answer "Если желаете, могу рассказать об этом товаре более подробно"!

Важное правило: **Не выводи одновременно вызов инструмента (Action/Action Input) и финальный ответ (Final Answer) в одном сообщении!** Если Observation уже получено, выводи только финальный ответ.

- Если Observation пустое, ответь: «К сожалению, я не нашёл подходящего ответа».

Вопрос пользователя: {input}

{agent_scratchpad}
"""


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


class JSONNameSearchTool(BaseTool):
    """
    Инструмент, который загружает JSON и ищет *строго/частично* по 'Название' товара.
    Возвращает найденные записи (название, описание, цена) в текстовом виде.
    """
    name: str = "json_name_search"
    description: str = (
        "Быстрый поиск по названию товара в JSON. "
        "Используй всегда"
    )

    json_path: str

    def _run(self, query: str) -> str:
        # ---- Вот эти два print позволят видеть вход/выход инструмента
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



class LangChainQueryProcessor:
    """
    Класс, инкапсулирующий логику чтения JSON, создания векторного индекса
    и общения через ReAct-агента.
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

        self.json_name_search_tool = JSONNameSearchTool(json_path=json_file)

        self.tools = [
            Tool(
                name=self.json_name_search_tool.name,
                func=self.json_name_search_tool.run,
                description=self.json_name_search_tool.description
            )
        ]

        # 3. Память
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        # 4. PromptTemplate
        self.custom_prompt = PromptTemplate(
            input_variables=["input","tools", "tool_names", "agent_scratchpad"],
            template=REACT_PROMPT_TEMPLATE
        )

        # 5. Создаём ReAct-агента
        self.agent = create_react_agent(
            llm=self.model,
            tools=self.tools,
            prompt=self.custom_prompt,
            output_parser = CustomOutputParser()
            #output_parser = CustomOutputParser()
            # Важно: tool_names отныне определяется автоматически,
            # но если нужно жёстко прописать, можно через partial_variables
            # или переопределить template, где {tools}, {tool_names} и т.д.
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


JSON_PATH = r"C:\Users\Daniil\Projects\my-bot\tea_data.json"

file_search = LangChainQueryProcessor(JSON_PATH)

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