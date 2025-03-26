import json
import os
from langchain.tools import BaseTool

class JSONNameSearchTool(BaseTool):
    """
    Инструмент, который загружает JSON и ищет *строго/частично* по 'Название' товара.
    Возвращает найденные записи с названием и значением указанного ключа в текстовом виде.
    """
    name: str = "json_name_search"
    description: str = (
        "Быстрый поиск по названию товара в JSON. "
        "Ожидает ввод в формате 'Название, Ключ'. Например: 'чай Эрл Грей, Описание'."
    )

    json_path: str

    def _run(self, query: str) -> str:
        # Ожидаем, что query имеет формат: "Название, Ключ"
        try:
            search_term, key = [x.strip() for x in query.split(",", 1)]
        except ValueError:
            return "Ошибка: ввод должен быть в формате 'Название, Ключ'."
        
        try:
            with open(self.json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            return f"Ошибка при чтении JSON: {e}"
        
        search_term_lower = search_term.lower()
        results = []
        for idx, item in enumerate(data, start=1):
            name = item.get('Название', '')
            # Проверка по частичному совпадению в названии
            if search_term_lower in name.lower():
                # Если искомый ключ не найден, выводим сообщение об отсутствии
                value = item.get(key, "Поле не найдено")
                snippet = f"Название: {name}\n{key}: {value}"
                results.append(f"[doc {idx}] {snippet}")
        
        if not results:
            return "Ничего не найдено."
        return "\n".join(results)

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("Async not implemented")

if __name__ == "__main__":
    # Определяем путь к тестовому JSON-файлу
    json_file = r"C:\Users\Daniil\Projects\my-bot\tea_data.json"

    # Если файла не существует, создаём тестовый JSON
    if not os.path.exists(json_file):
        sample_data = [
            {"Название": "чай Эрл Грей", "Описание": "Ароматизированный чай с бергамотом", "Цена": "200"},
            {"Название": "черный чай", "Описание": "Классический черный чай", "Цена": "150"},
            {"Название": "зеленый чай", "Описание": "Свежий зеленый чай", "Цена": "180"}
        ]
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(sample_data, f, ensure_ascii=False, indent=2)
        print(f"Создан тестовый JSON-файл: {json_file}")

    # Создаём экземпляр инструмента, передавая путь к JSON
    tool = JSONNameSearchTool(json_path=json_file)

    # Получаем запрос от пользователя
    query = input("Введите (Название, Ключ): ").strip()

    # Вызываем инструмент (синхронная версия)
    result = tool._run(query)
    if result:
        print("Результат поиска:")
        print(result)
    else:
        print("Ничего не найдено.")
