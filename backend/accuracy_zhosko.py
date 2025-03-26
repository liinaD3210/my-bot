import json

# Пути к вашим JSON-файлам
model_answers_file = r'backend\responses\tea_prices.json'
reference_answers_file = r'questions\tea_data_questions.json'
results_file = r'backend\test_results\tea_prices_results.txt'

# Считываем JSON с ответами модели
with open(model_answers_file, 'r', encoding='utf-8') as f:
    model_answers = json.load(f)  # Список словарей: {"Вопрос": "...", "Ответ модели": "..."}

# Считываем JSON с эталонными ответами
with open(reference_answers_file, 'r', encoding='utf-8') as f:
    reference_answers = json.load(f)  # Список словарей: {"Вопрос": "...", "Эталонный ответ": "..."}

# Создаем словарь эталонных ответов по ключу "Вопрос"
reference_dict = {item["Вопрос"]: item["Эталонный ответ"].strip() for item in reference_answers}

total_accuracy = 0  # сумма точностей (в числовом виде: 100 или 0)
count = 0         # количество проверенных ответов

with open(results_file, 'w', encoding='utf-8') as out:
    for answer_item in model_answers:
        question = answer_item["Вопрос"]
        model_answer = answer_item["Ответ модели"]
        
        # Получаем эталонный ответ для данного вопроса, если он есть
        if question in reference_dict:
            reference_answer = reference_dict[question]
        else:
            continue  # пропускаем, если вопрос отсутствует в эталонном файле

        # Проверяем, содержится ли эталонный ответ (число) в ответе модели
        if reference_answer in model_answer:
            accuracy_value = 100
            accuracy = "100%"
        else:
            accuracy_value = 0
            accuracy = "0%"

        total_accuracy += accuracy_value
        count += 1

        # Записываем результат для данного вопроса
        out.write(f"Вопрос: {question}\n")
        out.write(f"Ответ модели: {model_answer}\n")
        out.write(f"Эталонный ответ: {reference_answer}\n")
        out.write(f"Точность: {accuracy}\n\n")
    
    # Вычисляем среднюю точность
    avg_accuracy = total_accuracy / count if count else 0
    out.write(f"Средняя точность: {avg_accuracy}%\n")

print("Готово! Результаты сохранены в results.txt")
