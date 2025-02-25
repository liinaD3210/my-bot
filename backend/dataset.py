import json

# Загружаем данные из файла JSON
with open('tea_data.json', 'r', encoding='utf-8') as f:
    tea_data = json.load(f)

questions = []
answers = []

# Формируем набор вопросов и ответов
for tea in tea_data:
    title = tea.get('Название', 'Не найдено')
    description = tea.get('Описание', 'Не найдено')
    price = tea.get('Цена', 'Не найдено')

    questions.append(f"Какая цена у {title}?")
    answers.append(f"за 100 гр: {price}")
    questions.append(f"Опиши мне {title}")
    answers.append(description)

# Сохраняем набор вопросов и ответов в JSON
dataset = [{"question": q, "answer": a} for q, a in zip(questions, answers)]

with open('evaluation_dataset.json', 'w', encoding='utf-8') as f:
    json.dump(dataset, f, ensure_ascii=False, indent=4)

print("Датасет для оценки успешно сохранен в evaluation_dataset.json")
