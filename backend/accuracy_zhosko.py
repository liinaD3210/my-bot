import json

def calculate_accuracy(answers_file, evaluation_file):
    # Загружаем ответы модели
    with open(answers_file, "r", encoding="utf-8") as f:
        model_answers = json.load(f)
    
    # Загружаем эталонные ответы
    with open(evaluation_file, "r", encoding="utf-8") as f:
        reference_answers = json.load(f)

    correct_answers = 0
    total_answers = len(reference_answers)

    # Сравниваем ответы модели с эталонными ответами
    for model_answer, reference_answer in zip(model_answers, reference_answers):
        # Сравниваем ответы, игнорируя пробелы и регистр
        if model_answer['answer'].strip().lower() == reference_answer['answer'].strip().lower():
            correct_answers += 1

    # Рассчитываем точность
    accuracy = correct_answers / total_answers * 100
    print(f"Accuracy: {accuracy:.2f}%")

# Пример использования
calculate_accuracy('answers.json', 'evaluation_dataset.json')
