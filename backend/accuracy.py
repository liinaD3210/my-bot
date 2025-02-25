import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Загружаем модель для создания эмбеддингов
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def calculate_similarity(answers_file, evaluation_file, threshold=0.8):
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
        question = model_answer['question']
        model_answer_text = model_answer['answer']
        reference_answer_text = reference_answer['answer']

        # Преобразуем ответы в эмбеддинги
        model_embedding = model.encode(model_answer_text)
        reference_embedding = model.encode(reference_answer_text)

        # Рассчитываем схожесть между эмбеддингами
        similarity = cosine_similarity([model_embedding], [reference_embedding])[0][0]

        # Если схожесть больше порогового значения, считаем ответ правильным
        if similarity >= threshold:
            correct_answers += 1

    # Рассчитываем точность
    accuracy = correct_answers / total_answers * 100
    print(f"Accuracy: {accuracy:.2f}%")

# Пример использования
calculate_similarity('answers.json', 'evaluation_dataset.json', threshold=0.8)
