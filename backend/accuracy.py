import json
import torch
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F

# ---------------------------
# 1. Загрузка данных из JSON
# ---------------------------
reference_path = r"questions\tea_data_questions.json"  # Эталонные ответы
model_path = r"backend\responses\tea_descriptions.json"  # Ответы модели

with open(reference_path, "r", encoding="utf-8") as f:
    reference_data = json.load(f)

with open(model_path, "r", encoding="utf-8") as f:
    model_data = json.load(f)

# --------------------------------
# 2. Сопоставление пар по "Вопрос"
# --------------------------------
# Создаём словарь для быстрого поиска ответа модели по "Вопрос".
model_dict = {item["Вопрос"]: item["Ответ модели"] for item in model_data}

# Собираем пары (Вопрос, эталонный ответ, ответ модели)
pairs = []
for ref_item in reference_data:
    question = ref_item["Вопрос"]
    ref_answer = ref_item["Эталонный ответ"]
    model_answer = model_dict.get(question)
    if model_answer is not None:
        pairs.append((question, ref_answer, model_answer))

# Извлекаем списки эталонных и модельных ответов
ref_answers = [p[1] for p in pairs]
model_answers = [p[2] for p in pairs]

# ---------------------------------------------------------
# 3. Подсчёт косинусного сходства с Sentence-BERT
# ---------------------------------------------------------
# Используем многоязычную модель Sentence-BERT
model_sbert = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

ref_embeddings = model_sbert.encode(ref_answers, convert_to_tensor=True)
model_embeddings = model_sbert.encode(model_answers, convert_to_tensor=True)

# Вычисляем косинусное сходство для каждой пары
cosine_similarities = F.cosine_similarity(model_embeddings, ref_embeddings, dim=1)

# Функция для расчёта итогового балла по правилу
def calculate_score(similarity):
    if similarity >= 0.75:
        return 100
    elif similarity >= 0.5:
        return similarity * 100
    else:
        return 0

# Применяем функцию к каждой паре
scores = [calculate_score(sim.item()) for sim in cosine_similarities]
mean_cosine_similarity = torch.mean(cosine_similarities)
mean_score = sum(scores) / len(scores)

# -------------------------
# 4. Формирование результатов для сохранения
# -------------------------
results_lines = []
results_lines.append("----- Итоговые результаты -----\n")
for i, (question, ref_ans, model_ans) in enumerate(pairs):
    sim = cosine_similarities[i].item()
    score_val = scores[i]
    results_lines.append(f"Пара {i+1}:\n")
    results_lines.append(f"  Вопрос:            {question}\n")
    results_lines.append(f"  Эталонный ответ:   {ref_ans}\n")
    results_lines.append(f"  Ответ модели:      {model_ans}\n")
    results_lines.append(f"  Cosine Similarity: {sim:.4f}\n")
    results_lines.append(f"  Итоговый балл:     {score_val:.2f}%\n\n")

results_lines.append("----- Сводка по всем парам -----\n")
results_lines.append(f"Средняя Cosine Similarity (SBERT): {mean_cosine_similarity:.4f}\n")
results_lines.append(f"Средний итоговый балл: {mean_score:.2f}%\n")

results_text = "".join(results_lines)

# Вывод в консоль
print(results_text)

# -------------------------
# 5. Сохранение результатов в файл
# -------------------------
output_path = r"backend\test_results\tea_data_results.txt"
with open(output_path, "w", encoding="utf-8") as out_file:
    out_file.write(results_text)
