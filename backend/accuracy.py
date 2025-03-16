import json
import torch

# ---------------------------
# 1. Загрузка данных из JSON
reference_path = r"C:\Users\Daniil\Projects\my-bot\questions\tea_data_questions.json"  # Путь к JSON с эталонными ответами
model_path = r"C:\Users\Daniil\Projects\my-bot\backend\responses\tea_data_responses.json"           # Путь к JSON с ответами модели


with open(reference_path, "r", encoding="utf-8") as f:
    reference_data = json.load(f)

with open(model_path, "r", encoding="utf-8") as f:
    model_data = json.load(f)

# --------------------------------
# 2. Сопоставление пар по "Вопрос"
# --------------------------------
# Создадим словарь для model_data, чтобы быстро находить ответ модели по "Вопрос".
model_dict = {item["Вопрос"]: item["Ответ модели"] for item in model_data}

# Собираем пары (Вопрос, эталон, модель)
pairs = []
for ref_item in reference_data:
    question = ref_item["Вопрос"]
    ref_answer = ref_item["Эталонный ответ"]
    
    # Ищем соответствующий ответ из model_data
    model_answer = model_dict.get(question)
    if model_answer is not None:
        pairs.append((question, ref_answer, model_answer))

# Извлекаем списки для вычислений
ref_answers = [p[1] for p in pairs]
model_answers = [p[2] for p in pairs]

# -------------------------------------------------
# 3. Подсчёт BERTScore (семантическая близость)
# -------------------------------------------------
from bert_score import score

# Для русского языка указано lang='ru'
P, R, F1 = score(model_answers, ref_answers, lang='ru')

mean_bert_f1 = torch.mean(F1)
print(f"Средний BERTScore (F1): {mean_bert_f1:.4f}")

# ---------------------------------------------------------
# 4. Подсчёт косинусного сходства с Sentence-BERT
# ---------------------------------------------------------
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F

# Выбираем многоязычную модель Sentence-BERT
model_sbert = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

ref_embeddings = model_sbert.encode(ref_answers, convert_to_tensor=True)
model_embeddings = model_sbert.encode(model_answers, convert_to_tensor=True)

cosine_similarities = F.cosine_similarity(model_embeddings, ref_embeddings, dim=1)
mean_cosine_similarity = torch.mean(cosine_similarities)

print(f"Средняя косинусная похожесть (Sentence-BERT): {mean_cosine_similarity:.4f}")

# -------------------------
# 5. Вывод результатов
# -------------------------
print("----- Итоговые результаты -----")
for i, (question, ref_ans, model_ans) in enumerate(pairs):
    print(f"\nПара {i+1}:")
    print(f"  Вопрос:           {question}")
    print(f"  Эталонный ответ:  {ref_ans}")
    print(f"  Ответ модели:     {model_ans}")
    print(f"  BERTScore (F1):   {F1[i].item():.4f}")
    print(f"  Cosine Similarity:{cosine_similarities[i].item():.4f}")

print("\nСводка по всем парам:")
print(f"Средний BERTScore F1: {mean_bert_f1:.4f}")
print(f"Средняя косинусная похожесть (SBERT): {mean_cosine_similarity:.4f}")
