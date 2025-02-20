import requests
from bs4 import BeautifulSoup
import json
import html

# URL страницы с чаем
url = "https://ingresso.coffee/catalog/tea/"

# Загружаем страницу
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

# Находим все элементы с чаем
tea_items = soup.find_all('li', class_='product')

for tea in tea_items:
    # Извлекаем название
    title_elem = tea.find('div', class_='hcp_item_title')
    title = title_elem.get_text(strip=True) if title_elem else 'Не найдено'
    
    # Извлекаем описание
    desc_elem = tea.find('div', class_='hcp_item_description')
    description = desc_elem.get_text(strip=True) if desc_elem else 'Не найдено'
    
    # Пытаемся извлечь цену из JSON, записанного в data-атрибуте формы
    form_elem = tea.find('form', class_='variations_form')
    price = 'Не найдено'
    if form_elem and form_elem.has_attr('data-product_variations'):
        data_variations = form_elem['data-product_variations']
        # Раскодируем HTML-сущности, чтобы получить корректный JSON
        data_variations = html.unescape(data_variations)
        try:
            variations = json.loads(data_variations)
            if variations and isinstance(variations, list):
                price_val = variations[0].get("display_price")
                if price_val is not None:
                    price = str(price_val)
        except Exception as e:
            price = 'Ошибка при парсинге цены'
    
    # Выводим данные
    print(f"Название: {title}")
    print(f"Цена: за 100 гр: {price}")
    print(f"Описание: {description}")
    print('-' * 40)
