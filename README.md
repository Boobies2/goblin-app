
# 🏗️ Гоблинский МЛ-Сайт для Двача

Этот проект — солянка из 3 нейросетевых моделей, текстовой книжки и гоблинской эстетики

## 🧰 Технологический состав
- **Goblin Detector** (Keras) - определяет гоблинность по фото
- **Personality Classifier** (PyTorch) - интроверт/экстраверт по ответам
- **Goblin Chat** (HuggingFace) - дообученный GPT для гоблинского общения
- **Kafka Reader** - электронная книжка с автопрокруткой

## 📁 Структура проекта
```plaintext
goblin_site/
├── api/
│   ├── models/
│   │    ├── goblin_detector.h5          # Модель детекции гоблинов
│   │    ├── personality_model.pth       # Классификатор личности
│   │    └── model_chat/                 # Локальная LLM
│   │         ├── config.json
│   │         ├── model.safetensors
│   │         └── ...                    # Файлы токенизатора
│   └── app.py                           # Flask-бэкенд
│  
└── front/
     ├── static/
     │     ├── img/                      # Гоблинские арты
     │     ├── js/goblin_app.js          # Вся фронтенд-магия
     │     ├── styles.css                # CRT-стили
     │     └── kafka_metamorphosis.txt   # Книга для чтения
     ├── index.html                      # Главная страница
     └── tresure.html                    # Сокровищница с гифкой
```

## ⚙️ Установка
```bash
# Клонируем репозиторий
git clone https://github.com/Polynka/goblin-site.git
cd goblin-site

# Ставим зависимости
pip install -r requirements.txt

# Запускаем!
python api/app.py
```

`requirements.txt`:
```txt
flask==2.0.1
torch==1.9.0
transformers==4.12.3
tensorflow==2.6.0
Pillow==8.3.1
```

## 🧠 Модели

### 1. Гоблин-детектор
- Архитектура: CNN на Keras
- Обучен на 500+ изображениях гоблинов/не-гоблинов
- Возвращает вероятность гоблинности + забавный вердикт

### 2. Классификатор личности
```python
class PersonalityClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(7, 64)  # 7 вопросов -> 64 нейрона
        self.fc2 = nn.Linear(64, 2)   # 2 класса
```

### 3. Гоблин-чат
- Основа: `sberbank-ai/rugpt3small_based_on_gpt2`
- Дообучен на диалогах из Telegram
- Альтернатива: API через OpenRouter

## 🌐 Интерфейс
```bash
python api/app.py  # Запуск сервера
```
Открыть в браузере: `http://localhost:5000`

**Фичи:**
- 📷 Загрузка фото/съемка с камеры
- 📖 Автоматическая прокрутка "Превращения"
- 🧪 Тест на интроверсию
- 💬 Чат с выбором модели (локальная/API)

```
       ,      ,
      /(.-""-.)\
  |\  \/      \/  /|
  | \ / =.  .= \ / |
  \( \   o\/o   / )/
   \_, '-/  \-' ,_/
     /   \__/   \
     \ \__/\__/ /
   ___\ \|--|/ /___
 /`    \      /    `\
/       '----'       \
```

## 🏆 Сокровищница
- Секретная гифка 
- Анимация через CSS/JS
- Кнопка паузы с изменением состояния

```
