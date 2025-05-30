Backend (app.py)

Функционал:

Детектор гоблинов (/api/detect_goblin)

Анализ изображений (PNG, JPG, JPEG)

Определение "гоблинности" с вероятностью и описанием

Модель: TensorFlow/Keras

Тест личности (/api/personality_test)

7 вопросов для определения интроверт/экстраверт

Модель: PyTorch

Чат с гоблином (/api/chat)

Местная модель или API (OpenRouter)

Фильтрация мата

Модель: Transformers

Технологии: Flask, Python 3, Keras, PyTorch, HuggingFace, OpenRouter API
Особенности: CORS, логирование, защита от ошибок, многопоточность
Запуск:

Редактировать
pip install -r requirements.txt  
python app.py  

Frontend (index.html, tresure.html, styles.css, goblin_app.js)

Интерфейс:

Навигация между секциями (детектор, тест, чат, читалка Кафки)

Пиксельный дизайн и анимации гоблина

Поддержка камер, автопрокрутка, адаптивность

Компоненты (JS):

GoblinDetector – анализ изображений

PersonalityTest – личностный опросник

KafkaBookReader – текстовый ридер

Чат с гоблином – анимированный ввод/вывод

Технологии: HTML5, CSS3, JavaScript (ES6), Fetch API
Требования: современный браузер с поддержкой ES6, Camera API




Models
chat_model.py – чат-бот на Transformers или через API

personality.py – классификация интроверт/экстраверт

goblin_detect.py – модель определения "гоблинности"




Assets
personality_dataset – данные для обучения теста

goblin_photo dataset – изображения для обучения

goblin_dialog – фразы для гоблин-чата
