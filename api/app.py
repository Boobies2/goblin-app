import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

from flask import send_from_directory
from flask import Flask, render_template, request, jsonify
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from keras.models import load_model
import io
import logging
from flask_cors import CORS
import requests
from sklearn.preprocessing import LabelEncoder
from concurrent.futures import ThreadPoolExecutor  
import logging
from logging.handlers import TimedRotatingFileHandler
import os
from datetime import datetime

def setup_logging():
    """Настройка логирования с ротацией по времени"""
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_handler = TimedRotatingFileHandler(
        filename=os.path.join(log_dir, 'goblin_app.log'),
        when='midnight',  
        interval=1,       
        backupCount=7,    
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

setup_logging()

# Инициализация Flask приложения
app = Flask(__name__,
            static_folder=os.path.join('..', 'front', 'static'),
            template_folder=os.path.join('..', 'front'))

# Конфигурация CORS
CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:*", "http://127.0.0.1:*"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Настройка логгирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Константы
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
OPENROUTER_API_KEY = "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"  # Замените на реальный ключ
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "mistralai/devstral-small:free"  # Единственная модель

# Глобальные переменные для моделей
models = {
    'goblin': None,
    'chat': None,
    'personality': None
}
tokenizers = {
    'chat': None
}
label_encoder = None
executor = ThreadPoolExecutor(max_workers=4)  # Теперь определено

# Модель классификатора личности
class PersonalityClassifier(nn.Module):
    def __init__(self, input_size=7, num_classes=2):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

def allowed_file(filename):
    """Проверка разрешенных расширений файлов"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_models():
    global models, tokenizers, label_encoder
    
    try:
        # 1. Модель детектора гоблина (Keras)
        logger.debug("Загрузка модели goblin_detector2.h5...")
        models['goblin'] = load_model(os.path.join("models", "goblin_detector.h5"))
        
        # 2. Модель чат-бота (HuggingFace)
        logger.debug("Загрузка модели чата...")
        tokenizers['chat'] = AutoTokenizer.from_pretrained(os.path.join("models", "model_chat"))
        models['chat'] = AutoModelForCausalLM.from_pretrained(os.path.join("models", "model_chat"))
        
        # 3. Модель личности (PyTorch)
        logger.debug("Загрузка модели теста личности...")
        models['personality'] = PersonalityClassifier()
        models['personality'].load_state_dict(torch.load(os.path.join("models", "personality_model.pth")))
        models['personality'].eval()
        
        # Инициализация LabelEncoder
        label_encoder = LabelEncoder()
        label_encoder.fit(['Extrovert', 'Introvert'])
        
        logger.info("Все модели успешно загружены")
    
    except Exception as e:  # <- Этот блок был пропущен!
        logger.error(f"Ошибка загрузки моделей: {str(e)}")
        raise

def preprocess_image(image_file):
    """Предобработка изображения для модели"""
    try:
        logger.debug("[1/5] Начало обработки изображения...")
        
        # Проверка, что файл не пустой
        if not image_file:
            raise ValueError("Файл изображения пустой")
        
        # Перемотка файла на начало (если он уже читался)
        if hasattr(image_file, 'seek'):
            image_file.seek(0)
        
        logger.debug("[2/5] Чтение файла...")
        img_bytes = image_file.read()
        
        if not img_bytes:
            raise ValueError("Не удалось прочитать данные из файла")
        
        logger.debug("[3/5] Открытие изображения...")
        img = Image.open(io.BytesIO(img_bytes))
        
        logger.debug(f"[4/5] Формат изображения: {img.format}, режим: {img.mode}, размер: {img.size}")
        
        # Конвертация в RGB (если PNG с альфа-каналом или grayscale)
        if img.mode != 'RGB':
            logger.debug("Конвертация в RGB...")
            img = img.convert('RGB')
        
        logger.debug("[5/5] Изменение размера и нормализация...")
        img = img.resize((128, 128))  # Размер должен совпадать с входом модели
        img_array = np.array(img) / 255.0  # Нормализация [0, 1]
        
        # Проверка формы массива
        if img_array.shape != (128, 128, 3):
            raise ValueError(f"Неверная форма изображения: {img_array.shape}. Ожидается (128, 128, 3)")
        
        return np.expand_dims(img_array, axis=0)  # Добавляем размерность батча: (1, 128, 128, 3)
        
    except Exception as e:
        logger.error(f"Ошибка обработки изображения: {str(e)}", exc_info=True)
        raise

def interpret_goblin(prob):
    """Интерпретация результата детекции гоблина"""
    if prob > 0.8:
        return "Да ты.. Да ты.. да ты.. Гоблин!! 🏆"
    elif prob > 0.6:
        return "ну похож чучуть"
    elif prob > 0.4:
        return "Нос гоблинский чтоль"
    elif prob > 0.2:
        return "Если глаза закрыть то чучуть похож"
    else:
        return "ТЫ НЕ ГОБЛИН"

# Загрузка моделей при старте приложения
load_models()

# ===== Роуты =====
@app.route('/')
def index():
    """Главная страница"""
    return render_template('index.html')

@app.route('/tresure')
def tresure():
    """Страница сокровищницы"""
    return render_template('tresure.html')

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/api/detect_goblin', methods=['POST'])
def api_detect_goblin():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "Требуется файл изображения"}), 400
            
        file = request.files['file']
        
        # Проверка, что файл есть и имеет допустимое имя
        if file.filename == '':
            return jsonify({"error": "Файл не выбран"}), 400
        
        # Проверка расширения файла
        if not allowed_file(file.filename):
            return jsonify({"error": "Разрешены только PNG, JPG, JPEG"}), 400
        
        # Логирование информации о файле
        logger.info(f"Получен файл: {file.filename}, размер: {file.content_length} bytes")
        
        # Попробуйте сохранить файл для отладки
        debug_path = "debug_upload.jpg"
        file.save(debug_path)
        logger.info(f"Файл сохранён для отладки: {debug_path}")
        
        # Обработка изображения
        img = preprocess_image(file)
        probability = float(models['goblin'].predict(img, verbose=0)[0][0])
        
        return jsonify({
            "probability": probability,
            "verdict": interpret_goblin(probability)
        })

    except Exception as e:
        logger.error(f"Ошибка в detect_goblin: {str(e)}", exc_info=True)
        return jsonify({"error": f"Ошибка обработки изображения: {str(e)}"}), 500

@app.route('/api/personality_test', methods=['POST'])
def personality_test():
    try:
        # Добавим логгирование
        app.logger.info("Получен запрос на тест личности")
        
        if not models.get('personality'):
            app.logger.error("Модель личности не загружена!")
            return jsonify({"error": "Модель личности не загружена"}), 503
            
        answers = request.json.get('answers', [])
        app.logger.info(f"Получены ответы: {answers}")
        
        if len(answers) != 7:
            return jsonify({"error": "Требуется 7 ответов"}), 400
            
        input_tensor = torch.tensor([answers], dtype=torch.float32)
        
        with torch.no_grad():
            outputs = models['personality'](input_tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)
            personality = label_encoder.inverse_transform(predicted.numpy())[0]
        
        interpretation = "Экстраверт" if personality == "Extrovert" else "Интроверт"
        app.logger.info(f"Результат: {personality} (уверенность: {confidence.item()})")
        
        return jsonify({
            "personality": personality,
            "confidence": confidence.item(),
            "interpretation": interpretation
        })
        
    except Exception as e:
        app.logger.error(f"Ошибка в тесте личности: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/api/chat', methods=['POST'])
def api_chat():
    try:
        data = request.get_json()
        user_input = data.get('text', '').strip().lower()
        use_local = data.get('model', '').lower() == 'local'

        if not user_input:
            return jsonify({"error": "Введите текст"}), 400

        # Фильтрация нежелательных запросов
        restricted_words = ['пись', 'писа', 'пися', 'хуй', 'пизд']
        if any(word in user_input for word in restricted_words):
            return jsonify({
                "user_message": user_input,
                "response": "Гоблины не обсуждают такие вещи! *сердито надувает щёки*",
                "model": "filter"
            })

        if use_local:
            # Локальная модель
            inputs = tokenizers['chat'](user_input, return_tensors='pt')
            outputs = models['chat'].generate(
                **inputs,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                no_repeat_ngram_size=2
            )
            full_response = tokenizers['chat'].decode(outputs[0], skip_special_tokens=True)
            response = full_response.split('\n')[0]
            
            logger.info(f"Local model response: {response}")
            
            return jsonify({
                "user_message": user_input,
                "response": response,
                "model": "local"
            })
        else:
            headers = {
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost:5000",
                "X-Title": "Goblin Chat"
            }
            
            payload = {
                "model": OPENROUTER_MODEL,
                "messages": [
                    {
                        "role": "system", 
                        "content": "Ты дружелюбный гоблин-фильтр. Отвечай вежливо и культурно. Избегай неприемлемых тем."
                    },
                    {"role": "user", "content": user_input}
                ],
                "temperature": 0.7,
                "max_tokens": 100
            }
            
            response = requests.post(OPENROUTER_API_URL, headers=headers, json=payload)
            response_data = response.json()
            
            bot_response = response_data['choices'][0]['message']['content']
            
            return jsonify({
                "user_message": user_input,
                "response": bot_response,
                "model": OPENROUTER_MODEL
            })

    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        return jsonify({
            "user_message": user_input,
            "response": "Ой, что-то сломалось! Гоблин в замешательстве...",
            "model": "error"
        }), 500

def clean_goblin_response(text):
    """Очистка ответа гоблина от ненужных частей"""
    import re
    # Удаляем текст в скобках
    text = re.sub(r'\([^)]*\)', '', text)
    # Удаляем повторяющиеся фразы
    words = text.split()
    unique_words = []
    for word in words:
        if word not in unique_words[-3:]:  # Проверяем последние 3 слова
            unique_words.append(word)
    return ' '.join(unique_words).strip()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)