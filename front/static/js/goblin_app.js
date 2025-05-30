class GoblinDetector {
    constructor() {
        this.apiUrl = '/api';
    }

    async analyze(imageData, resultDiv) {
        try {
            if (!imageData) {
                throw new Error("Нет данных изображения");
            }

            resultDiv.innerHTML = '<p>Анализируем... ⏳</p>';

            const fileInput = document.getElementById("goblinInput");
            if (!fileInput.files.length) {
                throw new Error("Файл не выбран");
            }
            const file = fileInput.files[0];

            const formData = new FormData();
            formData.append('file', file);

            const response = await fetch(`${this.apiUrl}/detect_goblin`, {
                method: 'POST',
                body: formData,
                headers: {
                    'Accept': 'application/json'
                }
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.error || `Ошибка сервера: ${response.status}`);
            }

            const data = await response.json();
            this.showResult(imageData, data.probability, resultDiv, data.verdict);
        } catch (error) {
            console.error("Ошибка анализа:", error);
            resultDiv.innerHTML = `<p class="error">${error.message}</p>`;
        }
    }

    showResult(imageData, probability, resultDiv, verdict) {
        const percent = Math.round(probability * 100);
        resultDiv.innerHTML = `
            <div class="goblin-result">
                <img src="${imageData}" class="goblin-image" alt="Goblin photo">
                <div class="goblin-meter">
                    <div class="meter-bar" style="width: ${percent}%"></div>
                </div>
                <p class="goblin-percent">${percent}% гоблин</p>
                <p class="goblin-verdict">${verdict}</p>
            </div>
        `;
    }
}

const detector = new GoblinDetector();

async function submitGoblin() {
    const fileInput = document.getElementById("goblinInput");
    const resultDiv = document.getElementById("goblinResult");
    
    if (!fileInput.files.length) {
        resultDiv.innerHTML = '<p class="error">Пожалуйста, выберите файл изображения</p>';
        return;
    }

    const file = fileInput.files[0];
    const reader = new FileReader();
    
    reader.onload = async function(event) {
        try {
            await detector.analyze(event.target.result, resultDiv);
        } catch (error) {
            console.error("Error:", error);
            resultDiv.innerHTML = `<p class="error">${error.message}</p>`;
        }
    };
    
    reader.onerror = () => {
        resultDiv.innerHTML = '<p class="error">Ошибка чтения файла</p>';
    };
    
    reader.readAsDataURL(file);
}

async function captureFromCamera() {
    const resultDiv = document.getElementById('goblinResult');
    const fileInput = document.getElementById('goblinInput');

    if (!navigator.mediaDevices?.getUserMedia) {
        resultDiv.innerHTML = '<p class="error">Камера не поддерживается</p>';
        return;
    }

    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        resultDiv.innerHTML = `
            <div class="camera-container">
                <video class="camera-preview" autoplay></video>
                <button class="bios-button" id="capturePhotoBtn">Сделать фото</button>
            </div>
        `;

        const video = resultDiv.querySelector('video');
        video.srcObject = stream;
        window.cameraStream = stream;

        document.getElementById('capturePhotoBtn').onclick = async () => {
            try {
                // Создаем canvas и рисуем на нем текущий кадр с видео
                const canvas = document.createElement('canvas');
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                const ctx = canvas.getContext('2d');
                ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

                // Останавливаем поток камеры
                if (window.cameraStream) {
                    window.cameraStream.getTracks().forEach(track => track.stop());
                }

                // Конвертируем canvas в Blob и создаем файл
                canvas.toBlob(async (blob) => {
                    const file = new File([blob], "camera-photo.jpg", { type: "image/jpeg" });
                    
                    // Создаем DataTransfer и добавляем файл
                    const dataTransfer = new DataTransfer();
                    dataTransfer.items.add(file);
                    fileInput.files = dataTransfer.files;

                    // Создаем FileReader для отображения превью
                    const reader = new FileReader();
                    reader.onload = async function(event) {
                        await detector.analyze(event.target.result, resultDiv);
                    };
                    reader.onerror = () => {
                        resultDiv.innerHTML = '<p class="error">Ошибка чтения файла</p>';
                    };
                    reader.readAsDataURL(file);

                }, 'image/jpeg', 0.9); // 90% качество JPEG

            } catch (error) {
                console.error("Error capturing photo:", error);
                resultDiv.innerHTML = `<p class="error">${error.message}</p>`;
            }
        };

    } catch (error) {
        resultDiv.innerHTML = `<p class="error">Ошибка камеры: ${error.message}</p>`;
    }
}

document.getElementById('startCameraBtn').onclick = captureFromCamera;

class ModelInterface {
    async checkToxicity(text) {
        const response = await fetch('/api/check_toxicity', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        return await response.json();
    }
}

const modelInterface = new ModelInterface();

function startLoadingAnimation(elementId, baseText) {
    const el = document.getElementById(elementId);
    let dots = 0;
    el.textContent = baseText;
    return setInterval(() => {
        dots = (dots + 1) % 4;
        el.textContent = baseText + '.'.repeat(dots);
    }, 500);
}

async function checkToxicity() {
    const textInput = document.getElementById('toxicTextInput');
    const text = textInput.value.trim();
    
    if (!text) {
        showToxicityError("Пожалуйста, введите текст для проверки");
        return;
    }

    const verdictElement = document.getElementById('toxicityVerdict');
    verdictElement.innerHTML = '';
    
    const intervalId = startLoadingAnimation('toxicityVerdict', 'Анализируем');
    
    try {
        const result = await modelInterface.checkToxicity(text);
        clearInterval(intervalId);
        
        if (result.error) {
            throw new Error(result.error);
        }
        
        updateToxicityUI(result.score);
        
    } catch (error) {
        clearInterval(intervalId);
        showToxicityError(`Ошибка: ${error.message}`);
        console.error("Toxicity check failed:", error);
    }
}

function updateToxicityUI(score) {
    const roundedScore = Math.round(score * 10);
    const scoreElement = document.getElementById('toxicityScore');
    const verdictElement = document.getElementById('toxicityVerdict');
    
    scoreElement.textContent = `Токсичность: ${roundedScore}/10`;
    
    if (roundedScore >= 8) {
        verdictElement.innerHTML = '<span class="high-toxicity">ОПАСНО! Очень токсично! ☢️</span>';
    } else if (roundedScore >= 6) {
        verdictElement.innerHTML = '<span class="medium-toxicity">Токсично! Будьте осторожны!</span>';
    } else if (roundedScore >= 4) {
        verdictElement.innerHTML = '<span class="low-toxicity">Есть признаки токсичности</span>';
    } else {
        verdictElement.innerHTML = '<span class="safe">Всё чисто! 👍</span>';
    }
}

function showToxicityError(message) {
    const verdictElement = document.getElementById('toxicityVerdict');
    verdictElement.innerHTML = `<span class="error">${message}</span>`;
}

async function startChat() {
    const input = document.getElementById("userInput");
    const chat = document.getElementById("chatText");
    const message = input.value.trim();
    
    if (!message) return;
    
    input.value = "";
    chat.innerHTML += `<div class="user-message">Вы: ${message}</div>`;
    
    // Индикатор загрузки с анимацией
    const loadingId = 'loading-' + Date.now();
    chat.innerHTML += `
        <div id="${loadingId}" class="loading-message">
            <span class="loading-dots">.</span>
        </div>
    `;
    animateDots(loadingId);
    
    try {
        const model = document.querySelector('input[name="model"]:checked').value;
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                text: message, 
                model: model 
            })
        });
        
        const data = await response.json();
        
        // Удаляем индикатор загрузки
        document.getElementById(loadingId)?.remove();
        
        const botName = model === 'local' ? 'Локальный Гоблин' : 
                       data.model === 'filter' ? 'Гоблин-Фильтр' : 
                       data.model === 'error' ? 'Гоблин-Ремонтник' : 'Умный Гоблин';
        
        chat.innerHTML += `
            <div class="bot-message ${data.model === 'error' ? 'error' : ''}">
                ${botName}: ${data.response}
            </div>
        `;
        
    } catch (error) {
        document.getElementById(loadingId)?.remove();
        chat.innerHTML += `
            <div class="bot-message error">
                Ошибка: ${error.message}
            </div>
        `;
    } finally {
        chat.scrollTop = chat.scrollHeight;
    }
}

// Анимация точек загрузки
function animateDots(id) {
    let dots = 1;
    const interval = setInterval(() => {
        dots = (dots % 3) + 1;
        const elem = document.getElementById(id);
        if (!elem) {
            clearInterval(interval);
            return;
        }
        elem.querySelector('.loading-dots').textContent = '.'.repeat(dots);
    }, 500);
}

// Инициализация камеры
document.getElementById('startCameraBtn').addEventListener('click', captureFromCamera);
class KafkaBookReader {
    constructor() {
        this.bookPath = '/static/kafka_metamorphosis.txt';
        this.pages = [];
        this.currentPage = 0;
        this.autoTurnInterval = null;
        this.autoTurnSpeed = 5000; // 5 секунд между перелистываниями
        this.init();
    }
    
    async init() {
        await this.loadBook();
        this.setupControls();
        this.showPage(0);
    }
    
    async loadBook() {
        try {
            const response = await fetch(this.bookPath);
            if (!response.ok) throw new Error("Ошибка загрузки");
            const text = await response.text();
            this.pages = this.splitIntoPages(text, 900); // ~2000 символов на страницу
        } catch (error) {
            console.error("Ошибка:", error);
            this.pages = ["Ошибка загрузки текста"];
        }
    }
    
    splitIntoPages(text, charsPerPage) {
        const pages = [];
        for (let i = 0; i < text.length; i += charsPerPage) {
            pages.push(text.substr(i, charsPerPage));
        }
        return pages;
    }
    
    setupControls() {
        const prevBtn = document.querySelector('.book-prev');
        const nextBtn = document.querySelector('.book-next');
        const autoTurnCheckbox = document.getElementById('autoTurn');
        
        // Кнопки навигации
        prevBtn.addEventListener('click', () => {
            this.stopAutoTurn();
            this.showPage(this.currentPage - 1);
        });
        
        nextBtn.addEventListener('click', () => {
            this.stopAutoTurn();
            this.showPage(this.currentPage + 1);
        });
        
        // Чекбокс автоперелистывания
        autoTurnCheckbox.addEventListener('change', (e) => {
            if (e.target.checked) {
                this.startAutoTurn();
            } else {
                this.stopAutoTurn();
            }
        });
    }
    
    showPage(pageNum) {
        // Циклическая навигация
        if (pageNum < 0) pageNum = this.pages.length - 1;
        if (pageNum >= this.pages.length) pageNum = 0;
        
        this.currentPage = pageNum;
        document.querySelector('.book-text').textContent = this.pages[pageNum];
        document.querySelector('.book-page').textContent = `${pageNum + 1}/${this.pages.length}`;
        
        // Сброс прокрутки
        document.querySelector('.book-text').scrollTop = 0;
    }
    
    startAutoTurn() {
        this.stopAutoTurn();
        this.autoTurnInterval = setInterval(() => {
            this.showPage(this.currentPage + 1);
        }, this.autoTurnSpeed);
    }
    
    stopAutoTurn() {
        if (this.autoTurnInterval) {
            clearInterval(this.autoTurnInterval);
            this.autoTurnInterval = null;
        }
        document.getElementById('autoTurn').checked = false;
    }
}

class PersonalityTest {
    constructor() {
        this.questions = [
            "1. Сколько часов в день вы проводите в одиночестве? (0=почти никогда, 10=почти весь день)",
            "2. Насколько сильно вы боитесь публичных выступлений? (0=совсем не боюсь, 10=очень сильно боюсь)",
            "3. Как часто вы посещаете социальные мероприятия? (0=очень редко, 10=очень часто)",
            "4. Как часто вы выходите из дома (не по необходимости)? (0=почти никогда, 10=несколько раз в день)",
            "5. Чувствуете ли вы себя истощённым после общения? (1=Да, 0=Нет)",
            "6. Сколько у вас близких друзей? (0-15)",
            "7. Как часто вы публикуете посты в соцсетях? (0=почти никогда, 10=несколько раз в день)"
        ];
        this.currentQuestion = 0;
        this.answers = [];
        
        this.init();
    }

    init() {
        document.addEventListener('DOMContentLoaded', () => {
            const testDiv = document.getElementById('personalityTest');
            if (!testDiv) return;
            
            testDiv.innerHTML = `
                <p>Определите, интроверт вы или экстраверт</p>
                <button class="pixel-button" id="startTestBtn">Начать тест</button>
                <div id="testContainer"></div>
            `;
            
            document.getElementById('startTestBtn').addEventListener('click', () => this.startTest());
        });
    }

    startTest() {
        const container = document.getElementById('personalityTest');
        
        // Полностью заменяем содержимое, включая кнопку "Начать тест"
        container.innerHTML = `
            <div class="test-content">
                <div id="questionContainer"></div>
                <input type="number" id="answerInput" class="pixel-text">
                <button class="pixel-button" id="nextQuestionBtn">Далее</button>
                <div id="testProgress"></div>
            </div>
        `;
        
        this.currentQuestion = 0;
        this.answers = [];
        this.showQuestion();
        
        document.getElementById('nextQuestionBtn').addEventListener('click', () => this.nextQuestion());
        document.getElementById('answerInput').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.nextQuestion();
        });
    }

    showQuestion() {
        const questionContainer = document.getElementById('questionContainer');
        const progressDiv = document.getElementById('testProgress');
        
        questionContainer.innerHTML = `<p>${this.questions[this.currentQuestion]}</p>`;
        progressDiv.textContent = `Вопрос ${this.currentQuestion + 1} из ${this.questions.length}`;
        
        const input = document.getElementById('answerInput');
        if (this.currentQuestion === 4) {
            input.min = 0;
            input.max = 1;
            input.placeholder = "Введите 0 или 1";
        } else if (this.currentQuestion === 5) {
            input.min = 0;
            input.max = 15;
            input.placeholder = "Введите число от 0 до 15";
        } else {
            input.min = 0;
            input.max = 10;
            input.placeholder = "Введите число от 0 до 10";
        }
        input.value = '';
    }

    async submitTest() {
        const testDiv = document.getElementById('personalityTest');
        try {
            testDiv.innerHTML = '<p>Отправляем результаты... ⏳</p>';
            
            const response = await fetch('/api/personality_test', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ answers: this.answers })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const result = await response.json();
            this.showResult(result);
        } catch (error) {
            console.error("Error submitting test:", error);
            testDiv.innerHTML = `
                <div class="error">
                    <p>Ошибка при обработке теста: ${error.message}</p>
                    <button class="pixel-button" onclick="personalityTest.returnToStart()">Попробовать снова</button>
                </div>
            `;
        }
    }

    showResult(result) {
        const testDiv = document.getElementById('personalityTest');
        const confidencePercent = Math.round(result.confidence * 100);
        
        testDiv.innerHTML = `
            <div class="personality-result">
                <h3>Результат теста</h3>
                <p>Ваш тип личности: <strong>${result.personality === "Extrovert" ? "Экстраверт" : "Интроверт"}</strong></p>
                <p>Уверенность: ${confidencePercent}%</p>
                
                <div class="personality-meter">
                    <div class="meter-bar" style="width: ${confidencePercent}%"></div>
                </div>
                
                <button class="pixel-button" onclick="personalityTest.returnToStart()">Пройти тест снова</button>
            </div>
        `;
    }

    returnToStart() {
        this.currentQuestion = 0;
        this.answers = [];
        const testDiv = document.getElementById('personalityTest');
        testDiv.innerHTML = `
            <p>Определите, интроверт вы или экстраверт</p>
            <button class="pixel-button" id="startTestBtn">Начать тест</button>
            <div id="testContainer"></div>
        `;
        
        document.getElementById('startTestBtn').addEventListener('click', () => this.startTest());
    }

    async nextQuestion() {
        const input = document.getElementById('answerInput');
        const answer = parseFloat(input.value);
        
        // Validate input
        if (isNaN(answer)) {
            alert("Пожалуйста, введите число!");
            return;
        }
        
        if (this.currentQuestion === 4 && ![0, 1].includes(answer)) {
            alert("Для этого вопроса введите 0 или 1!");
            return;
        }
        
        if (this.currentQuestion === 5 && (answer < 0 || answer > 15)) {
            alert("Введите число от 0 до 15!");
            return;
        }
        
        if (this.currentQuestion !== 4 && this.currentQuestion !== 5 && (answer < 0 || answer > 10)) {
            alert("Введите число от 0 до 10!");
            return;
        }
        
        this.answers.push(answer);
        this.currentQuestion++;
        
        if (this.currentQuestion < this.questions.length) {
            this.showQuestion();
        } else {
            await this.submitTest();
        }
    }

    async submitTest() {
        try {
            const response = await fetch('/api/personality_test', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ answers: this.answers })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const result = await response.json();
            this.showResult(result);
        } catch (error) {
            console.error("Error submitting test:", error);
            document.getElementById('personalityTest').innerHTML = 
                `<p class="error">Ошибка при обработке теста: ${error.message}</p>`;
        }
    }

    showResult(result) {
        const testDiv = document.getElementById('personalityTest');
        const confidencePercent = Math.round(result.confidence * 100);
        
        testDiv.innerHTML = `
            <div class="personality-result">
                <h3>=== Результат теста ===</h3>
                <p>Ваш тип личности: <strong>${result.personality === "Extrovert" ? "Экстраверт" : "Интроверт"}</strong></p>
                <p>Уверенность в результате: ${confidencePercent}%</p>
                
                <div class="personality-meter">
                    <div class="meter-bar" style="width: ${confidencePercent}%"></div>
                </div>
                
                <button class="pixel-button" onclick="personalityTest.startTest()">Пройти тест снова</button>
            </div>
        `;
    }
}

const personalityTest = new PersonalityTest();

document.addEventListener('DOMContentLoaded', () => {
    window.personalityTest = new PersonalityTest();
    new KafkaBookReader();
});