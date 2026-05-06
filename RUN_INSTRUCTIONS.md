# 🚀 AutoDevAgent - Инструкция по запуску

## Быстрый старт

### 1. Установка зависимостей

```bash
# Установить все зависимости (включая OpenAI и Web UI)
pip install -e ".[all]"

# Или минимальная установка (без OpenAI и Web UI)
pip install -e .

# Только для разработки
pip install -e ".[dev]"

# Только Web UI
pip install -e ".[web]"
```

### 2. Настройка API ключей

#### Вариант A: Через переменные окружения (рекомендуется)

```bash
# Для OpenAI
export OPENAI_API_KEY="sk-your-api-key-here"

# Опционально: использовать альтернативный endpoint
export OPENAI_BASE_URL="https://api.openai.com/v1"

# Для локальных моделей через Ollama
export OLLAMA_BASE_URL="http://localhost:11434"
```

#### Вариант B: Через файл .env

Создайте файл `.env` в корне проекта:

```bash
OPENAI_API_KEY=sk-your-api-key-here
OLLAMA_BASE_URL=http://localhost:11434
ADEV_DEFAULT_MODEL=gpt-4
ADEV_LOG_LEVEL=INFO
```

### 3. Проверка установки

```bash
# Проверить версию и доступные команды
adev --help

# Показать статус системы
adev status

# Показать конфигурацию
adev config --show
```

---

## Запуск задач

### Через CLI

```bash
# Выполнить задачу с декомпозицией на подзадачи
adev run "Создать REST API для управления задачами"

# Выполнить задачу без декомпозиции
adev run "Написать функцию для сортировки массива" --no-decompose

# Указать количество шагов декомпозиции
adev run "Реализовать аутентификацию" --steps 7

# Использовать конкретную модель
adev run "Сделать тесты" --model gpt-3.5-turbo
```

### Примеры задач

```bash
# Генерация кода
adev run "Создать класс User с полями id, name, email и методами save(), delete()"

# Рефакторинг
adev run "Рефакторить функцию process_data, разбить на меньшие функции"

# Документация
adev run "Написать документацию для модуля auth.py"

# Тесты
adev run "Написать unit-тесты для функции calculate_total"
```

---

## Web UI

### Запуск веб-интерфейса

```bash
# Запустить на порту 8000 (по умолчанию)
adev web

# Запустить на другом порту
adev web --port 9000

# С авто-перезагрузкой при изменении кода
adev web --reload

# Запустить на конкретном хосте
adev web --host 127.0.0.1 --port 8080
```

После запуска откройте браузер:
```
http://localhost:8000
```

### Возможности Web UI

- 📊 Дашборд со статистикой (задачи, токены, стоимость)
- ✅ Создание новых задач через веб-интерфейс
- 📋 Просмотр списка задач и их статуса
- 🔄 Авто-обновление каждые 5 секунд
- 🔌 WebSocket для real-time обновлений

### REST API

Web UI также предоставляет REST API:

```bash
# Получить статус системы
curl http://localhost:8000/api/status

# Создать задачу
curl -X POST http://localhost:8000/api/tasks \
  -H "Content-Type: application/json" \
  -d '{"description": "Сделать что-то полезное", "steps": 5}'

# Получить список задач
curl http://localhost:8000/api/tasks

# Получить конкретную задачу
curl http://localhost:8000/api/tasks/task_20240101_120000

# Очистить кэш
curl -X POST http://localhost:8000/api/cache/clear
```

---

## Работа с кэшем

```bash
# Показать статистику кэша
adev cache stats

# Показать последние записи в кэше
adev cache list

# Показать 50 последних записей
adev cache list --limit 50

# Очистить весь кэш
adev cache clear
```

---

## Конфигурация

Файл конфигурации: `config/config.yaml`

### Основные настройки

```yaml
llm:
  default_model: "gpt-4"  # Модель по умолчанию
  
cost:
  max_cost_per_task: 5.0    # Лимит стоимости на задачу
  max_daily_cost: 20.0      # Лимит стоимости в день
  routing_strategy: "balanced"  # budget | balanced | premium
  fallback_to_local: true   # Переключаться на локальную модель при ошибке

cache:
  enabled: true
  ttl: 86400          # Время жизни кэша (24 часа)
  max_size_mb: 1000   # Максимальный размер кэша

orchestrator:
  max_retries: 3           # Попытки повтора при ошибке
  default_complexity: 3    # Уровень сложности (1-5)
  mode: "standard"         # fast-track | standard | enterprise
```

### Переменные окружения в конфиге

Конфиг поддерживает подстановку переменных окружения:

```yaml
llm:
  default_model: "${ADEV_DEFAULT_MODEL:-gpt-4}"  # Значение по умолчанию
  openai:
    api_key: "${OPENAI_API_KEY:-}"  # Обязательно установить
```

---

## Локальные модели (Ollama)

### Установка Ollama

```bash
# macOS / Linux
curl -fsSL https://ollama.ai/install.sh | sh

# После установки скачать модель
ollama pull llama3
ollama pull codellama
ollama pull mistral
```

### Запуск Ollama

```bash
# Запустить сервер Ollama
ollama serve

# В отдельном терминале проверить
ollama list
```

### Использование локальных моделей

```bash
# Через CLI
adev run "Сделать что-то" --model llama3

# Или изменить конфиг
# config/config.yaml:
# llm:
#   default_model: "llama3"
```

---

## Управление логами

Логи сохраняются в `./logs/adev.log`

```bash
# Просмотр логов в реальном времени
tail -f logs/adev.log

# Последние 100 строк
tail -n 100 logs/adev.log

# Изменить уровень логирования
export ADEV_LOG_LEVEL=DEBUG
adev run "..."
```

Уровни логирования: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`

---

## Структура проекта

```
/workspace/
├── autodev_agent/
│   ├── models/          # Модели данных и конфигурация
│   ├── orchestrator/    # Ядро оркестрации
│   ├── cache/           # Система кэширования
│   ├── sync/            # Синхронизация (Obsidian, Git)
│   ├── skills/          # Навыки агентов
│   ├── web/             # Web UI (FastAPI)
│   ├── utils/           # Утилиты
│   └── cli.py           # CLI интерфейс
├── config/
│   └── config.yaml      # Конфигурация
├── data/
│   ├── cache/           # Кэш запросов
│   ├── graph/           # Obsidian граф знаний
│   └── decisions/       # Журнал решений
├── logs/
│   └── adev.log         # Логи
├── tests/               # Тесты
└── pyproject.toml       # Зависимости
```

---

## Решение проблем

### OpenAI API Key не работает

```bash
# Проверить установку переменной
echo $OPENAI_API_KEY

# Проверить баланс аккаунта
# https://platform.openai.com/usage
```

### Ollama не доступна

```bash
# Проверить запущен ли сервер
curl http://localhost:11434/api/tags

# Перезапустить Ollama
ollama serve
```

### Ошибки кэша

```bash
# Очистить кэш
adev cache clear

# Проверить права доступа
ls -la data/cache/
```

### Web UI не запускается

```bash
# Проверить установлен ли uvicorn
pip show uvicorn

# Переустановить web зависимости
pip install -e ".[web]" --force-reinstall

# Проверить занятость порта
lsof -i :8000
```

---

## Примеры использования

### 1. Простая генерация кода

```bash
adev run "Написать функцию на Python для вычисления чисел Фибоначчи"
```

### 2. Создание модуля с тестами

```bash
adev run "Создать модуль calculator.py с функциями add, subtract, multiply, divide и покрыть tests/test_calculator.py"
```

### 3. Рефакторинг существующего кода

```bash
adev run "Проанализировать autodev_agent/orchestrator/core.py и предложить улучшения архитектуры"
```

### 4. Генерация документации

```bash
adev run "Создать README.md с описанием проекта, примерами использования и инструкцией по установке"
```

---

## Следующие шаги

После успешного запуска попробуйте:

1. ✅ Запустить простую задачу через CLI
2. ✅ Запустить Web UI и создать задачу через браузер
3. ✅ Проверить работу кэша (повторный запрос должен быть быстрее)
4. ✅ Посмотреть логи в `logs/adev.log`
5. ✅ Изучить созданные файлы в `data/graph/tasks/`

## Поддержка

При возникновении проблем:

1. Проверьте логи: `tail -f logs/adev.log`
2. Убедитесь, что все зависимости установлены: `pip list | grep autodev`
3. Проверьте переменные окружения: `env | grep -E "OPENAI|OLLAMA|ADEV"`
4. Попробуйте очистить кэш: `adev cache clear`
