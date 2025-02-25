### Описание
Простой чат-интерфейс, построенный на Gradio, который поддерживает работу с моделями GigaChat и OpenAI. Позволяет настраивать параметры генерации текста и переключаться между моделями.

### Установка

1. Клонируйте репозиторий
2. Создайте виртуальное окружение `uv venv` (если у вас нет uv, установите его через `pip install uv`)
3. Активируйте виртуальное окружение (смотрите подасказку в терминале после предыдущего пункта)
4. Скопируйте ключи из файла `.env.sample` и создайте файл `.env`, куда и вставьте эти ключи со значениями (если у вас нет ключа от openai, можете оставить пустую строку, а также не вставлять прокси)
5. Запустите приложение `python -m main`
