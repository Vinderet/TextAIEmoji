import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Conv1D, GlobalMaxPooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Глобальные переменные
data_path = ""  # Путь к файлу данных
model_lstm = None
model_cnn = None
tokenizer = None
le = None


# Предобработка текста (замените это на вашу реализацию)
def preprocess_text(text):
    # Приведение к нижнему регистру
    text = text.lower()

    # Удаление специальных символов
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Токенизация
    words = word_tokenize(text)

    # Удаление стоп-слов
    stop_words = set(stopwords.words('русский'))  # Подставьте свой язык
    filtered_words = [word for word in words if word.lower() not in stop_words]

    # Объединение токенов обратно в текст
    filtered_text = ' '.join(filtered_words)

    return ftext


# Функция обучения модели
def load_data_for_training():
    global data_path, model_lstm, model_cnn, tokenizer, le

    if not data_path:
        messagebox.showerror("Ошибка", "Укажите путь к файлу данных в настройках.")
        return

    try:
        # Загрузка данных
        df = pd.read_csv(data_path)

        # Предварительная обработка текста
        df['text'] = df['text'].apply(preprocess_text)

        # Разделение данных на обучающую и тестовую выборку
        X_train, X_test, y_train, y_test = train_test_split(df['text'], df['emotion'], test_size=0.2, random_state=42)

        # Преобразование меток в числовой формат
        le = LabelEncoder()
        y_train_encoded = le.fit_transform(y_train)
        y_test_encoded = le.transform(y_test)

        # Токенизация текста
        max_words = 10000  # Максимальное количество слов для токенизации
        tokenizer = Tokenizer(num_words=max_words)
        tokenizer.fit_on_texts(X_train)

        # Преобразование текста в числовой формат
        X_train_seq = tokenizer.texts_to_sequences(X_train)
        X_test_seq = tokenizer.texts_to_sequences(X_test)

        # Построение последовательности одинаковой длины
        max_len = 100  # Максимальная длина последовательности
        X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post')
        X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding='post')

        # Обучение LSTM модели
        model_lstm = Sequential()
        model_lstm.add(Embedding(max_words, 128, input_length=max_len))
        model_lstm.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
        model_lstm.add(Dense(7, activation='softmax'))
        model_lstm.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model_lstm.fit(X_train_pad, y_train_encoded, epochs=5, batch_size=64, validation_split=0.1)

        # Обучение CNN модели
        model_cnn = Sequential()
        model_cnn.add(Embedding(max_words, 128, input_length=max_len))
        model_cnn.add(Conv1D(128, 5, activation='relu'))
        model_cnn.add(GlobalMaxPooling1D())
        model_cnn.add(Dense(7, activation='softmax'))
        model_cnn.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model_cnn.fit(X_train_pad, y_train_encoded, epochs=5, batch_size=64, validation_split=0.1)

        messagebox.showinfo("Обучение", "Модели успешно обучены.")

    except Exception as e:
        messagebox.showerror("Ошибка", f"Возникла ошибка при загрузке данных: {str(e)}")


# Создание GUI
root = tk.Tk()
root.title("Emotion Analysis System")

# Создание вкладок
tab_control = ttk.Notebook(root)
tab_train = ttk.Frame(tab_control)
tab_settings = ttk.Frame(tab_control)
tab_usage = ttk.Frame(tab_control)

tab_control.add(tab_train, text="Обучение")
tab_control.add(tab_settings, text="Настройки")
tab_control.add(tab_usage, text="Использование")

tab_control.pack(expand=1, fill="both")

# Во вкладке "Настройки"
lbl_data_path = tk.Label(tab_settings, text="Путь к файлу данных:")
lbl_data_path.grid(column=0, row=0, padx=10, pady=10)

data_path_entry = tk.Entry(tab_settings, width=40)
data_path_entry.grid(column=1, row=0, padx=10, pady=10)


def choose_file():
    global data_path
    file_path = filedialog.askopenfilename(title="Выберите файл данных", filetypes=[("CSV files", "*.csv")])
    if file_path:
        data_path_entry.delete(0, tk.END)
        data_path_entry.insert(0, file_path)


btn_choose_file = tk.Button(tab_settings, text="Выбрать файл", command=choose_file)
btn_choose_file.grid(column=2, row=0, padx=10, pady=10)

# Во вкладке "Обучение"
btn_train_model = tk.Button(tab_train, text="Обучить модель", command=load_data_for_training)
btn_train_model.grid(column=0, row=0, padx=10, pady=10)

# Во вкладке "Использование"
lbl_input_text = tk.Label(tab_usage, text="Введите текст:")
lbl_input_text.grid(column=0, row=0, padx=10, pady=10)

input_text = scrolledtext.ScrolledText(tab_usage, width=40, height=10)
input_text.grid(column=1, row=0, padx=10, pady=10)

lbl_output_result = tk.Label(tab_usage, text="Результат:")
lbl_output_result.grid(column=0, row=1, padx=10, pady=10)

output_result = tk.Entry(tab_usage, width=40, state="readonly")
output_result.grid(column=1, row=1, padx=10, pady=10)

# Визуализация результата с цветовой разметкой
lbl_visualization = tk.Label(tab_usage, text="Визуализация результата:")
lbl_visualization.grid(column=0, row=2, padx=10, pady=10)

visualization_canvas = tk.Canvas(tab_usage, width=300, height=30)
visualization_canvas.grid(column=1, row=2, padx=10, pady=10)

# Запуск основного цикла
root.mainloop()
