import matplotlib
matplotlib.use('Agg')
import librosa
import matplotlib.pyplot as plt
from flask import Flask, render_template, request
import os
import uuid
import numpy as np
from werkzeug.utils import secure_filename
import librosa.display

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_unique_filename(file):
    filename, extension = os.path.splitext(file.filename)
    random_string = str(uuid.uuid4())  # Генерируем случайную строку
    return secure_filename(random_string + extension)

def extract_voice_features(file):
    y, sr = librosa.load(file)
    pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr)
    pitch = pitches.mean()
    return pitch, y, sr

def compare_voices(file1, file2):
    print("Comparing voices...")
    print("File 1:", file1)
    print("File 2:", file2)

    pitch1, y1, sr1 = extract_voice_features(file1)
    pitch2, y2, sr2 = extract_voice_features(file2)
    distance = abs(pitch1 - pitch2)

    print("Distance:", distance)

    # Визуализация спектрограммы
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    librosa.display.specshow(librosa.amplitude_to_db(librosa.stft(y1), ref=np.max), sr=sr1, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram of File 1')

    plt.subplot(1, 2, 2)
    librosa.display.specshow(librosa.amplitude_to_db(librosa.stft(y2), ref=np.max), sr=sr2, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram of File 2')

    plt.tight_layout()
    spectrogram_path = os.path.join(STATIC_FOLDER, 'spectrograms.png')
    plt.savefig(spectrogram_path)  # Сохранение спектрограммы в статический файл
    plt.close()

    return distance



@app.route('/', methods=['GET', 'POST'])
def index():
    distance = None
    error = None
    file1_info = None
    file2_info = None
    identical_content_message = None
 #   comparison_path = None

    if request.method == 'POST':
        file1 = request.files['file1']
        file2 = request.files['file2']

        if file1.filename == '' or file2.filename == '':
            error = 'Please select both files.'
        elif not allowed_file(file1.filename) or not allowed_file(file2.filename):
            error = 'Unsupported file format. Please select WAV, MP3, or OGG files.'
        else:
            # Генерация уникальных имен файлов
            file1_filename = secure_filename(file1.filename)
            file2_filename = secure_filename(file2.filename)

            file1_path = os.path.join(app.config['UPLOAD_FOLDER'], file1_filename)
            file2_path = os.path.join(app.config['UPLOAD_FOLDER'], file2_filename)
            file1.save(file1_path)
            file2.save(file2_path)

            distance = compare_voices(file1_path, file2_path)

            # Проверка на идентичное содержимое файлов
            if distance == 0:
                identical_content_message = "Files have identical content."

            # Получение информации о файлах
            file1_info = {'name': file1.filename, 'size': os.path.getsize(file1_path)}
            file2_info = {'name': file2.filename, 'size': os.path.getsize(file2_path)}

            # Удаление загруженных файлов после обработки
            if os.path.exists(file1_path):
                os.remove(file1_path)
            if os.path.exists(file2_path):
                os.remove(file2_path)

    return render_template('index.html', distance=distance, error=error, file1_info=file1_info, file2_info=file2_info, identical_content_message=identical_content_message)

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    if not os.path.exists(STATIC_FOLDER):
        os.makedirs(STATIC_FOLDER)
    app.run(debug=True)
