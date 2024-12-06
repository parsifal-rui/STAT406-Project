from flask import Flask, render_template, jsonify, request
import numpy as np
import time
from pca import PCA
from nmf import NMF

app = Flask(__name__)

# 读取训练集和测试集
training_faces = np.loadtxt('train.txt')
test_faces = np.loadtxt('test.txt')
current_index = 0
processed_faces = None
current_reconstructed = None

# 设置误差阈值
ERROR_THRESHOLD = 0.1  # 可以根据实际情况调整这个值

def calculate_error(original, reconstructed):
    """计算重建误差"""
    return np.mean((original - reconstructed) ** 2)

def is_face(error_rate):
    """根据重建误差判断是否是人脸"""
    return error_rate < ERROR_THRESHOLD

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    global processed_faces, current_reconstructed
    
    data = request.json
    algorithm = data.get('algorithm')
    rank = int(data.get('rank', 10))
    
    start_time = time.time()
    
    try:
        if algorithm == 'nmf':
            model = NMF(n_components=rank)
            model.fit(training_faces)
            W_test = model.transform(test_faces)
            processed_faces = model.inverse_transform(W_test)
        elif algorithm == 'pca':
            model = PCA(n_components=rank)
            model.fit(training_faces)
            transformed = model.transform(test_faces)
            processed_faces = model.inverse_transform(transformed)
        elif algorithm == 'autoencoder':
            pass
        
        runtime = time.time() - start_time
        current_reconstructed = processed_faces[current_index]
        error_rate = calculate_error(test_faces[current_index], current_reconstructed)
        
        return jsonify({
            'original': test_faces[current_index].tolist(),
            'reconstructed': current_reconstructed.tolist(),
            'runtime': runtime,
            'error_rate': float(error_rate),
            'is_face': bool(is_face(error_rate))
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/next')
def next_face():
    global current_index, current_reconstructed
    
    if processed_faces is None:
        return jsonify({'error': 'Please process images first'})
    
    current_index = (current_index + 1) % len(test_faces)
    current_reconstructed = processed_faces[current_index]
    error_rate = calculate_error(test_faces[current_index], current_reconstructed)
    
    return jsonify({
        'original': test_faces[current_index].tolist(),
        'reconstructed': current_reconstructed.tolist(),
        'error_rate': float(error_rate),
        'is_face': bool(is_face(error_rate))
    })

@app.route('/prev')
def prev_face():
    global current_index, current_reconstructed
    
    if processed_faces is None:
        return jsonify({'error': 'Please process images first'})
    
    current_index = (current_index - 1) % len(test_faces)
    current_reconstructed = processed_faces[current_index]
    error_rate = calculate_error(test_faces[current_index], current_reconstructed)
    
    return jsonify({
        'original': test_faces[current_index].tolist(),
        'reconstructed': current_reconstructed.tolist(),
        'error_rate': float(error_rate),
        'is_face': bool(is_face(error_rate))
    })

if __name__ == '__main__':
    app.run(debug=True) 