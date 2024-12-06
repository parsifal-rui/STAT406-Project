from flask import Flask, render_template, jsonify, request
import numpy as np
import time
from pca import PCA
from nmf import NMF

app = Flask(__name__)

# 读取训练集和测试集 - 修改为新的数据文件
training_faces = np.loadtxt('pp.txt')
test_faces = np.loadtxt('test2.txt')

# 数据预处理：将像素值从0-255缩放到0-1
training_faces = training_faces / 255.0
test_faces = test_faces / 255.0

current_index = 0
processed_faces = None
current_reconstructed = None

# 设置误差阈值 - 由于数据范围改变，可能需要调整阈值
ERROR_THRESHOLD = 0.1

# 在全局变量部分添加
image_states = {}  # 用于记录每张图片的状态

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
    global processed_faces, current_reconstructed, image_states
    
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
            'is_face': bool(is_face(error_rate)),
            'image_state': image_states.get(current_index, {}),
            'total_images': len(test_faces)
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
        'is_face': bool(is_face(error_rate)),
        'image_state': image_states.get(current_index, {})  # 返回当前图片的状态
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
        'is_face': bool(is_face(error_rate)),
        'image_state': image_states.get(current_index, {})  # 返回当前图片的状态
    })

# 添加新的路由来更新图片状态
@app.route('/update_state', methods=['POST'])
def update_state():
    data = request.json
    index = data.get('index')
    state = data.get('state')
    
    if index is not None:
        if index not in image_states:
            image_states[index] = {'viewed': False, 'status': None}
        image_states[index].update(state)
    
    return jsonify({'success': True})

if __name__ == '__main__':
    app.run(debug=True) 