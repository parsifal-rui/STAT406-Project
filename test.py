import numpy as np
from pca import PCA
from nmf import NMF
import matplotlib.pyplot as plt

def calculate_reconstruction_error(original, reconstructed):
    """计算重建误差（均方差）"""
    return np.mean((original - reconstructed) ** 2)

def test_algorithm(algorithm_class, training_data, ranks):
    """测试不同rank值下算法的重建误差"""
    errors = []
    for rank in ranks:
        # 初始化模型
        model = algorithm_class(n_components=rank)
        
        # 训练模型
        if algorithm_class == PCA:
            transformed = model.fit_transform(training_data)
            reconstructed = model.inverse_transform(transformed)
        else:  # NMF
            model.fit(training_data)
            W = model.transform(training_data)
            reconstructed = model.inverse_transform(W)
        
        # 计算误差
        error = calculate_reconstruction_error(training_data, reconstructed)
        errors.append(error)
        print(f"Rank {rank}: reconstruction error = {error:.6f}")
    
    return errors

def main():
    # 加载训练数据
    training_faces = np.loadtxt('train.txt')
    
    # 测试的rank值范围
    ranks = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    
    # 测试PCA
    print("Testing PCA...")
    pca_errors = test_algorithm(PCA, training_faces, ranks)
    
    # 测试NMF
    print("\nTesting NMF...")
    nmf_errors = test_algorithm(NMF, training_faces, ranks)
    
    # 绘制对比图
    plt.figure(figsize=(10, 6))
    plt.plot(ranks, pca_errors, 'b-o', label='PCA')
    plt.plot(ranks, nmf_errors, 'r-o', label='NMF')
    plt.xlabel('Rank')
    plt.ylabel('Reconstruction Error')
    plt.title('Reconstruction Error vs Rank')
    plt.legend()
    plt.grid(True)
    plt.savefig('reconstruction_errors.png')
    plt.close()
    
    # 保存结果到文本文件
    with open('test_results.txt', 'w') as f:
        f.write("Reconstruction Errors:\n")
        f.write("\nPCA:\n")
        for rank, error in zip(ranks, pca_errors):
            f.write(f"Rank {rank}: {error:.6f}\n")
        
        f.write("\nNMF:\n")
        for rank, error in zip(ranks, nmf_errors):
            f.write(f"Rank {rank}: {error:.6f}\n")
        
        # 计算平均误差
        f.write(f"\nAverage PCA Error: {np.mean(pca_errors):.6f}")
        f.write(f"\nAverage NMF Error: {np.mean(nmf_errors):.6f}")

if __name__ == "__main__":
    main() 