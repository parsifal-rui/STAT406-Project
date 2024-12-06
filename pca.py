import numpy as np

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.mean_face = None
        self.eigen_vectors = None
        
    def fit(self, X):
        """
        参考hw3.Rmd中的实现，使用幂迭代法计算特征值和特征向量
        """
        # 计算均值脸
        self.mean_face = np.mean(X, axis=0)
        
        # 中心化数据
        centered_data = X - self.mean_face
        
        # 计算协方差矩阵
        Sigma = np.dot(centered_data.T, centered_data)
        
        # 初始化特征向量矩阵
        p = Sigma.shape[0]
        self.eigen_vectors = np.zeros((p, self.n_components))
        
        # 使用幂迭代法计算特征值和特征向量
        for k in range(self.n_components):
            # 随机初始化向量并归一化
            v = np.random.randn(p)
            v = v / np.sqrt(np.sum(v**2))
            
            # 幂迭代
            for _ in range(1000):
                v_new = np.dot(Sigma, v)
                v_new_norm = np.sqrt(np.sum(v_new**2))
                v_new = v_new / v_new_norm
                
                # 检查收敛性
                if np.sum((v_new - v)**2) < 1e-6:
                    v = v_new
                    break
                v = v_new
            
            # 保存特征向量
            self.eigen_vectors[:, k] = v
            
            # 从协方差矩阵中移除已找到的特征值和特征向量的贡献
            lambda_k = np.dot(np.dot(v.T, Sigma), v)
            Sigma = Sigma - lambda_k * np.outer(v, v)
            
        return self
    
    def transform(self, X):
        """将数据投影到特征向量上"""
        centered_data = X - self.mean_face
        return np.dot(centered_data, self.eigen_vectors)
    
    def inverse_transform(self, X_transformed):
        """从投影数据重建原始数据"""
        return np.dot(X_transformed, self.eigen_vectors.T) + self.mean_face
    
    def fit_transform(self, X):
        """拟合数据并返回转换后的结果"""
        self.fit(X)
        return self.transform(X) 