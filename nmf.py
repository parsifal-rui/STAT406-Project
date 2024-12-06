import numpy as np

class NMF:
    def __init__(self, n_components, max_iter=100, tol=1e-4):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.W = None
        self.H = None
        
    def fit(self, X):
        """
        使用乘法更新规则实现NMF
        """
        n_samples, n_features = X.shape
        
        # 随机初始化W和H
        self.W = np.random.rand(n_samples, self.n_components)
        self.H = np.random.rand(self.n_components, n_features)
        
        for _ in range(self.max_iter):
            # 更新H
            numerator = np.dot(self.W.T, X)
            denominator = np.dot(np.dot(self.W.T, self.W), self.H) + 1e-10
            self.H *= numerator / denominator
            
            # 更新W
            numerator = np.dot(X, self.H.T)
            denominator = np.dot(np.dot(self.W, self.H), self.H.T) + 1e-10
            self.W *= numerator / denominator
            
        return self
    
    def transform(self, X):
        """返回W矩阵"""
        return self.W
    
    def fit_transform(self, X):
        """拟合数据并返回W矩阵"""
        self.fit(X)
        return self.W
    
    def inverse_transform(self, W):
        """重建原始数据"""
        return np.dot(W, self.H) 