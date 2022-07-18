# 数据集加载
from matplotlib import pyplot as plt
from sklearn import datasets

# 加载数据
iris = datasets.load_iris()
# help(datasets.load_iris)  # Check the data struture

X = iris.data # Each row is an observation, each col is a variable
y = iris.target
print(" The shape of X and y ", X.shape, y.shape,"\n",
    "Features are ", iris.feature_names, "\n", "Classes are ", iris. target_names)
# The shape of X and y  (150, 4) (150,)
#  Features are  ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
#  Classes are  ['setosa' 'versicolor' 'virginica']


# 数据预处理
from sklearn.preprocessing import StandardScaler
X_std = StandardScaler().fit_transform(X)

# 协方差矩阵构造
import numpy as np
X_train = X_std # Create a reference to the origin
X_mean = np.mean(X_train, axis=0)   # A vector with size (4,)
n = X_train.shape[0]
X_cov = (X_train - X_mean).T @ (X_train - X_mean) / n
X_cov   # A symmetric matrix with size (4,4)

# 特征值分解
# 计算协方差矩阵的特征值、特征向量
import numpy as np
eig_val, eig_vec = np.linalg.eig(X_cov)
# 检验得到的是否为单位正交矩阵，即每一列的norm和为1
for ev in eig_vec.T:
    np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))
# 这里直接用assert，标准过高，浮点误差会报错
# numpy自带的assert almost，默认精度为六位，可设置参数decimal修改
# 方法二，也可使用u,s,v = np.linalg.svd(X_std.T)，eig_vec = u
print("Eigen values\n", eig_val, "\nEigen vectors\n", eig_vec)
# Eigen values
#  [2.93808505 0.9201649  0.14774182 0.02085386]
# Eigen vectors
#  [[ 0.52106591 -0.37741762 -0.71956635  0.26128628]
#  [-0.26934744 -0.92329566  0.24438178 -0.12350962]
#  [ 0.5804131  -0.02449161  0.14212637 -0.80144925]
#  [ 0.56485654 -0.06694199  0.63427274  0.52359713]]

# 方差解释
tot = sum(eig_val)
var_exp = [(i / tot)*100 for i in sorted(eig_val, reverse=True)]
cum_var_exp = np.cumsum(var_exp) # Return a cumulative sum
print("Cumulative sum \n", cum_var_exp)


# 降维投影
# 根据上述的分析，选用前两个主成分进行降维
proj_mat = np.hstack((eig_vec[:, 0].reshape(-1,1),
                      eig_vec[:, 1].reshape(-1,1)))
print('Projection matrix\n', proj_mat)
Y = X_std @ proj_mat # Y即是我们最终需要的降维结果
# Projection matrix
#  [[ 0.52106591 -0.37741762]
#  [-0.26934744 -0.92329566]
#  [ 0.5804131  -0.02449161]
#  [ 0.56485654 -0.06694199]]


# 降维结果
target_names = ['setosa','versicolor','virginica']
with plt.style.context('seaborn-whitegrid'):
    plt.figure(figsize=(6, 4))
    for lab, col in zip((0, 1, 2),
                        ('blue', 'red', 'green')):
        plt.scatter(Y[y==lab, 0],
                    Y[y==lab, 1],
                    label=target_names[lab],
                    c=col)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(loc='lower center')
    plt.tight_layout()
    plt.show()
