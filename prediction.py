import numpy as np
from numpy.random import multivariate_normal as mvnrnd
from scipy.stats import wishart
from scipy.stats import invwishart
from numpy.linalg import inv as inv
import scipy.io as scio
import matplotlib.pyplot as plt
import sys

def kr_prod(a, b):
    """
    Khatri-Rao product
    :param a:
    :param b:
    :return:
    """
    return np.einsum("ir, jr -> ijr", a, b).reshape(a.shape[0] * b.shape[0], -1)

def cov_mat(mat):
    """
    计算指定矩阵的协方差矩阵
    :param mat: 待计算矩阵
    :return: 协方差矩阵
    """
    dim1, dim2 = mat.shape
    new_mat = np.zeros((dim2, dim2))
    mat_bar = np.mean(mat, axis=0)
    for i in range(dim1):
        new_mat += np.einsum("i, j -> ij", mat[i, :] - mat_bar, mat[i, :] - mat_bar)
    return new_mat

def cp_combine(U, V, X):
    """
    CP分解的逆操作，将分解后的U，V，X还原成原张量
    :param U:
    :param V:
    :param X:
    :return: 还原后的张量
    """
    return np.einsum("is, js, ts -> ijt", U, V, X)

def ten2mat(tensor, mode):
    """
    按指定模式将张量展开成矩阵
    :param tensor: 待展开张量
    :param mode: 模式
    :return: 展开后的矩阵
    """
    return np.reshape(np.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1), order="F")

def mat2ten(mat, tensor_size, mode):
    """
    将矩阵按指定模式和尺度转化为张量
    :param mat: 待转化矩阵
    :param tensor_size: 张量尺度
    :param mode: 模式
    :return: 转化后的张量
    """
    index = list()
    index.append(mode)
    for i in range(tensor_size.shape[0]):
        if i != mode:
            index.append(i)
    return np.moveaxis(np.reshape(mat, list(tensor_size[index]), order="F"), 0, mode)

def mnrnd(M, U, V):
    """
    生成矩阵正态分布随机矩阵
    :param M: m*n matrix
    :param U: m*m matrix
    :param V: n*n matrix
    :return: 生成的随机矩阵
    """
    dim1, dim2 = M.shape
    X0 = np.random.rand(dim1, dim2)
    P = np.linalg.cholesky(U)
    Q = np.linalg.cholesky(V)
    return M + np.matmul(np.matmul(P, X0), Q.T)


def BTTF(sparse_tensor, init, rank, time_lags, maxiter1, maxiter2):
    """Bayesian Temporal Tensor Factorization, BTTF."""
    U = init["U"]
    V = init["V"]
    X = init["X"]

    d = time_lags.shape[0]
    dim1, dim2, dim3 = sparse_tensor.shape
    dim = np.array([dim1, dim2, dim3])
    # pos = np.where((dense_tensor != 0) & (sparse_tensor == 0))
    position = np.where(sparse_tensor != 0)
    binary_tensor = np.zeros((dim1, dim2, dim3))
    binary_tensor[position] = 1

    beta0 = 1
    nu0 = rank
    mu0 = np.zeros((rank))
    W0 = np.eye(rank)
    tau = 1
    alpha = 1e-6
    beta = 1e-6
    S0 = np.eye(rank)
    Psi0 = np.eye(rank * d)
    M0 = np.zeros((rank * d, rank))

    mat_hat = np.zeros((dim1, dim2, dim3 + 1))
    U_plus = np.zeros((dim1, rank))
    V_plus = np.zeros((dim2, rank))
    X_plus = np.zeros((dim3, rank))
    X_new = np.zeros((dim3 + 1, rank))
    X_new_plus = np.zeros((dim3 + 1, rank))
    A_plus = np.zeros((rank, rank, d))
    tensor_hat_plus = np.zeros((dim1, dim2, dim3 + 1))
    for iters in range(maxiter1):
        for order in range(2):
            if order == 0:
                mat = U.copy()
            elif order == 1:
                mat = V.copy()
            mat_bar = np.mean(mat, axis=0)
            var_mu_hyper = (dim[order] * mat_bar + beta0 * mu0) / (dim[order] + beta0)
            var_W_hyper = inv(inv(W0) + cov_mat(mat) + dim[order] * beta0 / (dim[order] + beta0)
                              * np.outer(mat_bar - mu0, mat_bar - mu0))
            var_Lambda_hyper = wishart(df=dim[order] + nu0, scale=var_W_hyper, seed=None).rvs()
            var_mu_hyper = mvnrnd(var_mu_hyper, inv((dim[order] + beta0) * var_Lambda_hyper))

            if order == 0:
                var1 = kr_prod(X, V).T
            elif order == 1:
                var1 = kr_prod(X, U).T
            var2 = kr_prod(var1, var1)
            var3 = (tau * np.matmul(var2, ten2mat(binary_tensor, order).T).reshape([rank, rank, dim[order]])
                    + np.dstack([var_Lambda_hyper] * dim[order]))
            var4 = (tau * np.matmul(var1, ten2mat(sparse_tensor, order).T)
                    + np.dstack([np.matmul(var_Lambda_hyper, var_mu_hyper)] * dim[order])[0, :, :])
            for i in range(dim[order]):
                inv_var_Lambda = inv(var3[:, :, i])
                vec = mvnrnd(np.matmul(inv_var_Lambda, var4[:, i]), inv_var_Lambda)
                if order == 0:
                    U[i, :] = vec.copy()
                elif order == 1:
                    V[i, :] = vec.copy()

        Z_mat = X[np.max(time_lags): dim3, :]
        Q_mat = np.zeros((dim3 - np.max(time_lags), rank * d))
        for t in range(np.max(time_lags), dim3):
            Q_mat[t - np.max(time_lags), :] = X[t - time_lags, :].reshape([rank * d])
        var_Psi = inv(inv(Psi0) + np.matmul(Q_mat.T, Q_mat))
        var_M = np.matmul(var_Psi, np.matmul(inv(Psi0), M0) + np.matmul(Q_mat.T, Z_mat))
        var_S = (S0 + np.matmul(Z_mat.T, Z_mat) + np.matmul(np.matmul(M0.T, inv(Psi0)), M0)
                 - np.matmul(np.matmul(var_M.T, inv(var_Psi)), var_M))
        Sigma = invwishart(df=nu0 + dim3 - np.max(time_lags), scale=var_S, seed=None).rvs()
        A = mat2ten(mnrnd(var_M, var_Psi, Sigma).T, np.array([rank, rank, d]), 0)
        if iters + 1 > maxiter1 - maxiter2:
            A_plus += A

        Lambda_x = inv(Sigma)
        var1 = kr_prod(V, U).T
        var2 = kr_prod(var1, var1)
        var3 = (tau * np.matmul(var2, ten2mat(binary_tensor, 2).T).reshape([rank, rank, dim3])
                + np.dstack([Lambda_x] * dim3))
        var4 = tau * np.matmul(var1, ten2mat(sparse_tensor, 2).T)
        for t in range(dim3):
            Mt = np.zeros((rank, rank))
            Nt = np.zeros(rank)
            if t < np.max(time_lags):
                Qt = np.zeros(rank)
            else:
                Qt = np.matmul(Lambda_x, np.matmul(ten2mat(A, 0), X[t - time_lags, :].reshape([rank * d])))
            if t < dim3 - np.min(time_lags):
                if t >= np.max(time_lags) and t < dim3 - np.max(time_lags):
                    index = list(range(0, d))
                else:
                    index = list(np.where((t + time_lags >= np.max(time_lags)) & (t + time_lags < dim3)))[0]
                for k in index:
                    Ak = A[:, :, k]
                    Mt += np.matmul(np.matmul(Ak.T, Lambda_x), Ak)
                    A0 = A.copy()
                    A0[:, :, k] = 0
                    var5 = (X[t + time_lags[k], :]
                            - np.matmul(ten2mat(A0, 0), X[t + time_lags[k] - time_lags, :].reshape([rank * d])))
                    Nt += np.matmul(np.matmul(Ak.T, Lambda_x), var5)
            var_mu = var4[:, t] + Nt + Qt
            if t < np.max(time_lags):
                inv_var_Lambda = inv(var3[:, :, t] + Mt - Lambda_x + np.eye(rank))
            else:
                inv_var_Lambda = inv(var3[:, :, t] + Mt)
            X[t, :] = mvnrnd(np.matmul(inv_var_Lambda, var_mu), inv_var_Lambda)

        if iters + 1 > maxiter1 - maxiter2:
            U_plus += U
            V_plus += V
            X_plus += X

        tensor_hat = cp_combine(U, V, X)
        if iters + 1 > maxiter1 - maxiter2:
            X_new[0: dim3, :] = X.copy()
            X_new[dim3, :] = np.matmul(ten2mat(A, 0), X_new[dim3 - time_lags, :].reshape([rank * d]))
            X_new_plus += X_new
            tensor_hat_plus += cp_combine(U, V, X_new)

        tau = np.random.gamma(alpha + 0.5 * sparse_tensor[position].shape[0],
                              1 / (beta + 0.5 * np.sum((sparse_tensor - tensor_hat)[position] ** 2)))
        # rmse = np.sqrt(np.sum((dense_tensor[pos] - tensor_hat[pos]) ** 2) / dense_tensor[pos].shape[0])
        # if (iters + 1) % 200 == 0 and iters < maxiter1 - maxiter2:
        #     print('Iter: {}'.format(iters + 1))
        #     print('RMSE: {:.6}'.format(rmse))
        #     print()

    U = U_plus / maxiter2
    V = V_plus / maxiter2
    X = X_plus / maxiter2
    X_new = X_new_plus / maxiter2
    A = A_plus / maxiter2
    tensor_hat = tensor_hat_plus / maxiter2
    # if maxiter1 >= 100:
    #     final_mape = np.sum(np.abs(dense_tensor[pos]
    #                                - tensor_hat[pos]) / dense_tensor[pos]) / dense_tensor[pos].shape[0]
    #     final_rmse = np.sqrt(np.sum((dense_tensor[pos] - tensor_hat[pos]) ** 2) / dense_tensor[pos].shape[0])
    #     print('Imputation MAPE: {:.6}'.format(final_mape))
    #     print('Imputation RMSE: {:.6}'.format(final_rmse))
    #     print()

    return tensor_hat, U, V, X_new, A


def OnlineBTTF(sparse_mat, init, time_lags, maxiter1, maxiter2):
    """Online Bayesain Temporal Tensor Factorization"""
    U = init["U"]
    V = init["V"]
    X = init["X"]
    A = init["A"]

    d = time_lags.shape[0]
    dim1, dim2 = sparse_mat.shape
    dim3 = 1
    sparse_tensor = sparse_mat.reshape([dim1, dim2, dim3])
    t = X.shape[0]
    rank = X.shape[1]
    position = np.where(sparse_mat != 0)
    binary_tensor = np.zeros((dim1, dim2, dim3))
    binary_tensor[position] = 1

    tau = 1
    alpha = 1e-6
    beta = 1e-6
    nu0 = rank
    W0 = np.eye(rank)
    var_mu0 = np.matmul(ten2mat(A, 0), X[t - 1 - time_lags, :].reshape([rank * d]))

    X_new = np.zeros((t + 1, rank))
    X_new_plus = np.zeros((t + 1, rank))
    tensor_hat_plus = np.zeros((U.shape[0], V.shape[0], t + 1))
    for iters in range(maxiter1):
        vec0 = X[t - 1, :] - var_mu0
        Lambda_x = wishart(df=nu0 + 1, scale=inv(inv(W0) + np.outer(vec0, vec0)), seed=None).rvs()

        var1 = kr_prod(V, U).T
        var2 = kr_prod(var1, var1)
        var_mu = tau * np.matmul(var1, ten2mat(sparse_tensor, 2).reshape([dim1 * dim2])) + np.matmul(Lambda_x, var_mu0)
        inv_var_Lambda = inv(
            tau * np.matmul(var2, ten2mat(binary_tensor, 2).reshape([dim1 * dim2])).reshape([rank, rank]) + Lambda_x)
        X[t - 1, :] = mvnrnd(np.matmul(inv_var_Lambda, var_mu), inv_var_Lambda)

        mat_hat = np.einsum('ir, jr, r -> ij', U, V, X[t - 1, :])

        tau = np.random.gamma(alpha + 0.5 * sparse_mat[position].shape[0],
                              1 / (beta + 0.5 * np.sum((sparse_mat - mat_hat)[position] ** 2)))

        if iters + 1 > maxiter1 - maxiter2:
            X_new[0: t, :] = X.copy()
            X_new[t, :] = np.matmul(ten2mat(A, 0), X_new[t - time_lags, :].reshape([rank * d]))
            X_new_plus += X_new
            tensor_hat_plus += cp_combine(U, V, X_new)

    X_new = X_new_plus / maxiter2
    tensor_hat = tensor_hat_plus / maxiter2

    return tensor_hat, X_new


def st_prediction(dense_tensor, sparse_tensor, pred_time_steps, rank, time_lags, maxiter):
    #sparse_tensor是densor_tensor经过处理后的稀疏张量（按missing_rate的稀疏比）
	# sparse_tensor是我们需要去处理的张量。densor_tensor是参照张量
    start_time = dense_tensor.shape[2] - pred_time_steps
    dense_tensor0 = dense_tensor[:, :, 0: start_time]
    sparse_tensor0 = sparse_tensor[:, :, 0: start_time]
    # sparse_tensor0 = sparse_tensor
    dim1 = sparse_tensor0.shape[0]
    dim2 = sparse_tensor0.shape[1]
    dim3 = sparse_tensor0.shape[2]
    tensor_hat = np.zeros((dim1, dim2, pred_time_steps))      #创建包含所有预测结果的零张量

    for t in range(pred_time_steps):
        if t == 0:
            init = {"U": 0.1 * np.random.rand(dim1, rank), "V": 0.1 * np.random.rand(dim2, rank),
                    "X": 0.1 * np.random.rand(dim3, rank)}
            tensor, U, V, X, A = BTTF(sparse_tensor0, init, rank, time_lags, maxiter[0], maxiter[1])
            X0 = X.copy()
        else:
            #U和V为位置维度，X为时间维度，因此只需对X进行迭代更新即可
            sparse_tensor1 = sparse_tensor[:, :, 0: start_time + t]
            # sparse_tensor1 = tensor_hat[:, :, 0: t]
            init = {"U": U, "V": V, "X": X0, "A": A}
            tensor, X = OnlineBTTF(sparse_tensor1[:, :, -1], init, time_lags, maxiter[2], maxiter[3])
            X0 = X.copy()
        tensor_hat[:, :, t] = tensor[:, :, -1]     #预测的结果存储到tensor_hat张量中
        if (t + 1) % 40 == 0:
            print('Time step: {}'.format(t + 1))

    small_dense_tensor = dense_tensor[:, :, start_time: dense_tensor.shape[2]]
    pos = np.where(small_dense_tensor != 0)
    final_mape = np.sum(np.abs(small_dense_tensor[pos] - tensor_hat[pos]) / small_dense_tensor[pos]) / small_dense_tensor[pos].shape[0]
    final_rmse = np.sqrt(np.sum((small_dense_tensor[pos] - tensor_hat[pos]) ** 2) / small_dense_tensor[pos].shape[0])
    print('Final MAPE: {:.6}'.format(final_mape))
    print('Final RMSE: {:.6}'.format(final_rmse))
    print()
    return tensor_hat

if __name__ == '__main__':
    # tensor = scipy.io.loadmat("E:\\transdim\\datasets\\NYC-data-set\\tensor.mat")
    # dense_tensor = tensor['tensor']
    # print(dense_tensor.shape)
    # print(len(np.where(dense_tensor == 0)[0]))
    # print(len(np.where(dense_tensor == 0)[0])/30/30/1464)
    #
    # rm_tensor = scipy.io.loadmat("E:\\transdim\\datasets\\NYC-data-set\\rm_tensor.mat")
    # rm_tensor = rm_tensor['rm_tensor']
    #
    # missing_rate = 0.1
    #
    # binary_tensor = np.round(rm_tensor + 0.5 - missing_rate)
    #
    # sparse_tensor = np.multiply(dense_tensor, binary_tensor)
    # print(sparse_tensor.shape)
    # print(len(np.where(sparse_tensor == 0)[0]))
    # print(len(np.where(sparse_tensor == 0)[0])/30/30/1464)

    data = scio.loadmat("E:\成山头数据\\data.mat")
    gridTensor = data["tensor"]
    pred_time_steps = 5     #预测的时间
    rank = 21       #秩
    time_lags = np.array([1, 2, 24])
    maxiter = np.array([200, 100, 200, 100])
    tensor_hat = st_prediction(gridTensor, gridTensor, pred_time_steps, rank, time_lags, maxiter)
    tensor_hat = np.log(tensor_hat)
    # print(tensor_hat[:,:,0])
    # print(tensor_hat[:,:,1])
    # print(tensor_hat[:,:,2])
    fig = plt.figure(figsize=(2, 2))
    ax = fig.add_subplot(3,2,1)
    ax2 = fig.add_subplot(3,2,2)
    bx = fig.add_subplot(3,2,3)
    bx2 = fig.add_subplot(3,2,4)
    cx = fig.add_subplot(3,2,5)
    cx2 = fig.add_subplot(3,2,6)
    ax.matshow(tensor_hat[:,:,0])
    ax2.matshow(gridTensor[:,:,43])
    bx.matshow(tensor_hat[:,:,1])
    bx2.matshow(gridTensor[:,:,44])
    cx.matshow(tensor_hat[:,:,2])
    cx2.matshow(gridTensor[:,:,45])
    plt.show()
