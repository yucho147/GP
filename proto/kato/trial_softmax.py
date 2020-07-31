import numpy as np
import matplotlib.pyplot as plt

from gp.scripts.base_ApproximateGP import RunApproximateGP
from gp.utils.utils import tensor_to_array

#  作図参考:https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html 


def main():
    # データ作成
    from sklearn.datasets import load_iris
    data = load_iris()
    X, y = data.data, data.target
    X1 = np.vstack((X[:, :1]))  #sepal length(ガクの長さ)を取得
    X2 = np.vstack((X[:, 1:2])) #sepal width(ガクの幅)を取得
    X3 = np.vstack((X[:, 2:3])) #petal length(花弁の長さ)を取得
    X4 = np.vstack((X[:, 3:4])) #petal width(花弁の幅)を取得
    y_onehot = np.eye(3)[y]
    input_data = X[:,0:2]
    # import ipdb; ipdb.set_trace()

    run = RunApproximateGP(inducing_points_num=50,
                           kernel='RBFKernel',
                           likelihood='SoftmaxLikelihood')
    run.set_model(input_data, y_onehot, lr=3e-2, batch_size=300)
    run.fit(10, verbose=True)
    # prob, (predicts_mean, predicts_std) = run.predict(input_data)

    x_min, x_max = input_data[:, 0].min() - .5, input_data[:, 0].max() + .5
    y_min, y_max = input_data[:, 1].min() - .5, input_data[:, 1].max() + .5
    mesh_num = 10
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, mesh_num),
                         np.linspace(y_min, y_max, mesh_num))

    prob, (predicts_mean, predicts_std) = run.predict(
        np.c_[xx.ravel(), yy.ravel()]
        )
    # import ipdb; ipdb.set_trace()
    Z = prob.probs
    Z = tensor_to_array(Z)
    Z = Z.reshape(xx.shape)

    from matplotlib.colors import ListedColormap
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)
    ax.scatter(input_data[:, 0], input_data[:, 1], c=y,
               cmap=cm_bright, edgecolors='k', alpha=0.6)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    plt.show()


if __name__ == '__main__':
    main()
