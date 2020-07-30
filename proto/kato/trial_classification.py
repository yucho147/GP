import numpy as np
import matplotlib.pyplot as plt

from gp.scripts.base_ApproximateGP import RunApproximateGP
from gp.utils.utils import tensor_to_array

#  作図参考:https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html 


def main():
    # データ作成
    from sklearn.datasets import make_moons
    X, y = make_moons(noise=0.3, random_state=0)
    input_data = X

    run = RunApproximateGP(inducing_points_num=100,
                           kernel='RBFKernel',
                           likelihood='BernoulliLikelihood')
    run.set_model(input_data, y, lr=3e-2, batch_size=2000)
    run.fit(25, verbose=True)
    # prob, (predicts_mean, predicts_std) = run.predict(input_data)

    x_min, x_max = input_data[:, 0].min() - .5, input_data[:, 0].max() + .5
    y_min, y_max = input_data[:, 1].min() - .5, input_data[:, 1].max() + .5
    mesh_num = 30
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
