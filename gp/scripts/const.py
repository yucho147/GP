# likelihoods
gaussianlikelihood = {'GaussianLikelihood', 'GL'}
poissonlikelihood = {'PoissonLikelihood', 'PL'}
bernoullilikelihood = {'BernoulliLikelihood'}

regs = gaussianlikelihood | poissonlikelihood
clas = bernoullilikelihood

# mlls
variationalelbo = {'VariationalELBO', 'VELBO'}
predictiveloglikelihood = {'PredictiveLogLikelihood', 'PLL'}
gammarobustvariationalelbo = {'GammaRobustVariationalELBO', 'GRVELBO'}
exactmarginalloglikelihood = {'ExactMarginalLogLikelihood'}

# optimizers
adam = {'Adam'}
sgd = {'sgd'}
rmsprop = {'RMSprop'}
adadelta = {'Adadelta'}
adagrad = {'Adagrad'}
