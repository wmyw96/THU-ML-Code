# Probabilistic Matrix Factorization

## Models and Implementation

### PMF (linear)

**Model**

Using the traditional PMF model described in (Salakhutdinov & Mnih, 2008) *Probabilistic matrix factorization*, which minimize the cost function

$$\sum_{i\le N, j \le M} {(R_{i,j}-U_i^TV_j)^2} + \lambda_U\|U\|^2 + \lambda_V\|V\|^2$$

Here $U_1,\cdots,U_N$ is user embedding defined in $\mathbb{R}^D$, $V_1,\cdots,V_M$ is movie embedding defined in $\mathbb{R}^D$, $N$ is the number of users, $M$ is the number of movies, $R_{i,j}$ represent the rating of user $i$ to movie $j$, which is selected from $\{1, 2, 3, 4, 5\}$.

We use stochastic gradient descent & Adam to optimize.

Moreover, we use $3.0$ to padding(predict) the non-existed users or movies in training set, and report the results with padding.

**Hyperparameters**

Here we set $\lambda_U=\lambda_V=0.01$, $D=30$, learning rate=0.005, $beta_1=0.5$ for Adam, batch size $=100,000$.

## Results

### Dataset Moivelens 1M

**Description**:

**Specification**: the whole user-film-rating tuple is randomly divided into 85%, 5% and 10% separately for training/validation/test. We use training dataset to train, use validation dataset to do model selection(only use early-stopping to perform regularization), and use test dataset to report results.

**Results**: 

| Model                    | Validation RMSE | Test RMSE | Time [per epoch] |  Epoches [best(total)]  |
| ------------------------ |:---------------:|:---------:|:------------:|:-----:|
| PMF (linear)             | 0.8597          | 0.8808    | 1.5 s  | 100(300) |
| PMF (linear, AHMC)       |                 |           |        |     |
| PMF (sigmoid)            |                 |           |        |     |
| PMF (sigmoid, AHMC)      |                 |           |        |     |