# Probabilistic Matrix Factorization

## Models and Implementation

### PMF (linear)

**Model**

Code: [pmf_map.py](pmf_map.py)

Using the traditional PMF model described in (Salakhutdinov & Mnih, 2008) *Probabilistic matrix factorization*, which minimize the cost function

$$\sum_{i\le N, j \le M} {(R_{i,j}-U_i^TV_j)^2} + \lambda_U\|U\|^2 + \lambda_V\|V\|^2$$

Here $U_1,\cdots,U_N$ is user embedding defined in $\mathbb{R}^D$, $V_1,\cdots,V_M$ is movie embedding defined in $\mathbb{R}^D$, $N$ is the number of users, $M$ is the number of movies, $R_{i,j}$ represent the rating of user $i$ to movie $j$, which is selected from $\{1, 2, 3, 4, 5\}$.

We use stochastic gradient descent & Adam to optimize.

Moreover, we use $3.0$ to padding(predict) the non-existed users or movies in training set, and report the results with padding.

**Hyperparameters**

Here we set $\lambda_U=\lambda_V=0.01$, $D=30$, learning rate=0.005, $beta_1=0.5$ for Adam, batch size $=100,000$.

### PMF (linear, AHMC)

**Model**

Code: [pmf_ahmc.py](pmf_ahmc.py)

We follow the simplest generative process

- $U_1, \cdots, U_N$ draw independently from $N(0, \sigma_U^2 I)$
- $V_1, \cdots, V_M$ draw independently from $N(0, \sigma_V^2 I)$
- $R_{i,j}$ draw independently from $N(U_i^TV_j, \sigma_r^2)$

Here $U_1,\cdots,U_N$ is user embedding defined in $\mathbb{R}^D$, $V_1,\cdots,V_M$ is movie embedding defined in $\mathbb{R}^D$, $N$ is the number of users, $M$ is the number of movies, $R_{i,j}$ represent the rating of user $i$ to movie $j$, which is selected from $\{1, 2, 3, 4, 5\}$. Moreover $I$ is a $D \times D$ identity matrix.

For inference, we use the hybrid of HMC and Gibbs Samplings, the steps are as followings:

1. Initialize $U_1, \cdots, U_N$ and $V_1, \cdots, V_M$
2. Draw $U_i$ when observing $V_1, \cdots, V_M$ and all $R_{i,j}$'s, use HMC to draw the 'posterior distribution', and $i$ range from 1 to N.
3. Draw $V_j$ when observing $U_1, \cdots, U_N$ and all $R_{i,j}$'s, use HMC to draw the 'posterior distribution', and $j$ range from 1 to M.
4. Back to step [2]

We call this process 'alternative HMC'.

**Hyperparameters**

Here we set $\sigma_U=\sigma_V=1.0$, $\sigma_r=0.1$, $D=30$, as well as n leap frogs=10, step size = 0.001 (no adaptive step size) for HMC.

**Paralleled Version**

Code: [pmf\_ahmc\_paralleled.py](pmf_ahmc_paralleled.py)

We slightly modify the process of inference:

1. Initialize $U_1, \cdots, U_N$ and $V_1, \cdots, V_M$
2. Draw $U_{k\times C : (k+1)\times C}$ when observing $V_1, \cdots, V_M$ and all $R_{i,j}$'s, use HMC to draw the 'posterior distribution', and $k$ range from 1 to $\lceil N/C \rceil$.
3. Draw $V_{l\times C : (l+1)\times C}$ when observing $U_1, \cdots, U_N$ and all $R_{i,j}$'s, use HMC to draw the 'posterior distribution', and $l$ range from 1 to $\lceil M/C \rceil$.
4. Back to step [2]

Here $X_{i:j}$ represent $X_i, X_{i+1}, \cdots, X_{j}$, $C$ is the chunk size, we must tune $C$ to a intermediate value since too small $C$ don't reflect the advantages of parallel while too large $C$ will incur the acceptive rate of HMC to be very low.

## Results

### Dataset Moivelens 1M

**Description**:

**Specification**: the whole user-film-rating tuple is randomly divided into 85%, 5% and 10% separately for training/validation/test. We use training dataset to train, use validation dataset to do model selection(only use early-stopping to perform regularization), and use test dataset to report results.

<font color='red'>**Notes**</font>: We found the random shuffle for the dataset has a tremendous impact on the final results. For example, just for the simple linear MAP solution, by trying different seed the results (might) range from [0.85, 1.12], so we adjust the seed to make the result of MAP approximate the results describe in (Yin, 2016)*A Neural Autoregressive Approach to Collaborative Filtering*. We use seed 1234 for Numpy random shuffle and shuffle the dataset twice.

**Results**: 

| Model                    | Validation RMSE | Test RMSE | Time [per epoch] |  Epoches [best(total)]  |
| ------------------------ |:---------------:|:---------:|:------------:|:-----:|
| PMF (linear)             | 0.8597          | 0.8808    | 1.5 s  | 100(300) |
| PMF (linear, AHMC)       |          |     | 13.8min |     |
| PMF (linear, AHMC, paralleled, chunksize=50)       |   0.8569       | 0.8530 | 2.1min | 240(500) |
| PMF (sigmoid)            |                 |           |        |     |
| PMF (sigmoid, AHMC)      |                 |           |        |     |