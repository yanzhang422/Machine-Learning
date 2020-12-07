# ML3 - Basic probability
##  The axioms
The following two laws are the key axioms of probability:
1. $P(\phi) = 0 \leq p(A) \leq 1 = P (\Omega)$
$\phi$ means no event is selected, $\Omega$ means everything. So probability is measure between 0 - 1.
3. For **disjoint sets** $A_n, n \geq 1$, we have
$$ P (\sum_{n=1}^{\infty} A_n) = \sum_{n=1}^{\infty} P(A_n)$$

### Or and And operations
Given two events, A and B, that are not disjoint, we have:
$P(A \space or \space B) = P(A)+P(B)-P(A \space and \space B)$
$P(A \space and \space B)$ is intersection.

#### Conditional probability
Assuming (given that) B has been observed (i.e. there is no uncertainty about B), the following tells us the probability that A will take place:
$$P(A \space given \space B) = P(A \space and \space B) / P(B)$$

That is, in the frequentist interpretation, we calculate the ratio of the number of times both A and B occurred and divide it by the number of times B occurred.
For short we write: $P(A|B) = P(AB)/P(B)$; or $P(AB)=P(A|B)P(B)$, where $P(A|B)$ is the conditional probability, $P(AB)$ is the joint, and $P(B)$ is the marginal.
If we have more events, we use the chain rule:
$$P(ABC)=P(A|BC)P(B|C)P(C)$$

#### Conditional probability
Can we write the joint probability, $P(AB)$, in more than one way in terms of conditioal and marginal probabilities?
$$P(AB) = P(A|B)P(B)$$

$$P(AB) = P(B|A)P(A)$$

### Independence
We know that:
$$P(AB) = P (A|B)P(B)$$

But what happens if $A$ does not depend on $B$? That is , the value of $B$ does not affect the chances of $A$ taking place. How does the above expression simplify?

#### Conditional probability example
Assume we have a dark box with 3 red balls and 1 blue ball. That is, we have the set $\{r, r, r, b\}$. What is the probability of drawing 2 red balls in the first 2 tries?
$P(B_1 =r, B_2 =r) =P(B_2 =r|B_1=r) P(B_1=r)$ = $\frac{2}{3}$ x $\frac{3}{4}$ =$\frac{1}{2}$

#### Marginalization
Let the sets $B_{1:n}$ ($B_1, B_2,...B_n$) be disjoint and $\bigcup_{i=1}^{n} B_i = \Omega$ Then
$$P(A) = \sum_{i=1}^n P(A, B_i) = P(AB_1) + P(AB_2) + P(AB_n)$$
### Conditional probability exaple
What is the probability that the $2^{nd}$ ball drawn from the $set \space \{r,r,r,b\}$ will be red?

Using marginalization, 

$P(B_2=r) = P(B_2=r, B_1=r) + P(B_2=r, B_1=b)$ = $\frac{3}{4}$x$\frac{2}{3}$ + $\frac{1}{4}$ x $\frac{3}{3}$ = $\frac{1}{2}$ + $\frac{1}{4}$ = $\frac{3}{4}$

Because you don't know what happens to the $1^{st}$ ball, but you still can assumps the $2^{nd}$, you just need sum all the possible values. 

### Matrix notation
Assume that $X$ can be 0 or 1. We use the math notation: $X\in\{0,1\}$.
$Let \space P(X_1=0) = \frac{3}{4} \space and \space P(X_1=1) = \frac{1}{4}$. Assume too that $P(X_2=1|X_1=0) = \frac{1}{3}, P(X_2=1|X_1=1) = 0$. Then, $P(X_2=0|X_1=0) = \frac{2}{3} , P(X_2=0|X_1=1) = 1$

We can obtain an expression for $P(X_2)$ easily using matrix notation:
$(X_1 = [\frac{3}{4}\space\space \frac{1}{4}])$   x $\begin{bmatrix}\frac{2}{3} & \frac{1}{3} \\\\1 & 0\end{bmatrix}=$ $(X_2=[\frac{3}{4}\space\space \frac{1}{4}])$

The $X_2$ agrees with the previous slide (e.g. $P(B_2=r) = \frac{3}{4}$), pick up the the $2^{nd}$ time in red with $P(\frac{3}{4})$, in blue in $P(\frac{1}{4})$

Componentwise matrix vector product is:
$$\sum_{X_1\in\{0,1\}} P(X_1) P(X_2|X_1) = P(X_2)$$
So we can say when the $2^{nd}$ time in red like

$P(B_2=r) = \sum_{B_1} P(B_2=r, B_1) = \sum_{{B_1}\in\{r,b\}} P(B_2=r, B_1) = \sum_{B_1}  P(B_2=r|B_1) P(B_1)$

$\sum_{B_1}  P(B_2=r|B_1) P(B_1)$ it is the joint conditional rule

Support 
$\Pi_1 = [\frac{3}{4}\space\space \frac{1}{4}]$, it is size of 1 x 2

$G =\begin{bmatrix}\frac{2}{3} & \frac{1}{3} \\\\1 & 0\end{bmatrix}$ , it is size of 2 x 2

$\Pi_2=[\frac{3}{4}\space\space \frac{1}{4}]$, it is size of 1 x 2

For short,, we write this using vectors and a stochastic matrix:
$$\Pi_1^T G = \Pi_2^T \equiv \Pi_2(j) = \sum_{i=1}^2 \Pi_1(i) G(i,j)$$ 
$\Pi_1^TG= \Pi_2^T, Pi_2^TG= \Pi_3^T, Pi_3^TG= \Pi_4^T, ... \Pi_k-1^TG= \Pi_k^T \equiv \Pi_1^TG^{K-1}= \Pi_K^T$ 

 $if \space \Pi_K = \Pi, then \space \Pi^TG= \Pi_T^T$

**Claim**: For a very large number $k$, after $k$ iterations, the value of $\pi$ stabilizes.

That is, $\pi_k$ is an **eigenvector** of $G$ with **eigenvalue** 1.  

$$AX = \lambda X$$
$A$ reprsents $\pi_k$, $X$ represents $G$, $\lambda$ represent 1

### Bayes rule
Bayes rule enables us to reverse probabilities:
$$P(A|B) = \frac{P(B|A)P(A)}{P(B)} =\frac{P(B|A)P(A)} {\sum_A P(B|A)P(A)}$$

It is deducted by:
$$P(AB) = P(B|A)P(A) = P(A|B)P(B)$$

$$=$$

$$P(B|A)P(A)=P(A|B)P(B)$$
$$\sum_B P(B|A) = 1$$

Because A is given, there is no uncertainty of array and if you sum of probability, the probabilities must sum to 1, you are summing over B. 

### Learning and Bayesian inference
$$ p(h|d) = \frac{p(d|h)p(h)}{\sum_{h'\in H} p(d|h')p(h')}$$

#### Problem 1: Diagnoses
The test is 99% accurate: $P(T=1|D=1) = 0.99$ and $P(T=0|D=0) = 0.99$ Where $T$ denotes test and $D$ denotes disease.

The disease affects $1$ in $10000$: $P(D=1) = 0.0001$

$P(D=1|T=1) = \frac{P(T=1|D=1)P(D=1)}{P(T=1|D=0)P(D=0)+P(T=1|D=1)P(D=1)}$
$$=\frac{0.99*0.0001}{0.01*0.9999+0.99*0.0001}$$ $$=0.0098$$

#### Problem 2: Monty Hall
i) All doors (Hypotheses) have equal probability of having prize:
$$P(H=1) = P(H=2) = P(H=3) = \frac{1}{3}$$

ii) Contestant **chooses door 1**.
Let's think. If the prize is truly behind door 1, the host is indifferent and will choose 2 or 3 with equal probability. If the prize is behind door 2 (or 3), host chooses 3 (or 2).
$$P(D=2|H=1) = \frac{1}{2}, \space P(D=3|H=1) = \frac{1}{2}$$

$$P(D=2|H=2) = 0, \space P(D=3|H=2) =1$$

$$P(D=2|H=3) = 1, \space P(D=3|H=3) =0$$

iii) The host opens door 3 (**D=3**), revelaing a goat behind the door.
We use Bayes rule to computer the probability of the hypothesis that the prize is behind door $i$ (for $i=1,2,3$) given that the host has opened door 3 ($D=3$). That is we compute $P(H=i|D=3).$
$P(H=1|D=3) = \frac{P(D=3|H=1)P(H=1)}{P(D=3)} = \frac{(1/2)(1/3)}{P(D=3)} = \frac{1/6}{1/6+1/3} = \frac{1}{3}$
$P(H=2|D=3) = \frac{P(D=3|H=2)P(H=2)}{P(D=3)} = \frac{(1)(1/3)}{P(D=3)} = \frac{1/3}{1/6+1/3} = \frac{2}{3}$
$P(H=3|D=3) = \frac{P(D=3|H=3)P(H=3)}{P(D=3)} = \frac{(0)(1/3)}{P(D=3)} = 0$

Since $P(H=2|D=3) > P(H=1|D=3)$, the contestant should switch.
**Note:** I used the fact that 
$$\sum_{i=1}^3 P(H=i|D=3) = P(H=1|D=3) + P(H=2|D=3) + P(H=3|D=3) = 1$$ to normalize.
An Alternative is to compute 
$$P(D=3) = \sum_i P(H=i, D=3) = \sum_i P(D=3|H=i)P(H=i)$$ $$P(D=3|H=1)P(H=1) + P(D=3|H=2)P(H=2) + P(D=3|H=3)P(H=3)$$ $$= \frac{1}{2}*\frac{1}{3}+1*\frac{1}{3}+0*\frac{1}{3}=\frac{1}{6} + \frac{1}{3} + 0$$

### Bayes and decision theory
![Utilitarian](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/Utilitarian%20Example.PNG)

## Probabilistic graphical models
### The curse of dimensionality
This curse tells us that to represent a joint distribution of $d$ binary variables, we need $2^d-1$ terms!

![Ball Probability](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/Ball%20probabilistic.PNG)

The conditional probability is the sum of condition ($B_1 = r \space or \space B_1= b$)equal to 1, the joint probability is sum of all ($B_1 = r \& B_1= b$) equal to 1.

#### Directed Acyclic Graph (DAG)
![DAGe](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/DAG.PNG)

Drop the variables from the probabilities which have no influence of the ancesters.

![Joint vs Factorized](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/Joint%20and%20factorization.PNG)

We only need 9 variables, cause the other varialbe could be get from 1- probabilities. If we have 100 problems, base on dimension rule, it will be $2^100$, then how many variables will be need?

#### 3 cases of conditional independence to remember
![condition independence](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/condition%20independent.PNG)

Rain and Shower all make wet, let's assume you wet, in order to know whether it's raining or not. If you are wet in the shower, then you might (not) web because of the rain, but if you are not in the shower and you're wet that is probably because you're outside. So the **Rain (R) is not independent of Shower (S) given Wet (W)**

Recommend reading from company !{bayesia}(www.bayesia.com)

The world is very stochastic, can't be predicted, so the lack of probability made expert systems. 

#### Markov blankets
![markov blankets](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/makrov%20blanket.PNG)

In above image, the x_i represents **NOT $x_i$**, it consists of all the notes R, consist of all the notes that are parents of $x_i$ as U, consist of all the children of $x_i$ as Y, consist of all the notes that are the sort of the partners of $x_i$ as Z. R is the notes that ouf of the range, they don't have directly connect to $x_i$, so it can have 0 influence of probability of $x_i$, it could be remove from condition probality rule of the $x_i$

### Inference in probabilistic
#### The spinkler network
$P(S|C=1) = [0.9, 0.1]$, it is the distribution over S,  so $\sum_s P(S|C=1) =1$

![sprinkler networks](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/sprinkler%20networks.PNG)

$P(C,S) = P(S|C)P(C)$ = $\begin{array}{|c|c|} \hline {0.5*0.5}&{0.5*0.5} \\\\ \hline  {0.9*
0.5}&{0.1*0.5} \\\\ \hline \end{array}$ = $\begin{array}{|c|c|} \hline {0.25}&{0.25} \\\\ \hline  {0.45}&{0.05} \\\\ \hline \end{array}$

#### Inference in DAGs
Let us use 0 to denote false and 1 to denote true.
What is the marginal probability, $P(S=1)$, that the sprinkler is on?
$$P(S=1) = \sum_{C=0}^1\sum_{R=0}^1\sum_{W=0}^1 P(C,R,W,S=1)$$ If you don't know how may variables, you just sum them all, and sometimes on the paper reading, there is another way to write, you need understand they are present sum of all possibilities.
$$P(S=1) = \sum_{C}\sum_{R}\sum_{W} P(C)P(S=1|C)P(R|C)P(W|S=1,R)$$ Bifferentiating the oecause $C$ dose not have the parents, so it just variable, and there are $2^3$ terms possible for the combination of C, R, W. 

![brute force approach](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/Brute%20force%20approach.PNG)

There are $2^3$ x 3 floating-point operations for the computer, so even the small exercise is taking quite a lot of time, the problem will also be exponention hard.

![brute force coding method](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/brute%20force%20coding%20method.PNG)

#### Smart approach: variable elimination
**aka dynamic programming, aka distrubitive law**

And when you do brute force approach in code, it will take a lots of nested loops, so brute force approach is not good way. We can take the terms out which is multiple duplicated times, just takes it one time, just like $ab + ac$ to be $a(b+c)$. 

![smart approach](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/Smart%20approach.PNG)

In code, there are three variables $\psi, \phi, \theta$ is defined. 

![smart coding approach](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/smart%20coding%20approach.PNG)

The problem is reduced from exponential to linear computation.

We won't implement the general code in this course. To do this one needs to learn about the **junction tree** data structure. This structure, once created, enables us to conduct any query on the graph very efficiently.

These exact algorithms work well for small graphs and for graphs that are **trees** or close to trees (have low tree-width). For large densely connected graphs we require the use of algorithms beyond the scope of this course. One of those algorithms is called **Gibbs sampling**.

## Hidden Markov Models (HMM)
### Assistive technology
![Assistive technology](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/Assistive%20technology.PNG)

Assume that an expert has compiled the following **prior** and **likelihood** models:

![Baysian rule](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/Baysian%20rule.PNG)

$X$ is given, $Y$ is probability. Distribution over $Y$ with four values, the distribution of a $Y$ sum to 1. 

![sequence of observation](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/sequence%20observation.PNG)

$X_1$ indicates the time before, $X_2$ indicates the time after. If you are happy in time before and after, you may probability happy at this moment. The same probability theory of sad. This describes how our emotional state evolves over time. The $X_0$ could be the initial emtional state, how often happy or sad at the beginning. But if given the sequence of observations, eg. ($Y_1=w, Y_2=f, Y_3=c$), what suppose the state to be happy or sad?

#### Dynamic model
![HMM chain](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/HMM%20chain.PNG)

Each $Y$ has only one parent which is $X$, each $X$ has only one parent which is preceding $X$. So you have emotional state, then evolved to another emotional state, then it exhibits certain traits. This is HMM. $P(X_0, X_1, X_2, Y_1, Y_2) is the probability of full joint, we care about what is the probability of X given a sequence of observations. 

And why we call it is hidden, because there are things we can observe and things that we can't observe, so from the things we observe we try to infer the things we can't observe.

![Optimal Filtering](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/Optimal%20HMM%20coding.PNG)

Each of the $Y$ is given, the $X$ is random value. The table given the output $Y$ how sad and how happy it is. You have previous $Y$ in the memory, so there is affect pass to your current state.

So if given the previous state, the current state could be predicted. So if I have the previous one, then I can computer the new one, then basically can computer all of them. So even if start with a guess that's not good, the best possible case you will still be able to eventually do predict well.

### Prediction
![prediction](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/prediction.PNG)

$P(X_{t-1}|Y_{1:t-1}) is the one what we have, but it is a table, not a probability. Given $X_{t-1}$, it is not depend on $Y_{t-1}$, because **given your parents, you become independent**, it's outside of the Markov blank. So we can drop $\sum_{x_{t-1}} P(X_t|X_{t-1}, Y_{1:t-1})$ and rewrite it.

#### Bayes rule revision
Generalization of Bayes rule:
$$P(A|B,C) = \frac{P(B|A,C) P(A|C)}{P(B|C)}$$

![generation of bayes rule](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/Generalization%20of%20bayes%20rule.PNG)

Given $X_t$, the $Y_t$ is independent of the past, so $Y_{1:t-1}$ could be dropped. 

### HMM algorithm
Each of steps are broken into two sub steps which is prediction and update, prediction and update, ... and this is HMM algorithm.

## Discrete random variables
![discrete random variable](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/discrete%20random%20variables.PNG)

$X$ is the variable, when you are in the event {2, 4, 6}, then the $X=even$. So the mapping is , if the $\omega$ is 2, then the X is even (0)

![probability distribution](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/probability%20distribution.PNG)

The notion, $X$ is the variable, $x_i$ is the $X$ value could be taken.

### CDF (cumulative distribution function)
![CDF](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/CDF.PNG)

The cumulative is very useful when you want to simulate from the distribution.

### Expectation
![expection](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/Expectation.PNG)

Sum over all the values, they are random variable weighted by their probability. Do example for indicator variable, it is precisely to 1 if W is an elment of the set A, if W is not an element of the set A, then **the indicator variable** is equal to 0. So the indicator as the name says just indicates whether W belongs to A or not.

Question: What is $E(I_A(w))$ equal to?
$$E[I_A(w)] = \sum_{w\in\{A, not (A)\}} I_A(w) P(w)$$ 

$$ E[I_A(w)] = 1 * P(\omega \in A) + 0 * P(\omega \notin A) = P(\omega \in A) = P (A)$$

The probability of element in set A, that you'll get to 4 or 6 is just the probability of even, which is the probability of set A. So the expectation of an event happening of in this case, let's say A is rain, the expection of rain is the probability of rain. So in english, when we say expectation of event, is the probability of event.

### Bernoulli: a model for coins
We could not sure everything the varialbe will have binary result, so introduce the parameter $\theta$ when we unknown the objective of learning is.

![Bernoulli](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/Bernoulli.PNG)

$P(X|\theta) = \theta^x (1-\theta)^{1-x}$ because if x equal 1, the $(1-\theta)^{1-x}$ will get 0, if the x equal 0, the $\theta^x$ will get 1, the $(1-\theta)^{1-x}$  will get $1-\theta$.

#### Bernoulli expectation

![bernoulli expectation](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/Bernoulli%20expectation.PNG)

The probability of it being 1, the expecation probability is the same.

#### What is the variable of a Bernoulli variable?
![bernoulli variance](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/Bernouli%20variance.PNG)

The expectation $X$ minus the mean of X, ($(X- E(X|\theta))^2|\theta$), that just the expectation of X minus $\theta$ square ($(X-\theta)^2|\theta$)
$$\frac{d(\theta-\theta^2)}{d\theta} = 1- 2\theta = 0$$

And the derivative is equal to 0, then it is the slope of a function, and **its zero at the maximum**. If solve for $\theta$, then find $\theta = \frac{1}{2}$, then we will find the $var(X|\theta = \frac{1}{2}) = \frac{1}{2} - \frac{1}{4} = \frac{1}{4}$. 

**What does that mean intuitively?**
It means if my coin is unbiased, there is a lot of uncertainty as what it's going to be. If I tell you very biased coin, put a lot of head on the heads side, so the probability of tailsnow is 0.9. The variance is saying if the coin remember this parameter is telling us the bias of the coin. If the $\theta$ becomes 1/2, the variance become max, if the $\theta$ becomes 0 or 1, the variance is at least. $\theta$ is the probability $0\leq\theta\leq1$. 

The variance essentially quantifies uncertainty.

#### N independent tosses
Learning is try to minimal uncertainty, you entered this world and there's a lot of stuff you don't know, and what you do is try to minimize the things that you don't know. Minimizing uncertainty or very equivalently maximizing information.

Gaming information is means minimizing uncertainty.

![N indepedent tosses](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/N%20independent%20tosses.PNG)

What is the distribution of N independent coin tosses? The probability of A and B is just a product of the probability of A times the probability of A and so if two coins flips are indepent conclude $x_1$ and $x_2$ can write joint distribution, the 2 x 2 table its just a product of the individual tables, when you have n tables, you will have $2^n$ tables here, and you have n tables of size 2. Indepedence is just like conditional independence, like an extreme form of the condition independence, so that saying the graphic model is of this form. There are n variables, there's no connections. Since the variables have no parenets, so just the probability of the variables. And the variables is the X, not the $\theta$, $\theta$ is sort of parameter.

### Maximum likelihood
The maximum likelihood is the first strategy for learning. 

#### Frequentist learning
Assue you don't know what the true bias when I put on one side of the coin, so instead you guess it, when you guess it, the guess will be $\hat\theta$.

![Frequentist learning](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/Frequentist%20learning.PNG)

Suppose we have n coin tosses, it tries to find the $\theta$ that makes those {$x_1, x_2, ... x_n$} most probable.  

We have many of machine learning papers, the first equation on the first page is 
$$\hat\theta = arg\space max P(x_{1:n} | \theta)$$  We can think there is an object that depends on $\theta$, so the computer program that has $x_1$ inside, you pass a value $\theta$, the function returns $P(x_{1:n} | \theta)$. The location of the max is what we call the $arg \space max$ which is in this case it's equal to $\hat\theta$. So $arg \space max$ means the argument that maximizes. As just the picture shows $\hat\theta$ is just that $\theta$ for which the curve $P(x_{1:n} | \theta)$ is at its location of the maximum. 

![frequence learning example](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/frequence%20learning%20example.PNG)

Because 0.99 is much close to 1, so $\hat\theta = 0.99$ has much probability in the example for $\theta$. There is an tale about black swan, after seeing all of the white swan in Europe and build the model $\hat\theta = 0.99$ white swan which near to 100% only white swan in the world, after moving to Australian, there is a black swan is found, so the statistical model (100% white swan model) fails. 

### Maximum Likelihood procedure
![maximum likelihood procedure](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/maximum%20likelihood%20procedure.PNG)

Step 1: Make some assumptions over all the sequence events / data, example for those data assume is independent. 

Step 2: Define the log-likelihood, log (AB) = log A + log B, the reason transform to log is the next step which is differentiate and equate to zero $l(\theta)$ is much **easier to do in log space** than in the orginal space. So **log space is just  a trick** to **make the next part step easier**.

Step 3: Differential and equate to zero to find the estimate of $\theta$

We start with a likelikhood for all the data to joint, we factorize it, if we did not assume they (data) are independent, then it (**data is dependent**) would **factorize as a graphical model**. So if the data is independent, then you take the log, you differential any greate to zero. **Most** of things we do in **machine learning follow this formula** **except for Bayesian learning** which will be very different. But we always follow this recipe. 

There is other recipes, but this one about 70% of all papers are there use this formula. 

#### Maximum likelihood procedure example
**Bernoulli MLE** 
![Bernoulli MLE Step 1 and 2](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/Bernoulli%20MLE%20example-1.PNG)

![Bernoulli MLE Step 3](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/Bernoulli%20MLE%20example-2.PNG)

Finally we equate to zero to solve for $\theta$. Then we will get $\theta$ = m/n, maximal likelihood gives you right thing. So one way to remember likelihood, is to remember the procedure. Firstly, you write down the distribution, 2nd, you take the log, differentiate the and equal to zero. So if you have a complex graphical model as we'll see, write down its distribution, the probability of the nodes given its parent. Coming out of the model and designing good model is the hard thing. When you have the model, the procedure is much mechanical, **your write down the probability, you differentiate it, the equate to zero and you got the answer**. 

In this example we have the single independent distribution for each input $x_i$, later  there is continuous dependent distribution, it has the tricky. 

It is important to know why maximum likelihood works, and how some likelikhood relates, up to now it just looks mechanically, just a math, so to understand it, its a good idea to first introduce a concept entropy.

### Entropy
![entropy](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/entropy.PNG)

Entropy is a measure of uncertainty just like variance, its a negative of information. Information concept how these guys came to be. So by increasing uncertainty, you decrease information. 

### MLE advanced
For independent and identically distributed (i.i.d) data from $p(x|\theta_0)$, the MLE minimize the **Kullback-Leiber divergence**:

![MLE advanced](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/MLE%20advanced.PNG)

# ML 12 - Bayesian learning

## Maximum likelihood revision
Very important for learning, very quickly to revise what we've said about maximum likelihood, we said what the principle tries to do is, try to find $\theta$ by maximizing $P (data|\theta)$. 

![maximum likelihood revision](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/maximum%20likelihood%20revision.PNG)

Above is the operational mechanism, that's what we do in practice.

It is the minimize the difference between the average log probabilities, which is the data produced by our model, and the data produced by that we've actually observed in the real world. 

![maximum likelihood revision example](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/maximum%20likelihood%20revision%20example.PNG)

From the input X=111101100111, we try to guess an estimate $\hat\theta$, you never observed $\theta_0$ which produces the data, image it is a hidden being behind a door, and it will produce ones and zeros depending on what that $\theta_0$ is, it keeps slip and give you ones and zeros, from ones and zeros you gonna have to guess $\theta_0$, we can it is $\hat\theta$, and the good guess is $\theta$ maximum likelihood, with this $\hat\theta$, you can produce data too, we can it is sample data, it $\widetilde{x}$ which generated by the model. 

A good $\hat\theta$ is the one that's close to $\theta_0$, but because we don't know $\theta_0$, so we will never be able to know whether we have the right $\theta$ , so instead what you do is, choose $\hat\theta$ to minimize some difference between $|f(x) - f(\widetilde{x})|$, for maximum likelihood, the f happens to be the sum of the log probabilities that's the statistic that maximum likelihood uses which has interpretation also being the negative entropy. 

There are many ways you can say,
1) f just the number of ones
2) f could be the number of one's of x versus the number of one's of $\widetilde{x}$

If two agree, then you have learned. You observe the data x and if you have the data x, you can guess a $\hat\theta$, from that $\hat\theta$ which basically a model, from that model you can sample data, then you compare the data that you sample against the data that you observe if the two are the same, you are in good shape. 

In comparing the data, we will not compare the whole world to the whole world, we try to come up with a summary of the world that summary is f, f has different names, summary, statistic, pattern, and so what we're trying to make sure it that the patterns of the world are the same as the patterns that our model is capable of replicating. 

For now we're not going to use this procedure for learning. In neuroscience, we do use such procedure actually, so it is important to actually understand this procedure. 

Maximum likelihood is consistent what that means the $\hat\theta$ as the number of data grows to infinity always approaches $\theta_0$, if you have infinity data, you will get it. Practice we do not have infinite data, so that's problematic. 

In statistic theorem, the **maximum likelihood has the lowest variance**, for the difference of $\hat\theta$ and $\theta_0$, they follow the **central limit theorem** which means there distribute the distribution of the difference is a Gaussian and the **Gaussian has the zero mean** because they converse to be the same, the difference between the two guys is the variance and maixmum likelihood can provably be shown to have the lowest variance which is call the Cramer Rao bound (CRB) statistics. If you have **less data**, the **maximum likelihood can actually get you into the wrong conclusions**. 

### Bayesian learning procedure

![bayesian learning procedure](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/Bayesian%20learning%20procedure.PNG)

We have the data records, so the following steps:

Step 1: we have to specify a model, for now our model which be the $\theta^m (1-\theta)^{n-m}$ for a coin. In maximan likelihood in terms of specification here is done, the next step is differentiating with zero. The Bayesian need specify one more thing is prior, the prior captures our prior beliefs and it catpures the main knowledge.

Step 2: Specify a prior: $p(\theta)$

Do example for the bridge crash, you may not base on the number of the crashes event, but you do the prior about bridges base on your main knowledge that make judgment. 

Step 3: Computer the posteriror: 

The prior is very power, and posterior is likely hit times prior normalized, instead of maximizing. **Bayesian** reasoning is very different, **it requires multiplying a likelihood** and **a prior**, **then** it requires that we **normalize**.  

**In maximum likelihood**, the $x$ is the uncertainty, you **only specify p of x given** $\theta$, so **the distribution is on the data**, $\theta$ is not a random variable. 

**In Bayesian, the $\theta$ is the frequency given x, who is the frequentist** beliefs there is a true $\theta$ called $\theta_0$, we don't know what it is but we should guess it, they can only be one truth or not. This is the frequencies way of thinking $\theta$ is not a random variable, the only randomness is in the data. Bayesian says no $\theta$ is random variable, that's why we're going to specify a distribution this random, we are going to call it the prior, and then as we observe data, we update this prior and it becomes a posterior $p(\theta|x_{1:n})$ 

### Posterior vs Marginal likelihood

![posterior vs marginal likelihood](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/posterior%20and%20marginal%20likelihood.PNG)

$P(A) = \sum_B P(AB)$ is the marginalization, $P(A) = \int P(AB) dB$ is an integral, the integral is just a limit of sum which is essentially sums, so all the properties that we had for sums apply natural to integrals. So we can do the chain rule, do marginalization, do dynamic programming, successive conditioning, etc.  

We using integral remove the variable $theta$, just left with $x_{1:n}$, the other change is that we replace the sum by an integral. 

#### Bayesian learning for coin model
Step 1
![bayesian learning for coin model-step1](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/Bayesian%20learning%20for%20coin%20model-step%201.PNG)

Step 2
![bayesian learning for coin mode-step2](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/Bayesian%20learning%20for%20coin%20model-step%202.PNG)
$\theta$ is a continuous variable and it has to be $0 \leq \theta \leq 1$ it is follow the probability of $\theta = P (x_i =1)$ for any i. 

How can we represent the probability of $\theta$? Which is the suitable model?

Can we use the Bernouli distribution represent $\theta$?  
No, because **Bernoulli only talks about things that are 0 or 1**. $\theta$ is not just 0 or 1, it is any number between 0 and 1. 

Can we use Gaussian?
No, **Gaussian** it extends over the whole real line so you would say variables **could be any number between minus infinity and infinity**. So Gaussian will be the wrong model. 

**Uniform is a beautiful choice. Because uniform is between 0 and 1, and it's continuous**. So we can use that prior we would be done. It used a lot in NLP, text mining, bioinformatic, so it is call beta. And the beta is a sub case of a distribution that will introduced later called the parish lag. 

Log is the sum series, and if that series becomes infinite, fact it's an integral, so log is actually defined as an integral as well. 

Gamma $\Gamma$ function is just in the computer, you say give me gamma of two  and it turns out. 

The Gammas are just functions of hyperparmeters Alpha $\alpha$ and Beta $\beta$, itis not functions of $\theta$, so you know they are the normalizing constant. We never need to know $P(X)$, and the normalization constant, because some one already worked that out for us, and we know it is Gamma $\Gamma$ function. 

![beta distribution](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/beta%20distribution.PNG)

The uniform is prior, the uniform saying that I believe any $\theta$ has equal probability, so $\theta =2$ has the same probability of $\theta=1$ So you can believe the $\theta$ is not uniform, it could much more likely to be 0.8, which case the probability of coin being 1 is much higher. The Beta distribution is a way of producing many curves to describe your prior beliefs. 

So in the coin example, you can belief the $\theta$ much more like 0.5.

Prime recursion perspective is also interesting, because $\theta$ is probability and P of $\theta$ is probability on probabilities, so getting into this recursive thing, this is what's greatful language,  because language has recursion. So these prior becomes very useful because they can handle that sort of recursion and be able to specify probabilities on probabilities. 

Step 3
![bayesian learning for coin mode-step3](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/Bayesian%20learning%20for%20coin%20model-step%203.PNG)
$\theta^m (1-\theta)^{n-m}$ times the prior $\theta^{\alpha-1}(1-\theta)^{\beta-1}$, you can computer the posterior without knowing what the values of alpha and beta are.  Because all the terms group very nicely. 

What's that constant? What will make it sum to 1? It's the gammas again, 
$$P(\theta) = \frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)}$$ It will make the integral sum to 1, because
$$\int P(\theta)d\theta = 1 = \int \frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)} \theta^{\alpha-1} (1-\theta)^{\beta-1} d\theta$$

So we can do the derivation without doing any integrals
$$P(\theta|x_{1:n}) = \frac{\Gamma(\alpha'+\beta')}{\Gamma(\alpha')\Gamma(\beta')} \theta^{\alpha'-1} (1-\theta)^{\beta'-1} $$ But if you choose the integrals, it will also work obviously, but you will do a lot of work with algebra. 

Beta distribution sum to 1, by just accepting its definition as a valid distribution, we are able computer the posterior without having to do any calculations. We just essentially need to add exponents.

#### Example for posterior calculation base on $\alpha$ and $\beta$

![Example of beta distribution](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/Example%20of%20Beta%20distribution.PNG)

The answer for $\hat\theta$ is 1 after observed six ones divided by six point that is 1.  In Beta distribuiton we using the prior to calcuate the prosterior, the Baysian says you specifiy a prior if you believe a prior that it is.

More learning can from [Beta distribution wikipedia](https://en.wikipedia.org/wiki/Beta_distribution)

The mode is the peak where is the highest point.  The variance is how wide the distribution is.

# ML 13 - Learning Bayesian Networks
![beta probability density curve](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/Beta%20probability%20density%20curve.png)

For the coin is easy, it just Bernoulli and a beta distribution. $\theta$ is the probability of X equal one, $P(X) = 1$. Then we putting prior/distribution on $\theta$ which means that we're putting a distribution on a distribution that the coin will be equal to one. The probability is between 0 to 1, because the $\theta$ is probability must be between 0 to 1. When we have the curve which is higher than 1 on y axes, that is called density. The area under the curve is 1, so probability is the area under the curve. 

Probability is the density times something that you can measure a length. The probabability is hitting the continuous to zero. $\theta^m (1-\theta)^{n-m}$ is the likelihood, $\theta^{\alpha-1}(1-\theta)^{\beta-1}$ is the prior, $\propto$ is the proportion symbol. 

So we integral the $\theta$ get,
 $$E(\theta|x_1) = \int\theta^{\alpha'-1}(1-\theta)^{\beta-1}d\theta = \frac{\Gamma(\alpha')\Gamma(\beta')}{\Gamma(\beta'+\alpha')}$$
 
![beta rule in coins](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/beta%20rule%20in%20coins.PNG)

The $\alpha'$ = m+$\alpha$, $\beta'$ = n-m+$\beta$, so the first coin flip is $x_1$ =1, the $P(\theta|x_1=1)$ = Beta (2,1), the 2= (# flip m =1) + (prior $x_1$ =1), the 1 = (total # flip n = 1) - (#flip m=1) + (prior $x_1$ =1).

$E(\theta|x_1)$ = 2/2+1 = 2/3 

#### Learning Bayes nets example

![example data](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/learning%20bayes%20net%20example-data.PNG)

It has 5 observations. Next, we choose a model describing how we believe each variable influences the other:

![specify the model](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/learning%20bayes%20net%20example-specify%20model.PNG)

We don't know the probability of being thin given studying and Fritz, but should be inside of those tables, but using the maximal likelihood or Bayes learning we will be able to fill in the tables, and once we fill in the tables, we can answer questions like what's the probability that you're thin given that you had a martini, what's the probability that you had a martini given that you are studying.

How many graphs could be written down? Exponential numbers. So it is **very hard about model selection**.

So here assume the structure is given, and the task is to learn the parameters. We can model this as Bernoulli variable with parameter $\theta$, and then we specify the bernoulli.

![probability value](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/learning%20bayes%20net%20example-probability%20value.PNG)

Given the binary observations, we use bernoulli distributions to describe the probabilities of each of the variable in the bayes net. There are 4 times the Martini equal to 1, and 1 time the Martini equal to 0, so $P(M|\theta) = \theta^4(1-\theta)^1$, the following parameters are following the bernoulli distrubtion rules.

![maximum likelihood value](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/learning%20bayes%20net%20example-ml%20value.PNG)

How do we learn the parameters? We using the maximum likelihood $$\hat\theta_{ML} = \frac{\#1's}{\#tries/\#measurements}$$. Base on this we can get all the parameters, then we can fill in all those tables. 

Firstly observe data and then you use maximum likelihood or Bayesian to fill in those tables and once fill in the tables, then do what we did before, we do inference, computer the things like what's the probability of martini given Frites, and so on.  And hat we did before. we use independence and all of those things we did before. 

![beta prior](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/learning%20bayes%20net%20example-beta%20prior.PNG)

What Bayesian does, we specify Beta priors for each of the variables. Then we mulitple these prior times the Bernoulli likelihoods to derive the Beta posteriors.

Setting the prior $P(\alpha) = Beta(10,1)$ which says with very high probability you will be study. Following the steps, then do $\beta$, $\Gamma$, etc, you will get all the parameters for Bayesian learning.

![inference](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/learning%20bayes%20net%20example-inference.PNG)

Inference with the learned net
once we have the prior value to get all the parameters learning, then we can have the inference for the probability of each questions.

![two different model choosing](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/learning%20bayes%20net%20example-two%20different%20model.PNG)

When using different models, we will have different fractions. And also could be given new data base sets. Then compare which model gives you higher probability, then that's the best model. 

# ML 14 - Linear algebra revision for machine learning

When we manipulate many variables, the matrix is the good way to handle them. 

## Matrix Multiplication
![matrix multiplication](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/matrix%20multiplication.PNG)

One bar to indicate a vector (lower caser), two bars to indicate a matrix (upper case). And the linear algebra says that the answer is a linear combination of the column of a.

Here is a lemma very useful, it is  for google converges:
If $X^T G = Y^T$ and $\sum_i X_i =1$ and $\sum_jG_{ij}=1$ for all i, then $\sum_j Y_j =1$

## Eigen value and Eigen matrix
A matrix times the vetor is going to be equal to a scalar times the vetor, $Ax=\lambda x$. What doest that say, the matrix is a lot of number, if you multiple times a vector that's the same thing that if you just take this one single number $\lambda$ multiple the vector.  At left side you have a lot of number, at the right side you have only one number. 

This is very profound of learning, because learning to a large extend is about compression, there's a big world out there, we managed to compress a lot of what the world's about into very tiny amount of space. So when we see mathematical expression where you have a lot of numbers on one side and just a few on the other side, there things are of interest, because they give us an opportunity to do data compression. 

The identity $I$ times a vector is still effect, so it will be
$Ax=\lambda x$ => $Ax - \lambda x = 0$ => $Ax - \lambda I x =0$ => $[A-\lambda I]x=0$ 
The reason introduce identity $I$, because if I don't then the dimensions of $A$ and $\lambda$ wouldn't match, due to $\lambda$ is scalar, $A$ is matrix. 

Below is the determinant of the matrix $A$, and how we do find the $\lambda$.

$\begin{vmatrix} 1-\alpha-\lambda & \alpha \\\\ \beta & 1-\beta-\lambda \end{vmatrix}$ = $(1-\alpha-\lambda) (1-\beta-\lambda) - \alpha\beta = 0$ 

We can choose following $\lambda$ value to get the determiant to be zero.
$\lambda_1=1$ ,  $\lambda_2$ = $1-\alpha-\beta$

Typically we use computer to do eigenvectors, as its impossible to get them exactly when do the matrix multiple by hand.

So if the $\lambda =1$, the $Ax=1x$ => $(A-1I)x=0$
which means $\begin{vmatrix}-\alpha & \alpha \\\\ \beta&-\beta \end{vmatrix}$ $\begin{vmatrix} x_1 \\\\ x_2 \end{vmatrix}$ = $\begin{vmatrix}0\\\\0\end{vmatrix}$ => $x = \begin{vmatrix} 1\\\\1 \end{vmatrix}$

The $x$ is the eigenvector. 

Left eigenvector is more intereting, eg
$\begin{vmatrix}x_1&x_2\end{vmatrix}$ $\begin{vmatrix}-x&\alpha\\\\ \beta&-\beta \end{vmatrix}$ = $\begin{vmatrix}0&0\end{vmatrix}$

So we will have :
$-\alpha x_1 + \beta x_2 = 0$
$\alpha x_1 - \beta x_2 = 0$

We have $x = \begin{vmatrix} \frac{\beta}{\alpha+\beta}&\frac{\alpha}{\alpha+\beta}\end{vmatrix}$,and we have $\sum_i x_i =1$

### Simple proof of Google's PageRank
 It will help you understand how can easily manipuate eigenvalues to actually be able to discuss convergence algorithms.

![Google pages Algo](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/google%20pages%20algorithm.PNG)
 
 Google algorithm:
 1) Take initial vector $\Pi_0^T$ multiple Google matrix $G$, that gives $\Pi_1^T$, so we have  $(\Pi_0^T G = \Pi_1^T)$
 2. Then we do $(\Pi_1^T G = \Pi_2^T)$, then we keep doing m times, get $(\Pi_{m-1}^T G = \Pi_n^T)$

Equivalently we can simple state that as $(\Pi_0^T G^m = \Pi_m^T)$.
Let $x_i^T G = \lambda_i x_i^T$, assume G (an $N * N$ matrix) has $N$ distinct eigenvalues (difference from each other). Also suppose the $x_i$ for a bases, any vector can be written as a linear combination for some arbitrary coefficients.
$$\Pi_0^T = \sum_i^N c_i x_i^T$$

 A bases just means you can represent one vector in terms of the other vectors. Like the (0,1) at x axes and (1,0) at y axes,  the $\Pi_0$ could be represented by two bases (0,1) and (1,0). then
 $$\Pi_0^T G = \sum_{i=1}^{N} c_i x_i^T G = \sum_{i=1}^{N} c_i \lambda_i x_i^T = \Pi_1$$ $$\Pi_0^T G^2 = \Pi_1^T G = \sum_{i=1}^{N} c_i\lambda_i x_i^T G = \sum_{i=1}^{N} c_i \lambda_i^2 x_i^T = \Pi_2$$
 
 If we repeat the same argument when multiple x_i times G ($x_i^TG)$ that just equal to $\lambda_i x_i^T$.
 $$.... $$ $$\Pi_0^T G^m = \sum_{i=1}^{N} c_i\lambda_i^m x_i^T = c_1\lambda_1^m x_1^T + c_2\lambda_2^m x_2^T+.....$$

Approximately we say 

$\Pi_0G^m \approx c_1\lambda_1^mx_1^T$, m is large.

But we know the left side sum to 1, because the sum $\Pi$ is to 1 by multiple matrix G, each row sum to 1, its going to give us a vector $\Pi_1$, let $\Pi_m = c_1\lambda_1^mx_1^T$, so the vector $\sum_i \Pi_{m,i} =1$, so $c_1\lambda_1^m$ called as **constant**. So $\Pi$ is the normalized eigenvector $x_1^T$. So the vector will be such sums to 1, the direction will be $x_1c_1$ is 0, then that constant will be 0. 

Another different way of writing $Ax_i = \lambda_i x_i \space \forall i$. 

![eigenvector decomposition](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/Eigenvector%20decomposition.PNG)

The advantage writting in this way, I can then say that $A$ times eigenvectors $x$ equal to $x$ time diagnoal, which call as big lambda. 
$$Ax=x \Lambda$$ If multiple the inverse of $x$,
$$Ax x^{-1} = x \Lambda x^{-1} => A = x \Lambda x^{-1}$$ Matrix $A$ can be decomposed as product of the matrix of eigenvectors and the matrix of eigenvalues. The decomposition is going to be extremely important. 

If a matrix is symmetric which means its revise is equal to its transpose. And those eigenvectors are orthogonal. 

![symmetric matrix](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/symmetric%20matrix.PNG)

The matrix is m x m = vector (m x 1) * vector (1 x m)

### Vector norm
![rotation matrix](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/Rotation%20matrix.PNG)

$x\in R^{mx1}$ means x has m real numbers.  The norm of the vector is just the length of the vector, it can also be written as root of $x^Tx$. $||x||_2$ also call as L2 (norm 2). 

$(AB)^T$ = $B^T A^T$

Q is the orthogonal matrix, it rotates x, but does not change its norm/lengt. So the Q also called rotation matrix. 

# ML 15 - Singular Value Decomposition - SVD
SVD is a matrix factorization, if you can computer factorization, computer eigen vectors, you can computer the SVD. SVD it applies to matrices that are not squre.  SVD is the tool to do eigenvalues when the matrix is not squared. 

SVD consists of the product of three matrices. The first matrix $\Sigma$, it kind of like $\lambda$ here, it will have something related to the eigenvalues in the disgram. Those guys will be called the singular values. 

![SVD decomposition](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/SVD%20decomposition.PNG)

The matrix $A$, it gonna have two kinds of eigenvectors, it has the left and right eigenvectors. And why it has two, because the sizes are different. one size will be m, another the ones on the right will be n. The U times U transpose ($U^T U$) will be identity. V is equal to V transpose. 

In some code U is m x m,  when Python add some extra artificial columns in matrix, they will not be touched, so don't need worry about. Depnding on the **m may larger than n, the padding may change**. 

### Equaivalent ways of writting the SVD (thin)
![thin svd](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/thin%20svd.PNG)

![svd in image](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/svd%20in%20image%20example.PNG)

And when using SVD for images, the first $\Sigma_1u_1v_1^T$ is essentially capturing to sort of the **broad regions of light** of the image, the next $\Sigma_2u_2v_2^T$ is given you a bit detail, move to the right the subsequent components will focus on tiny details. So in nature is if you decomposing an image into smooth brush strokes. So the 1st eigenvalues capture most of the intensity of the image, most of the brightness of the image. 

And when computing the image, it is using a series of expansion of the vectors represent the matrix, so the expansion of the vectors could be stoppted early, maybe just in 10 terms could achieve the image compression. If its color image, it just three matrixs, RGB, here just one matrix, grayscale images. 

In grayscale image, we break the images into a scaler and two vectors when we multiple these two vectors, we also get an matrix, but we only generate a matrix at the time of display, it will not be in the storage.  we only store the vectors, its cheapter to store the vectors. So when we mulitpe the first vecorre, we get the 1st image. 

## How to computer SVD
![compputing  SVD](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/computing%20SVD.PNG)

$A^TA$ or $AA^T$ are the symmetric matrices. $\Sigma$ is diagonal matrix, so if you do transpose, it don't change.  U is the orthonormal columns, so U transpose U is just the identity. 

We have the $A^TA = V\Sigma^2V^T \equiv X\Lambda X^T$ and $AA^T = U\Sigma^2U^T\equiv X\Lambda X^T$  which means if I just created a matrix I multiply times itself transpose and computer the eigenvalues that I can computer eigenvalues, eigenvectors. Then it just matching, the $\Lambda$ is equal to $\Sigma$ and $X$ is equal to $V$. That allows me to computer $\Sigma$ and $V$. 

### Example for SVD
There are two composition options, either firstly computer $A^TA$ or $AA^T$, since the later composition computing is given matrix, it takes long time. so we do the first, because it is given the result directly, 14 is the 1 x 1 matrix. When we have the value for $A^TA$, then we can give the 3 x 3 eigenvalues.

![SVD example](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/SVD%20example.PNG)

Soving the matrix 3 x 3, we are using
$A = U\Sigma V^T$ = $AV= U\Sigma V^TV$ =  $AV\Sigma^{-1}= U\Sigma \Sigma^{-1}$ = $AV\Sigma^{-1}= U$ = $\frac{1}{\surd14} \begin {vmatrix} 1 \\\\ 2 \\\\ 3 \end{vmatrix}$

Learning is try to find the small representing of the world. And when do the image compressing, never store the matrix, only store the vector.

### Image compression in Python
![Image compression in Python](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/Image%20compression%20in%20Python.PNG)

    m, n = img.shape (# of rows and columns)
    U,S, Vt = svd(img) (Get the eigenvectors)
    S = resize(S, [m,1]) * eye (m,n) (It is tricky, make sure we get a diagonal matrix with the singular values.
    k = 20 (it is truncate, only take the top K components)
    # dot product of U*Sigma*V transpose, and only take the k largest, taking all the rows and # of k column from U matrix, taking k*k small matrix from Sigma matrix, taking # of k rows and all columns from T transpose. 
    imshow (dot(U[:,1:k], dot(S[1:k,1:k], Vt[1:k,:])))

The code:
- loads a clown image into a 200 by 320 array A
- displays the image in on figure
- performs a singular value decomposition on A
- displays the image obtained from a rank-20 SVD approximation of A in another figure.

**The truncated SVD**
It is the trciky in SVD. 

Suppose, m = 5, n =3, the U has the same matrix with A, the $\Sigma$ is the singular diagnoal matrix, the $V^T$ is the full 3 x 3 matrix.

![truncated SVD](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/truncated%20svd.PNG)

When we do truncated, eg k=2,  then the A just take the first two components. This is how doing compression, throwing away some of the vectos and only keeping the factors that are associated with the largest singula values. So the truncated just take the value in the green boxes.

So in the computing, firstly do SVD, then drop the small eigen values, $G_3u_3v_3^T$.

**Questions:**
The orginal storage requirements for A are: 
200 x 320 = 64000 pixels

The compressed representation requires: (k=20)
20x200 + 20 + 20x320 = 10420 pixels
$\Sigma$ is the diagonal, so don't need store all the values, just save the non-zeros values, so it is size of 20.  

# ML 16 - Principal Component Analysis - PCA

PCA is an application of the SVD essentially. It is the one of the machine learning techniques. 
- It useful for data visualization
- Reduction data dimensionality (data is very large dimensional projectors data to a lower dimensional space)

![Data visualization](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/data%20visualize.PNG)

The orginal data may be a table, including Country, area, income, population, etc. Fristly you want to visualize how similar the countires. Every countries have m attributes, Project to only has two dimensions, d=2. Why we can turn each country have only two features to represent? Because we can do 2D plot, if you have two numbers that the X & Y position, so you can just plot the coordinates. 

When you have the countries with 20,000 features, it will very hard to understand what's going on with these data. so using project can visualize what they look like.  If there is a curve in low dimension, then can come up with curve, then I have two recognizing. 

Followings are how to use the curves that fit the data. Another example, the images of sign language.

### PCA derivatiion: 2D to 1D
![2D to 1D ](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/PCA-2D%20to%201D.PNG)

If you take any vector $x_i$ can be written as the product of the vector $u_i$ times $\Sigma$ times the $V^T$. When we have the $x_i$, then we do the truncation, throw away the small eigenvalues, get $\hat x_i$. The vector in the direction of $v_1$ is $\hat x_i$, we can see $x_i$ is an orthogonal projection, just 90 degrees. 

If you have many points, we're replacing those points that lives in 2D with a projection that lives in 1D, projecting each of these individual points to the line.

### How PCA works
![how PCA works](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/how%20PCA%20works.PNG)

$x_i$ is the combination of two vectors, the points in 2d. The first we do with the SVD, we multiply by V transpose ($V^T$), now I'm actually carrying out the SVD. V is the rotation matrix. 
In Python, you just do, Xrot = np.dot (X, V.T). Xrot[:,1]=0 is projected two points down and then we rotate back. 

We have two dimensional dataset to one dimensional dataset, but one dimensional dataset still captures the topology of the points in 2D, in particular green/red/blue points are still close to each other. So the tricky is, do transformation, a reduction/compression, in compression we haven't lost some of the important properties of the data. So the data nearby in high dimensions, still be nearby in low dimensions. 

#### PCA for 2D visualization
![PCA for 2D visualization](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/PCA%20for%202D%20visualization.PNG)

$I_i$ image i , which is 40 x 40, then transform the image matrix to vector size 1 by 1600, $A$ is the database of n images. you call the SVD in python, it returns $U$, $\Sigma$ and $V^T$. You got the plot that will be the points instead of images. 

PCA in Python

    U,S,V = SVD(A)
    # truncation is 2
    K=2
    Z=dot(U[:,:k],eye(k) * S[:k])
    figure(1)
    # plot the images as points on 2D
    plot(Z[:,0], Z[:,1],'ro')
    grid()

### Standardize the data first!
![standardize the data](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/standardize%20the%20data.PNG)

When you get data if you get a matrix in Excel, and you want be able to do comparison, always a good idea to go over all your data instances subtract the mean and divide by the standard deviation to make all the data vary in the range zero and one. In terms of a Gaussian. 

Text is also matrix, the page with words, it is calculated. 

### Advanced: PCA as orthogonal reconstruction
![PCA as orthogonal](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/PCA%20as%20orthogonal%20reconstruction.PNG)

I'm gonna assume I can generate images or text documents orres of  collections of songs by features of song, so any data that I can write as a matrix and the excel sheet. Z and W are two matrices, and they are unknown, so the questions is how I could find the Z and W to multiply them, you will get the thing that's closest to the database.

![image and NN](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/image%20and%20NN.PNG)

SVD is essentially the tool that gives the optimal matrices and matrix encoding for any data matrix. 

Any image x can be represented as a number times a set of basis vectors, each of these components is a neuron and  a neuron  has associated with a weight which we learned that weight is $V^T$ and and the coefficient the Z is just U$\Sigma$ 

#### The weights found with sparse coding and PCA
![weights found with sparse](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/weights%20found%20with%20sparse%20coding.PNG)

Weights for PCA look like with different frequencies, this is why we see $\Sigma_1$ captures the big sort of variation in light, $\Sigma_2$ gives your more details and  as you go down the $\Sigma$s, you get higher and higher resolution and details. Think of this as sinusoids, you seen Fourier series in 2D. In later, we will learn other trick  to add a thing called a regularizer $\lambda ||C||_1$, and it will give us left side image, just a small trick is going to give us the features  

# ML 17 - Linear Prediction

Supervised learning requires some supervison, you have some ddata, you may have the labels for the data. When we have labels, when we have new person, I measure his height and weight, then measure the data what I've gathered before, then I predict his sex male or female. 

Linear regression is related technique aka **least squares**, least squares a maximum likelihood estimate. It is the easiest sort of formulation in supervised learning. 

## Linear supervised learning

![linear supervised learning](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/linear%20supervised%20learning.PNG)

Linear models are very simple, the essence of a linear model is that the world might look like thhis and you're approximating the world by a line. Something is may approximated by a line, it may well approximated by two lines. 

So if look at modular implementaions that use many linear models and linear models wok well in different regimes. 

**Set up linear model**
Give some data, the data will consist of pairs the data and the labels, and the output $y_i$ is real number, later example the $y_i$ will have binary value. 

![setup linear model](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/Setup%20linear%20model.PNG)

The typical dataset with $n=4$ instances and 2 attributes (Wind speed, People inside building), thermostat $y$ is the Energy requirement.

**Energy demand prediction**

![energy demand prediction](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/Energy%20demand%20prediction.PNG)

Learn a model of how the inputs affect the outputs, two phases in the prediction $\hat y(X_{n+1})$, first phase is we call training and in training, you given paris of data $X_{1:n}, Y_{1:n}$ and given to learning machine, the learning machine is described by a set of parameters, once we've estimated there parameters. the 2nd phase is validation or prediction.

When you have many variables which are useful for predicting?

![linear learning model](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/linear%20learning%20model.PNG)

$\hat y(X_i) = \theta_1 + x_i\theta_2$ is the linear model, $\theta_1$ is the intercept , $\theta_2$ is the slope. 

In learning we are given many data points, we try to get a line that goes through these data points. Each of there data points is a particular instance of an  input and an output. Go through those points what I'm going to require is that the quadratic distance between the points and the evaluation of the line at the point which is $\hat y(x_i)$, I want that gap to be small. We do $(y_i-\hat y(x_i))^2$ with square, because some of them will be negative. 

We want minimize the amout of purple in this image basically. We want do all the sums of the distance, so I look at all the distances  from the points to the line, the vertical distance, and minimize the sum of those vertical systems. The squares distance we call least squares, because we're trying to find the least of the square minimizing. 

So  $J(\theta) = \sum_{i=1}^n (y_i-\theta_i-x_i\theta_2)^2$ has the goal is to be able to solve for $\theta_1$ and $\theta_2$ that is what the learning for context of a linear model.  $J(\theta)$ is called as objective function, and only one attribute.

If we have D attributes, many more attributes, we gonna write this with following expression:
$$\hat y_i = \sum_{j=1}^d x_{ij}\theta_j = x_{i1}\theta_1+x_{i2}\theta_2+...x_{id}\theta_d$$

Each axes that I measure for times  D, in the case what I had height, so $x_{id}$ would be the height and $x_{i2}$ would be the weight, the module would be $y= 1*\theta_1+x_{i1}\theta_2+x_{i2}\theta_3$, if I want have the same scale, I probably would want to label this to $y= 1*\theta_1+x_{i2}\theta_2+x_{i3}\theta_3$, introduce 1 in the formular is as tricky, and the reason introduce 1 because I make that 1 be an $x$ all the time, then instead of having to equation in the form $y=\theta_0 + \theta_1x$, I could just wrtie it as $y=\sum_{j=1}\theta_jx_j , \space x_1=1$, provide that **$x_1$ is equal to 1**. 

![linear prediction](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/linear%20prediction.PNG)

And the matrix form, the expression for the linear model is:
$$\hat y=X\theta$$ We don't want to have plus $\theta_0$ all the time, so the way to do that is I just made that first vector of axes all be equal to 1, so we have $x_{11}=1$, $x_{21}=1$...$x_{n1}=1$, so if all the first columns are ones,  so it will allow me to write it as just $\hat  y= X\theta$. Because I will always need a constant, and the $\theta$ in 2D with the height and the weight, so one $\theta$ controls slope (front and back), another $\theta$ controls another slope (left and right), but I need the bias term, the free term $\theta_0$ to change the height. so I can change the height and tilt it.  So I need three parameters to control the directions. When there are three parameters I can full control the plane in higher dimension.

Typically we plug all the n observations and in the vector $Y$ which is n x 1, the matrix is n x d,  and d parameters which are unknow and the goal of leearning is to solve for those deep varieties.  

**Continue the example** for linear model searching for $\theta_1, \theta_2, \theta_3$

![thermostat example](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/Thermostat%20example%20with%20linear%20model%20.PNG)

Assuse we have learned the good $\hat \theta$, then we want to make predictions, I just need make predictions by multiplying $X$ times $\hat \theta$, get $\hat y$, then I compare it with true $y$ to know whether the prediction is doing well or not. 

Then I do prediction on new dataset (test dataset). 

## Optimization approach

![optimizaton approach](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/optimization%20approach.PNG)

Let's go back to objective function, our aim is to minimise the sum of squares, in the matrix form I can write the sum of squares as $(Y-X\theta)^T (Y-X\theta)$, and we do know $\hat Y_1 = X_1^T \theta$, and $Y$ is vector in one dimensional, the dot product of  $(Y-X\theta)^T$ and $(Y-X\theta)$ is just $||Y-X\theta||^2$

In the example there only with two $\theta$ ($\theta_1$ and $\theta_2$), if I have only two $\theta, it always reach the same point, it has an unique solution, but learning is about starting at some point in some objective function and going downhill. You formulate the problem as a cost function that you need to so optimize solve for and then you solve for $\theta$ by going downhill. Later on when we do neural networks there's not going to be just one minimum but it's gonna are going to be very jaggedy they're going to have factorial number of minima and the way to get there will still be by following the gradients. 

Because its is quadratic equation, it is always could find its minimum, so the solution will be unique, that's why you'll only need a line of code. 

# Least squares and multivariate Gaussian

## Optimization
![Optimization](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/Optimization.PNG)

There is a trick in $-Y^TX\theta-\theta^TX^TY$ to be $-2Y^TX\theta$, you're multiplying a $\theta$ and and X and a Y, because $Y^TX\theta$ and  $\theta^TX^TY$ have the same dimension result, that is 1 x 1, they are actually scaler, when you have a scalar, the transpose of a scale is just the scale. So I can add them up and written them as $-2Y^TX\theta$. 

The derivative is not a scalar in this case, but vector. Because the first term $-2X^TY$ is d x 1, the second term $2X^TX\theta$ is d x 1, we call this is gradient of J($\theta$), so the gradient is just vector of derivatives. **If you take the derivatives respect to a vector to the vector of all parameters that vector you get is the gradient**. 

What's the significance of the gradient? The significance of the gradient is that if I have two dimensions ($\theta_1$ and $\theta_2$), let's plot J of $\theta$ for a two-dimensional $\theta$, for the two-dimensional $\theta$, $J(\theta)$, its gonna be a ball, and it could be cut anywhere, and that give you the contour plots are the circles that have same height or ellipsis. Now the gradient has the property that it is a vector and it's a vector that at any point it can be evaluated at any $\theta$ with the gradient is a function of $\theta$ and any $\theta$ that gradient is perpendicular to the contour plot. So in a sense the contour surface, if you follow the contour surface your height doesn't change, you go around the grass mount and at the same height, the gradient tells you the point that if you were to follow, it would give you the biggest change in height or the steepest descent. 

So the way to remember gradients if you like snowboarding is you stand on the top, you pick the descent whatever the clip goes down the quickest and you follow that direction, when you follow that direction you're following the gradient.

How do we find the location of minimal? This is the location in the bottom of ball, to find it we just need to figure out where the derivative is zero, so going to equate it to zero.

### Least squares estimates
![LSE](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/Least%20squares%20estimates.PNG)

The least squares estimates is $\hat\theta_{LS} = (X^TX)^{-1}X^TY$ (this is the learning), and if you want the prediction $\hat Y$, the new $X^*$ times $\hat\theta_{LS}$.

Its very essential keep track the matrix size, the number of data by number of features. The matrix is not do the operation. We don't divide matrix, we multiply inverses, then use SVD rules to get identity, the identity $I$ multiply $\theta$ (a vecto) will get $\theta$ (a vector).  

**Important: not all matrix are invertable!!**

### Multiple outputs
![multiple outputs](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/multiple%20outputs.PNG)

Do example, there are two classfication output, $\hat Y_{11}$ and $\hat Y_{12}$ and with sequence of signals $X_1, ...X_d$.

## Probabilistic linear prediction

When we do probability ways, using the maximal likelihood and take its log, then differentiate it and equat to zero. Then we will get same value of the $\theta$ just like Least squares estimates. Whatever the predication of coins, linear model or NN in the later future, the procedure is always the same. 

### Univariate Gaussian distribution (1D Gaussian)

![univariate gaussian distribution](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/Univariate%20Gaussian%20Distribution.PNG)

The probability density function of a Gaussian distribution, its a functional $X$ and you can plot is. Its 1D functional $X$, and looks like a bell.  The location of the mean because its symmetric, so it is also maximum, called $\mu$, and the spread is controlled by something called $\sigma^2$ by the variance.  For short, I would say that any $X$ that you were to sample if I use a random number generator to produce excess  I will mostly get $X$es that tend to be around the peak, so if I call a random number generator and I say produce random numbers according to this, it will give me mostly random numbers than look like $\mu$ plus or minus some nosie. 

We describe this equation, we say $X$ is a sample from a Gaussian distribution that has mean $\mu$ and variance $\sigma^2$, and often we use an $N$, because the **Gaussian distribution** **is also called Normal distribution**.  

### Multivariate Gaussian Distribution
![multivatiate gaussian distribution](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/multivariate%20gaussian%20distribution.PNG)

Because machine learning we're going to have many inputs trying to predict several outputs, so we're going to work in high dimensions, we then have to deal with multivariate distributions and multivariate just means many variables, so the generalization of the Gaussian is essentially this expression that I'm showing here it's also a function of $Y$ but in this case **$Y$ is vector** and $Y$ has n dimensions, so this could be a very large vector, a very high dimensional object, it still has a location which is the mean that's the location of the peak. And the $\sigma$, the **variance** now gets **replace by a covariance matrix** $\Sigma^{-1}$, covariance you need matrix because now if you only had a 1D curve, you only can make it fat or thin but if you have a 2D curve now you can change it in many ways.

This is why I only have three parameters because I only have 3 degrees of freedom $X$, $Y$ and rotation. 

#### Bivariate Gaussian distribution example

![BGD example](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/BGD%20example.PNG)

If two gaussians, two variables are independent, you will get zeroes in the off idagonal in the matrix. Essentially you get a circular covariance matrix in this cases.

# ML - 19 Cross-validation, big data and regularization

## Regularization
![regularization](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/regularization.PNG)
You try to regularize the problem, they add the small elements in the diagonal of the matrix to each element in the diagonal. We start cost the function, we differentiate it equate to zero and then we get these $\theta$ that are better behave. 
![derivation](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/Deviation%20of%20regularization.PNG)
We get $\hat\theta_r$ theta ridge is euqal to $(X^TX+\delta^2_I)^{-1}X^TY$, it is exactly same to maximum-likelihood except it has that delta square there. 

**Question**: what will be the solution if replace the matrix X by the SVD of X?

### Ridge regression as constrained optimization
When you replace the matrix X by the SVD of X, you actually learned that what ridge does is it get rid of the small eigenvalues. 

The objective has two components, it has a form from the squares (the blue color) and the regularize (the green color), in order to minimize J to have as few as little cost as possible to minimize, so the blue and green both need be minimized. But sometimes minimizing blue will be increasing green and vice-versa. So the $\theta$ minimizes blue is not a $\theta$ that minimizes green, the $\theta$ minimizes green has to be feet equals zero, if the $\theta$ turns zero in blue, then the blue term will be $y^Ty$, it could be very high value. So there is a trade-off between these two terms, we need to find a $\theta$ that minimizes one that minimizes the other.  An alternative way to formulate this, think of an arbitrary function, it's an unknow function, wedon't need to knowit but you could think of it. Essentially we wanted trying to do we minimizing $t(\delta)$ likelihood subject to $\theta^T\theta$, constraint subject $\theta^T\theta$ being smaller than some constant that depends on $\delta$

The equation $\theta_1^2+\theta_2^2$ = constant is a circle, the contours is what we observe here at the bottom they're all circles. The circle centered at the origin that's the shape of the first component. The other shape represents $J_L$ is some cup around $J_R$, the minimum of $J_L$ and equate get to zero, you get the minimum which happens to be the lease squares, the $\hat\theta_{LS} = \hat\theta_{ML}$, it is also quadratic, because there 's $\theta_1$ or $\theta_2$, but in this case there's cross terms, so the blue form can be expanded to be $a\theta_1^2 + b\theta_2^2+c\theta_1+d\theta_2+e+f\theta_1\theta_2$, when do contour plot for blue term, there are ellipises shape.  

![Ridge regression plot with 3D](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/Ridge%20regression%20plot%20with%203D.PNG)

Drawing it in 3D is very hard, so we only going to plot the contour plots, we know the plot for green box with the circle centre in original. And we know the ellipse for blue term, but we don't know where they are, we put blue dot arbitrarily, then put the ellipses. We know that the center  of the blue guys, which point here has coordinates. Because the $\delta^2=0$, then the green term is vanished, only the blue term existed. We also know if we let $\delta^2$ go to infinity, then the 2nd term basically have much more importance than the first term.  The $\delta$ is squared because we want that weighting $\delta$ basically balances you desire to have small $\theta$ and your desire to fit the data, and that balance I just want it to be **positive** that's why put the square root on $\delta$. 

What different $\delta$ that is not zero?

![ridge regression without 3D plot](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/ridge%20regression%20without%203D%20plot.PNG)

I will make the following claim, the optimum will be the point for any $\detla$ where the two curves, **the ellipses and the circles are parallel**, where the contour plots are bound the gradient by definition is perpendicular to the contour plots, that's the starting point for the whole field of convex optimization. The height of all these purple point are the same because they're all in the same contour plot for jail, if move the purple point which on the green circle to the purple point which on the blue ellipses, the green is moving bigger,  suppose the yellow points have the same height, moving the yellow point, the blue will be increased, so if I **did derivatively, either the likelihood or the regularizer will increase**, so I should avoid deviations and I should stick to that point. It will be true for the sky blue point and other points where the curves intersect, there is infinity number of kind of this becaue there is infinity number of possible values that $\delta$ could take between zero and infinity, for each of those values we get a curve that looks like the line connected from the original point to the intersect points (**the sky blue line**). When we increase $\delta$, $\theta$s go to zero, each $\theta$ gives us a point in that blue line, each of those is a solution.  

![L1 norm regularizer](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/L1%20norm%20regularizer.PNG)

With L1 regularization on the other hand, the regularizer is not circles,  but its those four lines, which give rise to that diamond shape, the point at which these guys will be the point of deviation, different yellow arrow direction will increase likelihood (sky blue circle) or regularizer (green diamond), so the optimal point, the difference there is that because I have now a diamond the intersection will often be one of the axis, it possible the likelihood is not touch one of the diamond corners (the yellow ellipses), but it is very rare. when you have the insector touch the corner, one variable is equal to zero, so there is a lot of variables are go to zero, this is why this is a better regularizer. If there is no intersection, which means all the features are important, eg for the genes, there are thousand of genes are all important. 

**L1 is much popular than L2**

![regularization path](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/regularization%20path.PNG)

$\hat\theta_2$ is going to 0 much quicker than $\hat\theta_1$

### Bayesian linear regression
![Bayesian linear regression](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/Bayesian%20linear%20regression.PNG)

For ridge, we've seen that least squares is the solution that you would get if you just do maximum likelihood, we know there are two ways of learning, maximum likelihood and Bayesian way, maximum just use likelihood, Bayesian use of prior and posterior, for bayesian it's about normalizing, it's about integrating. When you do the maximum likelihood you get as a solution least-squares, when you do the Bayesian estimate, the resudible prior, you get the ridge. And the Bayesian, we have other properties.

Basically you want the posterior of $\theta$, so you multiple the prior of $\theta$ ($N(\theta\space \theta_0, V_0)$) times the likelihood ($N(y \space X\theta, \sigma^2I_n)$), here we use a Gaussian prior if I multiple the Gaussian prior multiple the Gaussian likelihood.

![Bayesian linear regression proof](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/Bayesian%20linear%20regression%20proof.PNG)

Complete squares thing which is very tedious  to do but very important.

![Bayesian linear regression proof2](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/Bayesian%20linear%20regression%20proof%20continue.PNG)

Discover the things, the likelihood, the posterior is also Gauss. 

![Bayesian linear regression proof  end](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/Bayesian%20linear%20regression%20proof%20end.PNG)

More importantly, if you choose a particular prior, a prior that has zero mean,  your prior is saying I believe the $\theta$ should be zero, and my preference is that the variance of the $\theta$ be no bigger than $\tau^2$, and that prior, you do that completing squares, normalizing, trickery which is the same as what you do for the Beta binomial model and it's the same as what you do every time. **Bayesian is always prior times likelihood**, you group terms and then the group terms look like a distribution. When you do this, that posterior is just a reach estimate, so we recovered the ridge.  In regression, $\theta$ is always d x 1, $\sigma$ is scale. 

Ridge regression is equal to the mean of the posterior, the mean of the Bayesian in solution. 

### Bayesian versus ML plugin prediction
![Bayesian versus ML plugin](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/Bayesian%20versus%20ML%20plugin%20prediction.PNG)
For the Gaussian, we need the mean and the variance. If you want do prediction, there are difference between the frequentist and the Bayesian way. For the Bayesian making a prediction is equivalent to integrating out  p of y and $\theta$, bascially given a new data point $x_*$ and given all your previous data, so any data points in other words inputs output pairs if you want to make a prediction you have integrate out $\theta$ over all the $\theta$s. Then you do the factorization, then go to the bottom line to do the conditioning.  And the fact that the posterior is not a function of future axis, sojust drop future axis. So the basis solution is marginalized basic. You want to make a prediction you need to marginalize. 

I have infinite  $\theta$s, in fact they all have a weight which is the posterior probability and what I do for each $\theta$ I make a prediction in our weight prediction by the posterior height that's what integral means. The first term is likelihood, but I'm waitting each of the predictions, the likelihood is just $\hat y=x_*^T\theta_{ML}$, that's the prediction, and the variance of that prediction is your $\sigma^2$, it is a single prediction. The Bayesian is saying it's going to be many $\theta$s, they gonna over all the possible features in this case, there's internal features and each of those terms which out of this form will be multiplied weighted by the posterior, and it is a necessary marginalization. When you do prediction, the prediction should be independent of the paramenters you have. It should be robust. 

Bayesian is kind of hedge, they have many $\theta$ and many predictions, this idea of ensemble learning is very important, the idea of using many predictiors that what people call ensemble learning. If you marginalize, it is the $\delta$ function, then you can add in the end with $\delta_{\theta_{ml}}(\theta)d\theta$, before it just a spike at a point. If you integrate with respect to the $\delta$ function.

The maximum likelihood think there is only one solution, the Bayesian on the other hand, they believe you have to multiple two gaussians and integrate. The green $\sigma$ is always the same quantity. But the variance are difference in two formula (yellow box). And the mean is also different because the $\theta$ maximum likelihood is the least squares where is the $\theta_n$ is the ridge. So B**ayesian is a bit better because it's using the ridge where is maximum liklihood is using the least squares which is not as good**. 

the Bayesian is adding an extra term  to the variance what that $V_n$ is the proportional to the data, the inverse of the data, so it's basically an inverse of your data matrix.

![Bayesian versus ML in different variance](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/Bayesian%20versus%20ML%20plugin%20prediction%20diff%20variance.PNG)

The predictions of the mean tend to look the same but the variance the confidence intervals of Bayesian are different, theyy hae a data dependent term, this $V_n$ depends on the input data. when you have the data the variance is low, when you don't have the data the variance is high. $V_n$ is proportional to $X^TX$. And there is also have problems in reinforcement learning, this is sort of the basis to attack that problem.

### How to choose the $\delta^2$
Cross validation comes in many flavors and shapes. we don't know which is the $\delta$, so we just pick a bunch of values. All that I could is guess the values which is good, so in a sense we are doing optimization, we're searching for a good $\delta$ here. When doing the automatic procedure, the variance of those estimates is essential because using those variances to decide what's the next best $\delta$ to check. For each of the $\delta$ you pick, you will fit he model , then do the $\hat\theta_r$ ridge.

We computing the error each point, we sum the error of each point in the training set and then after we sm the errors for each point in the test set, **least squares means the sum of squared error**. So each point has a squared error, and so we sum all the errors. 

![cross validation](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/cross%20validation.PNG)

Base on those red numbers how do we pick $\delta$? There are many ways, but here I introduce a way whihc is called the pessimistic view or it's called a worse-case analysis. This is the decision theroy problem. you can do balance between the training and test error, you want to deal with the worst case, so you better choose $\delta^2$=1 because it has the lowest worse case (max=11), or you could pick the lowest average + oompare lower average. so could be $\delta^2=10$. 

Also could  break the data input three, training, testing and validation. 
### K-fold crossvalidation
![K-fold](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/K-fold%20crossvalidation.PNG)

You train on the white ones and you test on the red and then you cycle through them and then you take the average of the five errors, this is also good strategy.

### Analysis of the data
![Ridge Regression example](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/Ridge%20regression%20example.PNG)

It is the important thing, analysis is not about math, you can use neural nets and random forests, you can done those two boxes, if you increase the regularizer  $\delta^2$, your training error gets worse that makes sense because the total error is two things, the likelihood and the regularizer if you're increasing $\delta$, you going up, you going down in two circles in the middle but you're actually getting a worse likelihood, so your training errror is going up as you increase $\delta$, however the testing error goes down, cross validation is giving the sweet spot, which did the worse in this case is minimized, we want to do well not only in training but also in tests, if you want to minimize errors in the future , minimizing errors in the future is what we call good generalization, you want models that will generalize to new situations to the future, another technical term that we use is for the same thing is we don't want models that overfit. 

![Effect data on right model](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/effect%20of%20data%20on%20right%20model.PNG)

Now here are the important lessons, take a quadratic function, in order to generate data from the model, I'm going to generate synthetic data, pick a bunch of X's maybe on a grid, for each X you evaluate the quadratic, you choose a particular value, eg pick $\theta_0$ equal 1,  ..., just pick those value randomly, the truth $\theta$s. For each of X's, plug them into $\hat y=\theta_0+x_i\theta_1+x_i^2\theta_2$ formular, that give you get the height (green line between linear line and X axis), and then add noise $N(0,\sigma^2)$ (purple dot), that's how we generate data. You use the true model, then use a probability to generate it. The reason using a model to generate data is because now I do know the true model, in this case I do know the truth.

Let's do example choose another model, $\hat Y=\theta_0+x\theta_1+x\theta_2$, but in this case I forget that I know $\theta s$, and we try to learn the $\theta s$, and I gonna assume what I have is x and y and I'm going to learn the $\theta s$, but I do have the right model, I have a quadratic. I don't know ridge or least squares or whatever to learn $\theta$, onece I have learned the $\theta$ I look at the mean squared errors again which is just average of $(y-\hat y)^2$, and then divided by N a number of data, the average error.  They don't go to 0 because the quadratic can't go to 0 because the quadratic can't go straight through each point. We always making these small errors. 

**Lesson 1**: If you download some software you try on your data and the error is zero, you might be doing something not caution. 

The other way, the error might be really low in this case well this is the situation, the training error is really small for that number of data. But what happens is that you're overfitting the data at that stage. 

What happens for right model is that the data increases the training error tends to converge more to that expected thing that should be due the lowest that you could make which is the blackboard and test error goes down, in this case the convergence is very quick. In fact after just a copule twenty-five data pooints, we've gotten fine. This is when the model that you choose has the same complexity as the model that you will learn that generated a data. 

![Effect data on too simple model](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/Effect%20data%20when%20model%20is%20too%20simple.PNG)

Let's assume another simpler model, $\hat y = \theta_0+x\theta_1$, in other words my data was generated according to a quadrtaic function (purple is the true data). And then we are using prediction $\hat y$ linear line. You can see the problem of trying to fit a quadratic data with  a line, it's impossible to go through all the points and so even if you get more points it's not going to get better. 

Now  **lesson 2**: If you keep through adding more and more data and results are not improving, time to change the model, time to try something a bit more complex. 

![Effect data on too complex model](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/Effect%20data%20on%20too%20complex%20model.PNG)

Let's assume another complex model, $\hat y=\theta_0+x\theta_1+x^2\theta_2+...+x^{14}\theta_{14}$, a polynomial of degree 14 is a very squigly model, so if you have only a couple data in the quadratic, your polynomial of dgree 14 is going to do fit the data, the three points the error is zero. On the test data the curve is really high, so if your model is very complex, you doing what we in this case say overfitting, so **how can we stop the model from overfitting**? **We can just simply add more data**.  The complex model is constrained by the data, it can't jump off when within the data. the cost of complexy mode is that you need a lot more data. 

![ridge regression with 14 degree polynomial](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/ridge%20regression%20with%2014%20degree%20polynomial.PNG)

An alternative another example for polynomial of degree 14, let's assume we doing ridge regression to estimate the $\theta$ in other words we have basically the ridge cost, nonlinear regression with places with polynomials is still linear regression, we just need to reinterpret the axis as $\Phi$, so you observe $x^{14}$, $x^{13}$ and so on, and just the same as ridge and we solve for $\theta$, once we have the matrix $\Phi$, this is just again linear regression, but when we choose $\delta$ without increasing the data, we can choose the $\delta$ to balance the training and a test error. If you have too tiny $\delta$ you get over fit, When you choose large $\delta$, you get the polynomial of low degree, so if you choose $\delta$ you basically are choosing the degree of the polynomial automatically. 

So there are two ways, either you choose $\delta$ or you just do cross-validation with different polynomial degrees, but the better way is to just choose $\delta$ because it is also applicable in other settings, so a regularizer can give you the right generalization balance, increasing the data is always good thing but it's not the only way to do it. **If you're smart about how you do regularization**, **you can get very good erformance with few data points**.

# ML 21 - Sparse models and variable selection
## Selecting features for prediction
![select feature for prediction](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/select%20features%20for%20predictioin.PNG)

Think of you have many $X$es, many predictors, each of these predictors gets multiplied by a $\theta$, ad then it predicts a $\hat Y$, in particular for linear regression we know is equal to $\hat Y=x_1\theta_1+x_2\theta_2+...+x_d\theta_d$, now it might be that some of $X$es that we don't want to measure, so we would love is for some of these to go to zero, because imagine that $x_1$ is expensive and does not contribute to good predictions  $\hat Y$, then we want $\theta_1$-> 0, we will take all the inputs (the D inputs), we will fit a model, we're going to use a regularizer that will wet some of the $\theta$ to zero and once the $\theta$ are zero, we can drop those variables, so the variables associated with zero we remove. Like this case, the $\theta_1$ is zero that means we don't need $x_1$.

Assumption that we have several data, $x_1$ to $x_n$, for each data we have n labels, so n points x and n labels y, this is what I call iid which means that they're all from the same distribution, in particularly in this case they're all ID gausian, they all come from the same distribution with the same variance. what  that means if they're independent, essentially saying that P of Y want to n given $X_1$  to n if they Y's are independent, as a product over P of $Y_i$ given $X_i$, so  the graphical model that I have for this is that I have $x_1$ and $y_1$ and then I also have $x_2$ and $y_2$, $x_n$ and $y_n$, so my graphical models should product of these independent graphical models as suppose to a single graphical model, there is however a common thing to them which is $\theta$, they all different but they all share the same $\theta$ and that common $\theta$ that makes the points have a structure. In particular if you have a line, that $\theta$ happens to be the slope and the intercept, so giving you $\theta$ which is consists of the slope and intercept. The relation between each of the $X$es must be the same which is difficult to say that they are dependent on each other.

![Ridge and Lasso compare](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/select%20features%20for%20predictioin-part%202.PNG)

The ridge show you the plot on the left, he basically says you increase $\delta^2$ or you minimized the $t(\delta)$, it could be other function, we don't need to know what that function is but just a function that's by increasing $\delta^2$, it goes down and so as $\delta^2$ becomes very large, larger $\delta^2$ would be going left as you increase  $\delta^2$, you get closer to zero, because the ridge basically is a trade-off between $\theta^T\theta$ and $y-x\theta$, so that $\delta$ by increasing to zero you're basically syaing the most important thing is to minimize $\theta^T\theta$, the only way you can minimize that is by setting $\theta$ to zero. So if your cost function is $(y-x\theta)^2+\delta^2\theta^2$ and $\delta$ is huge, the only way I can make that cost function vanishes by making $\theta$ tiny and that's why as we increase $\delta$, the $\theta$ goes to zero. So when $\delta$ is very large, when $\theta$ is very small, you're back to maximum likelihood. This technique will make for example this guy here (red line) is getting very close to zero but it's not 0, it will be 0.01 or 0.1, so we then have to decide when should we **get rid of something when is something small enough and that creates problems**. The new technique will be such that when you have something like age, it will automatically set $\theta$ to 0, so $\theta$=0.7 will become equal to 0 for this particular value of $\delta$, and then there is some optimal value of $\delta$, that we find by cross-validation and that optimal value of $\delta$, there are only three things that we need to measure and we only measure those three things we can get a good prediction that is the best possbile prediction and the test set so instead of using eight variables we just use three variables.

We already have one that the $\theta$ gives us very close to zero, and that's ridge regression, I want to do better than ridge, max is argued that ridge is better than maximum likelihood, and I argued give you better confidence estimates and so on, when I want do better **not only will do better predictions or comparable predictions**, but I actually want to **also select the important variables automatically**, the spot on the right is saying that all of these guys are precisely zero, once they hit zero, they stay zero. Then there's no threshold and no heuristics.

The cost function has two functions, the light blue component (the likelihood) and sum of squared errors tha's usually squares criteria. The cost will be higher whenever the sum of square errors is high or  if the sum of $\theta$ is also high. So in order to reduce the cost I have to do two things, **I have to make my $\theta$ small and I have to make my error small.** I am not only worry about good predictions, also worried about making the $\theta$ small. so that I can identify which are the axis that ar e relevant, because if one of the $\theta$ goes to zero, I know I don't need that x.  

There is whole path solutions (red path), each point in that red curve corresponds to a different value of $\delta$. If you have the pyramid and an ellipse (in 2D), the intersect with great probability intersect at the corners, the coordinates coincide with the axis and if they coincide with axis means that the axis one od the variables is zero. When the axis to zero that $\theta_2$ will be equal to the maximum likelihood to $\theta_2$, and when $\delta$ goes to infinity (very large) all the $\theta$ will be 0. It is make sense when look in the cost function, the $\delta$ is like a quadrillion,  then the only way you can minimize $J(\theta)$ is by making $\theta$ very tiny. For learning is to derive an estimate of $\theta$, how do we do it? Take ojective function with derivative and equate to zero. 

### The Lasso: least absolute selection and shrinkage operator 
When it's applied to linear regression, it is called Lasso. $J(\theta)$ is objective, it instead of adding $\theta^2$, it adding the absolute values of each of the $\theta$s. In 2D, $\theta=(\theta_1, \theta_2)$, then the absolute value of $\theta$ will be $|\theta_1|+|\theta_2|$, if I look at the contour plots of that function, I would equate it to a constant, so I want to know at the point at which this is constant, and the absolut value ahs two solutions plus or minus x. So I have the euqations of four lines.  The $(y-x\theta)^T(y-x\theta)$ is the general quadratic. And another equation is in green, it is put lines on the axis. The significance of this is that the intersection point in this case of the quadratic and the diamond shape is very likely to be at the corner. now the coordinates are at zero. Now I still have a whole regularization part, but for most of time it is zero. Unlike the ridge where it's small but not zero. 

### Differentiating the objective function
![differentiating the objective function](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/differentiating%20the%20objective%20function.PNG)

$\hat y_i$ is completely different objective fucntion, it is prediction as opposed to the $y$. Westill dealing with linear model. $x_{i,j}$ that means all the $X$es but excluding the j,  $\theta_{-j}$ removed the j then add in $\theta_j$, it is the same thing.  $y$ is the noisy verison of the $\hat y$, the derivative is respect to its $J(\theta)$. Then we do rename for the part of terms.
$$a_j=2\sum_{i=1}^nx_{ij}^2$$ $$c_j=2\sum_{i=1}^n(y_i-x_{i-j}^T\theta_{-j})x_{ij}$$
so we have the equation to $a_j\theta_j-c_j+\delta^2\frac{\alpha}{\alpha\theta_j}|\theta_j|$

### Subdifferentials
![subdifferentials](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/subdifferentials.PNG)

You have  equal x on this side and so the slope is +1 on the other side the slope is -1, so we know the slope, when the x=0, we have non differentialble function, here is an infinite number of possible slopes at the corner. We introduce a new mathmatically concept, it is called subdifferentials. We need this concept in order to be able to do the derivative of an absolute value. It is very important mathematical concept, most of the currents trained in how we are going to acquire data is based on the principle that we instead of acquiring data in hight resolution and compressing it, we are going to cook get it straight from the world to a compressed version and the data will always loop in compressed form, if you want to view the data we can uncompress it and see it in its full splendor of either solution. In order to get very high compression rates people often use there L1 regularizes, there is a whole theory now called **compressed sensing**. The green line is the derivative, that's the sub differential. It is not one derivative, it is a set derivative. It could be anything that goes from -1 to +1, the x=0 is a whole set. 

![derivative objective](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/Deviation%20of%20regularization.PNG)

Then we going to differentiate it using the rule that's of the ratio. How do we get the $\hat\theta_j$, let the equation to zero. The $a_j$ is guarantee, it is positive, because it is  sum of squares. In the first equation, the $\theta_j$ is negative, the $\delta^2$ is positive, the $a_j\theta_j$will be negative,  so $c_j$ has to be less than $-\delta^2$, in other words , the $c_j$ has to be greate than $\delta^2$.

![derivative objective with differetiate using](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/derivative%20objective%20with%20differentiate%20using.PNG)

Above is just for one $\theta$, for all $\theta$s, all J, that we need the algorithm:

![sparse prediction algorithm](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/sparse%20prediction%20algorithm.PNG)

1. Initialize $\theta$ first, you may want to initialize just using ridge but if you just pick a random $\theta$ that's also good.
2. Keep repeating until converge and we know the converged when the $\theta$ stopped us changing, we repeat for all d
    - we first computer $a_j$
    - we computer $c_j$ 
    - then we just do the check

And there is smart programming, the $a_j$ is not include $\theta_j$, so could be pre-computer which is out of the loop. I am trying to update $\theta_1$ given the other $\theta$s, and then I have to take $\theta_2$ given the other $\theta$. And this algorithm is actually convex so it's actually converges. It does't always converge if we change the objective function. Converge means you reach a point, $\theta$ stops changing. This technique adding a L1 regularier, but also important in neural networks. 

### The effect of L1 regularization on PCA
![the effect of L1 on PCA](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/the%20effect%20of%20L1%20on%20PCA.PNG)

On the right hand side if you took a collection of images, PCA gives these filters. If I did not have the penalty $\lambda||C||_1$ and I decide to optimize and ask for the components to be orthogonal, I get PCA. PCA is the optimal solution to that optimization problem. If I add L1 norm I get sparse faces (left hand side) and I get them to be local and what this image here, there is a region that is black on one side that is dark on one side and that is light on the other side. what's dark on one side and light another side and edge, you can think these V's as being a detector for the line in the world. So if in the neural networks ehc of theser guys is essentially the receptive field of neuron. The key to get such image is add L1 norm, it is not just for linear regression, but we use it a lot in all sorts of problems. 

# ML 22 - Dirichlet and categorical distributionses

Here is the extension to more than to two values, the multivariate extensions.

## Revision: Beta-Bernoulli
![Beta Bernoulli](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/Beta-Bernoulli.PNG)

If we have variable that's binary, the likelihood can be represented in terms of a Bernoulli distribution, $N_1$ say the coin was one and $N_0$ is say the coin was zero. The curly $D$ represent $X_{1:n}$ all the data. We also learned that the natural prior for the Bernoulli distribution was the beta distribution, so the beta distribution allows us to encode our beliefs about what the value of $\theta$ will be and $\theta$ is between 0 & 1.

![conjugate](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/conjugate.PNG)

We learned that in order to do Bayesian inference,  one doesn't do optimization, one doesn't believe that there is only one value of $\theta$, one usesthis notion that there is a subjective prior that encodes the beliefs $P(\theta|D)$ and then multiplies the prior times the likelihood and normalizes to get the posterior distribution which in this case is also beta distribution when the **prior and the posterior have the same form**, **we say that the prior is conjugate to the likelihood**, so the word conjugate is oftern used in books and that just meas that if you start with a prior of type X you multiple it by a likelihood ,your posterior is also type X.  This case, where the prior is beta, the posterior is beta, the other case is Gaussian case, so if  you have a Gaussian prior and $\theta$ in linear regression, the posterior is also Gaussian.  We didn't do the variance but for the variance there is a similar, if there variance we use a distribution called the $\gamma^{-1}$, that is just like a Gaussian but it only allows for positive values because atevariances can only be positive.

Today we talk is the **extension of the beta distribution** and the **extension of the Bernoulli distribution**.

### Multivariate version of Bernoulli distribution
![categorical distribution](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/categorical%20distribution.PNG)

This is going to be called the categorical distribution, it just say something another of this of category X or category Y or category Z, so we're categorizing data. Let's assume we have n data points, $x_{1:n}$,  each $x_i$ will take K values, we just using binary encoding to represent each number, like (100), (010),(001) when k=3. 

For variable $x_i$ will be the product $\prod$  from j=1 to K of $\theta_j^{\amalg(x_{ij=1})}$ theta j to the indicator variable of $x_{ij}=1$, Let's see why it is make sense, if I have a vector, $x_i=001$, then the $P(x_i|\theta)=\theta_1^0\theta_2^0\theta_3^1=\theta_3$, then we add condition, $\theta_1+\theta_2+\theta_3=1$, so the last  $\theta$ much be 1 minus the sum of the previous $\theta$s, so this is the categorical distribution, it sas what the probability that you belong to each of the categories. When we have several data points if we assume that these the data iid that there all come from the same distribution in this case categorical distribution and if you assume that they're independent then we can just multiply them. Once you have likelihood, you know how to computer $\theta$, take the log, differentiate equate to zero. 

### Dirichlet distribution

![Dirichlet distribution](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/Dirichlet%20distribution.PNG)

Just like we had a $\beta$ distribution, we're going to hae a generalization of the $\beta$ distribution which is as you would have expected it will just look like the beta distribution, so recall that the beta was $\theta$ times 1 minus $\theta$ into these parameters. In this case we're just going to assume that there's K possbile values that the variable could take, then we just can take $\theta$ to a variable and instead of having $\alpha$, $\beta$ and $\gamma$ and all of these variables we're just goint to call them all $\alpha$ and introduce index K.  $\beta$ was proportional to $\theta$ to the $\alpha_1-1$ and $(1-\theta)^{
\alpha_2-1}$, this is a beta distribution, just looks like the beta distribution except that now instead of going up to two values, we're going to go up to K values, going to multiple K $\theta$s.  $\frac{1}{\beta(a)}$is the normalizing constant which is this ratio of $\gamma$ distributions just like the $\beta$ distribution. 

### Dirichlet-categorical model

Before I have a Beta prior $\theta^{\alpha_1-1}(1-\theta)^{\alpha_2-1}$, and a Bernoulli likelihood $\theta^{N_1}(1-\theta)^{N_2}$, and when I did multiplied the $\beta$ prior times the Bernoulli liklihood, and I derived the $\beta$ posterior. 

![Dirichlet-categorical model](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/Dirichlet-categorical%20model.PNG)

Right now I'm going to do the same analysis but instead of using a $\beta$ and Bernoulli distribution, I'm going to use a categorical distribution and Dirichlet distribution, so we're going from coins to dice. The posterior of theta given all the data $x_{1:n}$ ($P(\theta|x_{1:n})$ is proportional to the data $x_{1:n}$ given $\theta$ times the prior $P(\theta)$, that gonna be proportional, the data seem to be independent, so the product from i euqal 1 to n and the product from j equal 1 to K of $\theta$ j indicator of $x_{ij}=1$, **that's just a likelihood**  times the proudct from j equal 1 to K of $\theta$ j to the $\alpha_j -1$ (green part). This is equal to the probability from j equal 1 to K of $\theta_j^{N_j}$, so the $N_j$ represents the number of time you had a one in position j. So the posterior has the shape of a Dirichlet distribution as well. So if the piror is there is a in the likelihood is categorical, then the posterior will also be Dirichlet.

Let's have the **example** on text classification 

![Text classification example](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/Text%20classify%20example.PNG)

Suppose you wanted to assess the popularity of the president in this country south of the border, you can search for a lorge collection of tweets that have  a happy face, then you say that every tweet that have a happy face is a positive treat, then look for a collection of tweets that have an negative sad face and then you asy taht those tweets that have negative tweets. Then you have three categories, postive, negative, neural.  To encode the word we need vector X will have zeros everywhere and we'll accept where the words occur in the tweet in which case there will be ones. Because the Omg, cat sat is appeare in my idctionary, so put 1, if the word does not appear, then put 0.  But we will not pick of the word that, it is obviously will not help me decide whether a tweet is positive or negative. So when you know the multi categorical distribution you can do this base on $x_i$. Because we can rewrite this problem just as a problem of learning $\theta$ for a categorical distribution and then we will solve this using maximal likelihood by differentiating the log likelihood equating to 0, and add conditional to constraint to ensure that the $\theta$ sum to 1, that is a new tricky from optimization and then we're going to do the Bayesian analysis. Once we have the $\theta$ we will see that it's possible then to compute the probability of a new X given $\theta$ and X, so the probability of a new I gien $\theta$ and X that is when we see a new tweet we will able to predict it as being happy or sad.

**How we gonna do this?**

### Naive Bayes classifier

![naive bayes classifier](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/Naive%20Bayes%20Classifier.PNG)

In this case, we're going to have two multinomial. One for y and one for x, so now we're going to model the distribution of the data of the input, unlike linear regression here we are gong to model a probability of x and there is going to be two parameters, one for the probability of x which is $\theta$, and the probability of y which is $\pi$. And I am going to use Bayes rule to computer the prbobaility of y given and input x and given the two parameters that we will learn by maximum likelihood or Bayesian inference and the normalizing I could also write this as P of $Y_i$ given PI times the probability  of $x_i$ given $y_i$ and $\theta$ dived by the sum of $y_i$ of P of $y_i$, say $y_i$ being of calss c and this is the application what we've been learning of Bayes rule, the normalizing constant is just a quantity that ensures that you have a probability of our y that sums to. 

Coming back to y, the distribution probability of $y_i$ given $\pi$ is going to be a multinomial from small c equal 1 to big C of $\pi_c$ indicate of $y_i=c$, we just use a categorical distribution to indicate the prior probability for the calsses what's the probability that a priority something is positive or negative. There are some candidates for which the sentiment tends to be more positive. The probability that $y_i$ is of value small c given $\pi$ is just equal to $\pi_c$, and the sum of $\pi_c$ to 1,so it's just a categorical distribution for the x's is also going to use a categorical distribution. 

![Naive Bayes Classifier plot](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/Naive%20Bayes%20Classifier%20plot.PNG)

![Test classification example update](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/Text%20classify%20example%20update.PNG)
I am going to assume d=5, so d is the size of the number of X binary variables, the number of distinct words that item use. So for $P(X_i|\theta, Y_i =c)$ Y is the praticular class with in the class. And I'm going to make the assumption that from j=1, j=2, all the way up to j=d, I have d binary variables, I'm going to model X as d binary variables, it is means the word present or absent in the tweet, and I assume they're either present or absent independently from each other. One class you might have words that tend to right-side, another class might tend to left-side, with slightly different shape, so those two figures there on the x-axis with 600 words and y-axis the frequency of those words and that's for class 2 and class 1. So in the different classes you would see different kind of words, for example for class possitive things like health care, New Hope, etc, for the negative class you might find things like socialist, etc. If you did the assumption which the words are independently, you get a classifier actually words remarkabley well.

![Naive Bayes Classifier with binary features x](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/Naive%20bayes%20classifier%20with%20binary%20features%20x.PNG)

We assume all the words are separate and then using the joint distributon of y, given the new input and the parameters can be written in this from where you will have $\pi_c^{\amalg_c(y_i)}$, this is choose the prior of a c and $\theta{jc}^{\amalg_c(y_i)\amalg_i(x_{ij})}(1-\theta_{jc}^{\amalg_c(y_i)\amalg_0(x_{ij})}$, this is the binomial that ended for each word for each j and it will depend on whether what class you're at which data point your add and then a word is on or off.

# Text classification with Naive Bayes

Given that in a particular class ,, so we've selected y equal to c, indicate we are now in Class C, so that's why we need to put that indicator in both terms, but given it we are in Class C, assuming we are in Class C, we could still the word could be on or off, so we need a Bernoulli model.

![Naive Bayes classifier with binary features x update](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/Naive%20bayes%20classifier%20with%20binary%20features%20x%20update.PNG)

We have a vector X, so a tweet will be a vector of $x_i=[10001001]$, each postion indicates a particular word and this goes from j=1 to d=8, each entry of the vector $x_i$ is Bernoulli, in the $y_i$ it is equal to 1 or 2, in other words class 1 or class 2. So i is the index over the data goes from 1 to n, j is the index of word features it goes from 1 to dupdate, and c is the index classes, it goes from 1 to captial C. So we have n observations iid, so we take the product over n and since we have c classes, we use a categorical distribution for C classes, so in other words we're pinning into class 1 or class 2, I could use a Bernoulli model here because we only have two classes. And we have this quantity $\theta_jc$ which just a probability that the j's element of the vector $x_i$ is on when you are in class c.

![Naive Bayes classifier plot ](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/Naive%20Bayes%20Classifier%20plot%20update.PNG)

And each class will have different distribution, for each class you will have many of these $\theta$s, and in particularly have d of these $\theta$s, because there are d words ,each word gets its own $\theta$, each word has a success probable clipping on or off what's going to allow us to separate one class fro another one is that these classes when you consider all the words they will have different distributions.

### MLE for Naive Bayes classifier with binary features x

![MLE for Naive Bayes classifier with binary features x](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/MLE%20for%20Naive%20Bayes%20classifier%20with%20binary%20features%20x.PNG)

I'm going to define so n is the number of data, here introduce get two more indices, two more summaries. one id gonna be the number of points of Class C, $N_c$ is just he sum over all the tweets and you count how many times y was of class c, we also count the number of times the tweets were of class c and in a particular word was on, so how many times did the word occur in the positive class and how many times it over occur in the negative class, thats $N_{jc}$, now if you want to get $\hat\pi_c$ which is just a probability that you will be of class c, what do you think what the probability that tweet will be positive or negative? What would be sort of an estimate for it?

### Predicting the class of new data

![Predicting the class of new data](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/predicting%20the%20class%20of%20new%20data.PNG)

Let's assume we've got these estimates, you use the precisely the model that we wrote, so our model is use Product of $\pi_c^{\amalg_c(y_i)}$ and small d Bernoulli variables and that's precisely what I'm gonna do. So given a new data point x*, the way I'm going to predict if it's of class c whether it's positive or negative is I'm just goint to multiply the prior of the class times Bernoulli for each of the D components of that vector x*, so I evaluate the model and I only need to evaluate it up to proportionality, because I oly have two classes positive and negative. so I computer each and then I just divide by their sum and that gives me the probability that the tweeter is positive in the probability that it which is negative and that's how we are going do predictions we're going to have probabilistic on these tweets. 

I just now going to assume I am of class c ,so I set this to one and then I just have the estimate $\theta_{jc}$ to the power of whether the word is on or off, so just evaluating the probability of a Bernoulli distribution, that gives us a prediction. So each time we type a words, it applies this equation to each of thoese tweets and then it sum to the number of postive or the negative and that's how you get sentiment for the tweets. 

**There only one warning**: I'm multiplying many probabilities now, and probabilities between number 0 & 1 when we deal with millions of tweets and possibly millions of words you're gonna multiplying millions of small numbers, using the computer multiply many small numbers you're gonna get 0 because the machines are rounding off. This is the problem when you invert matrices. To solve this issue , so we are using log-sum-exp trick technical.

### Naive Bayes classifier with binary features algorithm

![Naive Bayes classifier with binary features algorithm](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/Naive%20Bayes%20calssifier%20with%20binary%20features%20algorithm.PNG)

All we have to do is count, we just loop and we count the number of times, you estimate basically $\pi_c$ and $\theta_{jc}$ which are just count. 

### Log-sum-exp trick

![log-sum-exp trick](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/log-sum-exp%20trick.PNG)

I fyou have to take th log of the log of the sum of two very tiny numbers, in other words you're multiplying two tiny numbers, you exponentiate them, so that you can factor them out and then you take the log because the log is the terms of the exponentiation property, so you still have the right thing. You can write it in log e form, then you take the common factor and then you take the  log of that common factor out and now you just need to deal with exponentiating like bigger terms, e to the 0 which is e to the minus 1 whihc is 1 over 2.7, so it's a very useful trick to deal with under flow.   

### Log-sum-exp trick algorithm

![logsumexp algorithm](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/log-sum-exp%20trick%20algorithm.PNG)

If you don't do this trick, it will not work. Your code will not handle a million tweets for your to trian.  

And this is also good interveiw question if you want to apply for a text analytics company, like how to deal with under flows with multi normals.  

### Maximum likelihood estimate (MLE)

![MLE](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/MLE.PNG)

Rewrite this using $\pi_c^{N_c}$ and $\theta_{jc}^{N_{jc}}$, it is the same trick when doing for Bernoulli distribtuion in detail before which is if I multiply the same term to an exponent you just take that term to the sum of the exponents, in other words 2 to the 3 times 2 to the 4 is equal to 2 to the 3+4, that's the property that I'm using. When using this property i get an expression that's a bit more condensed, sum of the two indicators were the definition of $N_{jc}$, once I have this form the log-likelihood of the parameters which in this case are the $\theta$ and $\pi$ are just sum of a c=1 to capital C, $N_c$ log of $\pi_c$ as one term plus another term. 

When I have multiple coins are just the same thing, I take the log then I will differentiate equate to zero and I will get $\theta$ and I will get $\pi$

We gonna take the derivative of this term and wwe gonna estimate $\phi$ and we're going to estimate $\theta$. 

# Twitter sentiment prediction with Naive Bayes
The log likelihood involves two from, one is $\pi$, another is $\theta$, so when I take the derivaties, if I take the focus on $\theta$, the first derivate for $\pi$ form will be zero, inverse is the same. 

## MLE for $\theta$

![MLE for theta](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/MLE%20for%20theta.PNG)

Using the different l, just  at term for $\theta$ as $l(\theta)$. Then take the derivative, and I'm going to learn each coin separately. Then equating to zero. 

The result is the number of coin tosses in Class c, for word j and then the number of times you saw word j in class c divided by the number of times view you  are in class c.

So when we do maximum likelihood we recover what intuition tells us we're just doing coin flips.

## MLE for $\pi$

![MLE for pi](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/MLE%20for%20pi.PNG)

$\pi$ is a bit more interesting, it is the first term so $l(\pi)$, $\pi_c$ is the probability of calss, the prior probability, in the end it allows us to computer the posterior probability which is what we care about the posterior is what he used to decide whether the tweet is positive or negative.  In the problem is how to define the categorical distribution, in Bernoullion ,there are two possible, $\theta$ and $1-\theta$, but in categorical distribution, the sum of multi $\theta$ to 1, it need to let the algorithm know that. The parameterization is not telling us that, This could be an issue, so we need to add an extra thin to ensure that these features up to 1.  So in the cost function, this is sort of similar how the LASSO also gets implemented. So add another term with a weight $\lambda$, we will learn this weight by derive it. And the sum of the $\pi_c$ has to be 1. My cost function should penalize me if the $\pi_c$ is not sum to 1. 

The first term saying that my observations are of capital C classes, and 2nd term telling me that the sum of $\pi_c$ has to be one. If I go back drawing contour plots and so I can actually derive just like derive the equations for Lasso. 

When we have the $N_c = \pi_c\lambda$, we got the trick, I have n equations on equality, so I can sum both side over C, if I sum all the number of times you fell in Calss C, I will get the total number of times you belong to class C, which is n, it is the total number of observations. Then we learned the $\lambda$ must be equal to n.

The probability is the number of times you are positive divided by the number of times you're positive plus the number of times you are negative. 

## Bayesian analysis

How do we do in the Bayesian way, the bayesian way will allow us to deal with the fact that sometimes some words never appear, we use the bayesian analysis to deal with the fact that sometimes you might not have observed the word but doesn't mean wthat you shouldn't assignn some probability to it. 

![Bayesian analysis](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/Bayesian%20analysis.PNG)

In order to do Bayesian inference, we do just like a coin model, we have likelihood, we will construct a prior, we multiply the prior times the likelihood and that gives us the posterior. In this case, the likelihood is the product of d times C bernoulli coins plus a product of Dirichlet distribution over the class probabilities, $\pi_c$, so you know a categorical distribution of the $\pi_c$, the natural prior for the categorical distribution is Dirichlet, the natural prior for the Bernoulli variables will be $\beta$ distribution. $\pi$ is the categorical distribuiton, so it makes sense to use a prior that is there is Dirichlet. $\theta_{jc}$ is benoulli prior, $\beta_1$ and $\beta_2$ are hyper parameters, and I have chosen thoese parameters to be the same for all words. This one could be changde, you could say that you want to put more prior belief on the words, and we have see the prior is better tahn the maximum likelihood. In other words, I have d times C $\beta$ priors, one for each word in each class.

![Bayesian analysis posterior](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/Bayesian%20analysis%20posterior.PNG)

And this you can see, my likelihood in my prior have the same shape, so I just need to sum the exponents and that's my posterior. The post mean ($\pi$) means posterior mean ($\pi)$, and post mean ($\theta_{jc})$ means posterior mean ($\theta_{jc})$. This is the estimate you want for the Bayesian that the full posterior is the solution.

**Important**: Even the $\beta$ distribution , mean doesn't make sense all the time. 

# ML 26 - Optimization

**To understand neural networks, you need to know optimization and logistic regression**.  Optimization builds on calculus, when we take the frequentist approach to learning, we do maximum likelihood, we often take an objective fucntion the log likelihood, we differentiate it and equate it to zero. And that gives us the estimates, now in everything we've done up to now except for the Lasso when we equate to zero and solve for $\theta$, we will get answer for $\theta$ like ridge. So we were always able to get a closed form solution, we saw with the Lasso that it was not possible learning more to get a closed form solution, and in fact for most problems it is not possbile to get closed form solutions. Just by depreciating n equations to zero.  In order to find a place where the derivative becomes flat, that is in order to find the minimum of the error function for the maximum of the probability to problems are equivalent. In order to find a place where the derivative becomes flat we're going to follow the derivatives

On other hand the Bayesian algorithms are based on conjugage analysis and we saw how to do that for the Gaussian model. Methods we have to develop ways of solving integrals in high dimensions and so there are different things out there called variational methods, there's techniques called Monte Carlo.

Hessian is just second derivatives but when you have more variables and then we're going to discuss three algorithms, gradient descent following the gradient immediately also known as steepest descent. Newton's method which not only uses that the gradient but also looks at the curvature of the error function that we're minimizing, we will look at stochastic gradient descent which is what you need when the data when you don't get to see all the data but when the data comes to you one at a time online learning for experience or for application. All these techniques are applicable to neural networks but today we will apply them to the linear model, they also be available to find the least square solution. 

## Gradient vector

![gradient vector](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/gradient%20vector.PNG)

As we've seen up to now, we always talk about an objective function for a log likelihood which is an objective function as well and the objective is to find a set of $\theta$s, a multivariate vector of $\theta$s that allow us to find a solution . For example when we deal with quadratic problems, we have a intercept and a slope and we try to find the minimum of a function. If we have a quadratic function $f(\theta_0, \theta_1)$ like this ball, then the gradient in this case, these arrows here, the gradient evaluated at a particular $\theta$, say $\bigtriangledown_{\theta}f(\theta_*)$ is just a derivative of function to $\theta_0$ and $\theta_1$. $\theta$ is a vector with comonent $\theta_0$ and $\theta_1$, if we're fitting a line to points, we have two parameters the slope $\theta_1$ and intercept $\theta_0$ in our objective, then our objective then is to find the $\theta_0$ and $\theta_1$ that goes through the points. The top is the optimal solution, we're still dealing here with the regression problem, the gradient is just a derivative or with respect of each of the parameters and evaluated at $\theta$ equal $\theta_0$. I try to say the gradient is a vector, it has two component and also that the gradient is a function of $\theta$, you get a blue vector everywhere, so you can evaluate the gradient at any point. 

We will not have single optimal, but we wil have many optima but if we follow the gradients we will get to a place that has lower cost, we will not guaranteed always to find the optimal one, the best one, but the hope is that we will find one value of $\theta$ of the $\theta$ vector which will allow us to make reasonable predictions. 

## Hessian matrix

![Hessian matrix](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/Hessian%20matrix.PNG)

It is a generalization of the gradient and it's the finest nature is the matrix of second derivatives so if you can think of all the possible to second derivatives you could take and you were to place them inside of a matrix, you would get this quantity which we call the Hessian, so it's the derivative of a function with respect to each of the parameters twice in the diagonal and then you get the cross derivatives of the data, so that's just a definition.

![offline learning](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/offline%20learning%20function.PNG)

If we doing offline learning, if we have a set ose f data / a batch of data, so we've seen n data, then in machine learning typically the objective is to minimize an $f(\theta)$ which is the sum of objective functions, $\frac{1}{n}\sum_{i=1}^n f(\theta, x_i)$, in least squares we typically minimize $\sum_{i=1}^n(y_i-x_i\theta)^2$, so in **least squares $f(\theta)$ is just quadratic difference**. Now the gradient is just the derivative of the function and so an important properties that gradients like derivatives because their derivatives. In fact they still linear operator so if you have a the grradient of a sum is the sum of the gradients. So we can just take the gradient inside the sum when we taking the gradient of $f(\theta)$.

$\frac{1}{n}$ it is not important because the algorithm will have a scale factor that will subsume the 1 over n, if put the $\frac{1}{n}$ in linear regression, that would not affect the answers, it is just a scaling fact. 

## Gradient vector and Hessian matrix

![gradient vector and hessian matrix](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/gradient%20vector%20and%20hessian%20matrix.PNG)

If we have known linear regression, so we know that when wo do least squares we're just solving the quote the sum of suqred errors, so we know what the objective function is, the gradient, we can computer it if you wish one by one like I did here one by one term at a time. A faster way of doing it is just use the rules of matrix differentiation, so the derivative of this with respect to the vector many times throught out the course. We can eigher use sum, we can use matrix differentiation. But now we can also computer the Hessian  which is the second derivatives a centry. Up to now we've not built an algorithm all we've done is, let's get what is the gradient and what is the Hessian for linear regression. Later will show you an algorithm that only uses the gradient in order to find the $\theta$ and then we will discuss an algorithm that ues both the Hessian and the gradient to find the $\theta$ and that algorithm is called newton's method. We'll see the newton's method is so good that is only needs one step, one iteration to find the best solution whereas the gradient descent might require many steps and it's quite tricky to decide how many steps you will need. 

### Steepest gradient descent algorithm

![steepest gradient descent algo](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/steepest%20gradient%20descent%20algorithm.PNG)

Thi is the first algorithm introduced, it always following the steepest direction.  We start somewhere in the function at the bottom of plotting the contour plots and what we do is we just travel perpendicular to the contour plots and if we go in a direction that's perendicular to the countour plot we will get to the center of this quadratic function. So in the function what's happending is that we're going to the minimum. So we're just descending to find the minimum and when you descend to find the minimum you're just going perpendicularly to the contour plots. In 2D, it  is the same story.  But If I start from the different location, I will get to a different minimum, so we start at any point we follow the gradient and we go to a minimum. In the quadratic case which is least squares there's only one minimum and that's why there was only one single $\theta$ and there is ony one $\theta$ that's the solution. But if we do neural networks that will be many solutions and we'll be able to do is find one solution and **in order to know what is a good solution we'll have to do cross-validation** and many other tricks. 

Now in math what we're saying is that the neew $\theta$ at iteration k+1 will be equal to the old $\theta$ and because we're minimizing we going in the opposite direction as gradients point toward the maximum, so minus and we're going to decide a step size, this step size also know as the learning rate $\eta$, you can think of it as how fat you're going how much you trust the gradient. For now we will choose $\eta$ by hand and g (gradient) is just a vector. If there is two components, the d equal to the gradient has two components. Esstentially  we have the d equations, and it makes sense that we have d equations because we need to update the $\theta_0$ and then we need to update $\theta_1$, all the way up to $\theta_{d-1}$, so we update all the features. It's not obvious that it will converge at all in fact, so we're gonna come to that issue. So how you choose convergence  of this algorithm, we will depend on the choice of $\eta$ as we will succeed. 

So the algorithm is you're at any random point, you just pick a random initial $\theta$ and then you just keep updating $\theta$ using just that one single equation, so the algorithm requires that you compute the gradient and also it requires that then you take a step in the direction of the gradient, then evaluate the gradient step and so on. 

### Steepest gradient descent algorithm for least squares

![steepest gradient descent algorithm for least squares](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/steepest%20gradient%20descent%20algirithm%20for%20least%20squares.PNG)

For the linear model, we know the derivate $f(\theta)$, there is a 2 in deriviate, but it is disappear in steepest gradient descent algorithm, the reason because I can subsum it in the $\eta$, $\eta$ is an arbitrary number. And the gradient evaluated at $\theta_k$

**What is this actually doing?**

You can think of this as an iterative algorithm, if you have several points, it fits a line to the point at random, so you start with a guess and when you do an iteration, you find another line,  another line, until you find the line go through the points. So in each update you're find a new value of $\theta_0$ and $\theta_1$, $\theta_0$ and $\theta_1$ is all your need for the line, because the line given by one you konw its slope and its intervept. So the algorithm is just updating the lines as you go. The size of the step will depend on $\eta$ parameter as we point out this parameter can determine convergence or not. 

Before coming to newton's method, the gradient descent for the linear model is fairly easy, we compute the gradient and you can either go least square or differentiate maximum likelihood, and basically it gives you the same expression. Once yu have an expression for the gradient, then the algorithm is just the new $\theta$ is the old $\theta$ by going the opposite direction of the gradient by an amount $\eta$ which you can choose to be 0.2 or 0.8 and then times the gradient.

![how to choose the step size](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/hwo%20to%20choose%20step%20size.PNG)

If the derivative function $\bigtriangledown f(\theta)$ is so small where the derivative becomes almost flat that you might be going down to a function. In a big region where the derivative is not flat but it looks almost flat so it's almost zero, if you multiply by a small number you might get somthing that is approximately zero, if it must be zero means the $\theta_{k+1}$ will be equal to $\theta_k$ you won't see any updates. So if there is a choice of $\theta$ there will give you just the right speed to get there, $\eta$ is a tuning parameter, there are **techniques for choosing $\eta$ called line search**.  what is just a way of testing the function to see how and readjusting your $\eta$. So basically you take a few steps if you don't see much of a changing in the function height, you choose a bigger theta to take bigger steps if you see huge changes then that means you probably are oscillating, so you reduce the ste size. 

### Newton's algorithm

![Newton's algorithm](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/Newton%27s%20algorithm.PNG)

Gradient is the method may not converge, Netton's method on the other hand will converge. The gradient descent says $\theta_{k+1} = \theta_k-\eta g_k$, $g_k$ is the gradient, so gradeint descent just use $\eta$. Newton's method replaces $\eta$ by a whole matrix, this matrix essentially tells you how fast you should go in each of the $\theta$directions. You have  multivariate function , so here in 2D, it's telling you how fast to go in the x-direction how fast to go in the perpendicular direction in the Y direction. 

What a lot of people doing practice with neural networks is they just choose those by hand and sometimes they choose an $\eta$ to that of the form1 over n, so we bring back this 1 over n because basically means that you do big updates in the beginning and then as time goes by your updates become smaller and smaller, $\eta$ you can choose constant or choose the number of steps. The line search intuition, no much change in the height , you increase it and if there's big changes in the height you decrease it if you do that you can see that it is a function of time of K. 

Newton's method will however address this problem which will give us a way of automatically choose the learning rate but it's not just going to choose the learning rate as a scalar but it will choose it as a matrix because there's going to be learnng rates along all possible directions. The catch of Newton's method is that it requires a matrix of derivatives if you happen to have two parameters, your vector gradient has two entries in your Hessian has four entries because the Hessian requires the derivatives with respect to $\theta_0\theta_1$ and $\theta_1\theta_0$ and then $\theta$ two squared and $\theta_0^2$ and $\theta_1^2$.  so you have four terms for the Hesse, now if you have many parameters. For example the twitter classification you guys have to implement if you were to try linear regerssion then you may have thousand parameters for a million parameters, then your gradient vector with a million parameters would be of size a million. Your Hessian would be of size a million by a million, so that would be a lot of storage. So Newton's method will turn up to be much faster and it gets rid of this issue of having to choose the step size but you pay for it in storage, so there's a trade-off, there is no free lunch. 

So Newton's method chooses as the step size the inverse of the Hessian a ietration K ($H_K^{-1}$), and we can derive this algorithm if we think of a Taylor series approximation. Now Taylor series approximation just means expand the function $f_{quad}(\theta)$ is an approximation up to some degree in this case we're going to go to degree two of the function evaluated at a point times the derivative $g_k^T$ and then interval $(\theta-\theta_k)$, and then the quadratic term involving the second derivatives, $(\theta-\theta_k)^TH_k(\theta-\theta_k)$. So it's just an expansion of a function in terms of its derivatives.

So we create this quadratic function to have the same slope and same surfce if we differentiate it with respect to $\theta$, we will get 0. Because $\theta_k$ is a specific value of $\theta$ and we will get the gradient $g_k$, and we will get $H_k(\theta-\theta_k)$, and if I equate this to 0,   then get $-H_,^{-1}g_k=\theta-\theta_k$ from where the Newton algorithm step above comes. 

I want you to be aware of is that the Hessian requires now that you go and compute the second derivatives, so it's more work, more store this matrix and that you invert this matrix, inverting a marix is one of the most expensive steps that computers have to deal with. To large extend the progress in science has depended on a polynomial and cube, and cube is what it cost you to invert a matrix to solve a linear system and eigenvalues and eigenfunctions. All these problems have this fundamental bottleneck and cube. And when n is 20,000 genes or 2 million n grams for like wikipedia titles, inverting a 2 million by 2 million matrix is not an easy task. 

However the small problems Netwon's method is the right method because it converges very fast, you don't need to choose the $\alpha$, so you have this matrix H and then think the Newton's method as a quadratic approximation to a cost function, so there is a cost function f What Nwton's method does, it fits a quadratic to $f_{quad}$, it will fit it at the point $\theta_k$ and what you do is you find the minimum of the quadratic. The quadratic has only one minimum, so we differentiate the quadratic, we equate to 0 and  that gives us that equation which is the equation of the minimum. So it saying the optimal feature if you saw here in the **optimal $\theta$ is $\theta_n - H^{-1}g_k$**, so the **optimal feature which will be $\theta_{k+1}$**.

### Newton's as bound optimization

![Newton's as bound optimization](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/Newton%27s%20as%20bound%20optimization.PNG)

One way to think of case here where we have the function f of $\theta$, then the quadratic Taylor series approximation. It's just a quadratic function that near the value of $\theta_k$ provides a very good fit, but at $\theta_k$ the two functions are tangent to each other. In particular at $\theta_k$, you ensure that these functions have the same slope and the same curvature that's basically what a second-order expansion is. So the Newton algorithm what it's doing is find the next $\theta_k$ as the minimum of the quadratic function, we don't know how to minimize the function so what we do it we approximated at $\theta_k$, we approximate it with a quadratic function, the blue line that quadratic function has the same slope in the same curvature at that point, we go down that quadratic function all the way to find its minimum, that'sthen our value of $\theta_{k+1}$ and then we repeat it, we again fit a quadratic and we follow it to minimum. Essentially using an idea which is since we can't optimize the original function, we optimize the function that we know how to optimize which is a quadratic.

Newton's method just minimizing quadratic uppper bound and the main advantages we get rid of each other instead of having to choose either by hand we just need a matrix of second derivatives H and then we're done. 

### Newton's algorithm for linear regression

![Newton'salgorithm for linear regression](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/Newton%27s%20algorithm%20for%20linear%20regression.PNG)

If we do update for linear regression we would have $\theta_{k+1}$ is equal $\theta_k$ minus $(x^Tx)^{-1}$, and multiply the common factor times in two terms of inside the bracket {$x^Tx\theta_k-x^Ty$], $(x^Tx)^{-1}x^Tx$ is just the identity

# ML 27 - Logistic regression

## Advanced: Newton CG algorithm

![Advanced Newton CG algorithm](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/Advanced%20Newton%20CG%20algorithm.PNG)

There 's the pseudo code for Newton's method.

## Online learning - aka stochastic gradient descent

![online learning](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/online%20learning-stochastic%20gradient%20descent.PNG)

when you do linear regression, the gradient descent way of computing $\theta$ is going opposite direction of the gradient and then you had this expression sum. When we have n data points, this is what we call batch learning, we take the whole batch of data and we update a grade. There is another way of doing gradients, called the online way, every time you get the one data point you do the gradient with one data point, it called stochastic gradient. why this is useful? If you 're learning a linear regression from tweets if you're trying to take the words of a tweet and you're trying to predict a poll or price of a stock from twitter, so what you do is you take that input vector x, which is basically the sort of the word is present zero or 1, if you have a very large vector, your $\theta$ will be very large 10,000 dimensional also, and then you just trying to predict, each day you get one update , get one x. Twitter get input and never stop, you're always learning and in fact you're always learning, you will also adapt because the statistics of the problem change, your $\theta$ will also change. the $\theta$ might never converge to a particular $\theta$, but $\theta$ might be always tracking the solution, so what we can also do if we do have a batch of data, for the streaming machine learning, streaming problems like Twitter where the data is always coming. What you doing environment monitoring is the same situation.  And there is somthing in between what's called the mini batch approach, which is super popular in machine learning. Neural netwoks use exactly the same working, the only think that's going to change is the error function but it's always going to tbe the same thing we compute the derivative of the gradient, then we have these three choices, either do batches, online, or mini-batches. 

### The online learning algorithm

![the lonline learning algorithm](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/the%20online%20learning%20algorithm.PNG)

When you do online learning, you don't go straight, because you only see on data point, so you don't go straight to the bottom and sometimes your error gets worse, sometimes you go back up that's not necessarily 

# ML 29 - Neural netwoks

## Logistic regression

![logistic regressionn](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/logistic%20regression.PNG)

I shown you how to apply gradients a Newton's method to linear models, now I'm going to tell you how to do that for a neural network with one neural.me that all the coin flips are independent, so I take 

We will be given x, y data points, instead of being like linear gregresson, y is real number, now the y is binary, 0, 1. x is given, y is the variable over which we will have a distribution, it is stochastic here. $\hat y$ will be the prediction of y, if we have the 0,1 data in y, it is the data of tossing a coin in seeing. So the right model for 0,1 , it is the Bernoulli distribution. But now the success of y probability is depend on x, which is interesting. My y's, I assume that all the coin flips are indepentdent , so I take a product from1 to n, and each y is just Bernoulli distribution with success probability $\pi_i^{y_i}$, base on this,  we haven't done anything new which is just coin flips. But now what we're saying is that $\pi_i$ is itself parametrized and essentially we're doing a linear regression, we start with x, we fit the line x times $\theta$, that's essentially the equation that we have here that's the equation of a plane and then we squash it through a sigmoid function, and the reason why we squashes through a sigmoid function is because we know that this function here is between 0 & 1, so we get a probability. so it's a trick that allows us to get a problem be able to interprent $\pi$ is about the probability. So there's two things here, one is your define a probability on the y, the distribution is over the y, and  compositionality where the success probability of the y, $\pi_i$, and the function of the input and this is how we start introducing probabilistic models for more complex settings, as we go into neural networks. $\theta$ is a continuous variable etween minus infinity and infinity, it's completely unconstrained, but $\pi$ will naturally be constrained to be between 0 and 1, so we don't need to introduce funky constraints that require that the $\pi$ sum to 1. 

**Once we have a likelihood, how do we find what $\theta$ is?**
The usual procedure, you take the log, differentiate, equate to zero. only one difficulty in this case when you differentiate and equate to zero. You're not going to be able to solve for a closed form for $\theta$

## Sigmoid function

![sigmoid function](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/sigmoid%20function.PNG)

Sigmoid function had the important properties between 0 & 1.

## Linear separating hyper-plane

![linear separating hyper-plane](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/linear%20separating%20hyper-plane.PNG)

In 2D, the sigmoid function will look like the surface that I'm showing here on the right-hand side and then we're going to threshold, so we threshold this function at say 1/2 when the height is 1/2 and then evertything to one side of 1/2 greater than 1/2 is considered to be of the class 1 and another side to be class 0. And now we have a classifier, whenever you have a new x, you compute what's the probability, the y is equal to 1 for x, if that's greater than 0.5 you assign it to 1 class, and if it's 0 if it less than 0.5.

## Gradient and Hessian of binary logistic regression

![gradient and hessian of binary logistic regression](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/Gradient%20and%20Hessian%20of%20binary%20logistic%20regression.PNG)

 When we differentiate this and equate to zero, we get the following expressions for the gradient and Hessian. The gradient is given by expression, $\sum_{i=1}^n x_i^T (\pi_i - y_i)$, it comparing y and the $y_s$ are the data either 0 or 1, the $\pi$ is the variable between 0 and 1, we comparing the prediction against the actual data, if your predictions are different than the actual data, your gradient update is big because basically if your prediction don't match the data, you should learn and learn in big steps, whereas if your predictions agree with the data then your update is small. We could implement gradient descent but it's also possible to go in compute Hessian. In this case, for logistic regression is very easy to get the Hessian. Once you have the Hessian which looks at the curvature, the second derivative and as you can  see somewhat related to the entropy, $\pi_i(1-\pi_i)$ to the variance of the predictions, then you'll be able to implement Newton's math. Newton's math requires the Hessian and g. That's how we solve the logistic regession, we just try Newton's method.
 
Hessian is positive definite which in the multivariate case can be just interpreted as it has positive courage that made you having a postive definite Hessian with all positive eigen values means that you have a ball that you have a convex function and that ball is not symmetric, but still it has a unique minimum. If we follow the Newton's method, we can attain the optimal in this problem. So logistic regresson is still considered a fairly easy problem to solve, 

### Iteratively reweighted least squares (IRLS)

![IRLS](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/iteratively%20reweighted%20least%20squares.PNG)

In the particular case when we implement Newton's method, showing you here that we have the expression of the Hessian and gradient from the previous, and if you substitue those in the estimate for $\theta$ and do some simplificatioins to get something that's a bit more computationally efficient you get an algorithm, it calls high RLS or iteratively reweighted least squares, because the expression looks like a least squares expression, except there the weight given by the matrix.

Following are the Newton's method for doing logistic regression code :

![IRLS code](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/IRLS%20code.PNG)

And then  the main function just loaded data, estimate and then you just need to  call newton's method to give you a new $\theta$, $\theta=irls(Xtrain, ytrain)$ and then you evaluate the training error. 

![IRLS main function](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/IRLS%20code.PNG)

## MLP - 1 neuron

![MLP with data](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/MLP%20with%20data.PNG)

This is your first neural network, its logtistic regression with one neuron. Our data is show on the slide, you would have two columns of inputs.  The boundary for logistic regression is  a line 

![MLP](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/MLP.PNG)

You have input $x_i$, it gets weighted by $\theta_2$, you add to it $\theta_1$ that gives you u, and then we put it through a sigmoid function and we get $\hat y_i$, the $P(y_i=1|x_i, \theta)$ is just the Bernoulli distribution with success probability $y_i$.

![MLP with one neuron](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/MLP%20with%20one%20neuron.PNG)

The output of one neuron is just a number between 0 and 1, because it's the output of a sigmoid, that's why we use a sigmoid , so we can interpret the output as a probability. 

## MLP - 3 neurons, 2 layers

![MLP with multiple neuros in data example](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/MLP%20with%20multiple%20neuros%20and%20layers%20in%20data%20example.PNG)

However we want to do better, the data might not be linearly separable in which case we want to use nonlinear functions like, example in the picture there is two on-lineare functions, the blue linear function is for a neural network that have very few neurons in the hidden layer, probably three or four and then the purple line would be for a neural netowrk that would have like 20 neurons in the hidden layer, so you sou get zero training error but ee the trade of the more neurons the better your separation with lots of neurons, but probably that the test set will do poorly. How to choose the proper neurons, agian do with cross-validation. Another thing you can do is add an L2 regularizer to the neural network because the L2 regularizer will set some of the $\theta$ to zero. It will smooth the decision boundary. 

![enter image description here](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/MLP%20with%20multiple%20neuros%20and%20layers.PNG)

We will have an $x_i$, it is the intermediate inputs. the $P(y_i|x_i,\theta)$ still model with a Bernoulli distribution, nothing has changed in that respect becase we saw our binary data, the only think that has changed is that the success probability $\hat y_i$ is now given by a much more complex function. 

The question to you is why do I bother making the function more complex? Because initially my boundary was linear but now by using several sigmoids I can actually have a nonlinear function.

![MLP with multiple neurons and cost](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/MLP%20with%20multiple%20neuros%20and%20layers%20and%20cost.PNG)

The only difference in logistic regression is, the mapping to go from input to output is a simple one, we first computer u, then we put this u which is basically a linear regression, then we put the linear regression through sigmoid to given a number between 0 and 1.  The neuron networks a bit more involved, because now you have two linear regressions , one for each u. 

The treatment before for a neural network if you have many data from 1 to n, then the overall likelihood is just the product of the likelihood of each data point, so its product of Bernoullis. And if you want the cost function, just like least squares if you don't want make probabilistic sense, but all you want to minimize is a cost like the sum of square errors. It's possible to transform a maximum likelihood problem to a problem of cost minimization cost. The cost also could be called as objectives or energy. So minimizing the cost is essentially the negative log likelihood, for the Benoulli distribution if you take the log and negate, essentially you get this very simple expression which is the cross entropy. So you want minimize entropy, means you want minimize uncertainty, there is clearly mapping between cross entroy and likelihood.  so the objective function makes sense.  Like quadratic error in linear regressino is just the negative log of a Gaussian. Cross entropy is the natural function for maximum likelihood when you have binary observations.

### MLP - Regression 

![MLP Regression](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/MLP%20regression.PNG)

Here is the example for MLP, we start looking the regression and without different output neuron,  instead of using sigmoid, I'm going to use the identity in $u_{21}$. $\hat y$ is just the combination, the $\theta_5$ control the height of the sige the y is 0.6, could I every predict y if my $\hat y$ was between 0 & 1? moid, $\theta_6$sort of squashes it up and down, $\theta_1$ moves the sigmoid left and right and $\theta_2$ squashes it horizontally. So either thin or fat sigmoid, we can squash it in both directions and  we can lift it and we can shift, so we have very nice breaks basically and you can add up many of these breaks and you can construct any function. If you negate the $\theta$, we can change the direction of  the sigmoid. 

Why using the identity function? 

![MLP regression with identity](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/MLP%20regression%20with%20identity.PNG)

If I use the sigmoid in $u_{21}$ instead of identity, what would be happened for $\hat y$, in what range? $\hat y$ inbetween 0 & 1, I have y is 0.6, whether I could prdict y if my $\hat y$ was between 0 & 1? The sigmoid isn't creating a scaling to 0, 1. I'm going to interpret in the linear case as the mean, the prediction. And I want the prediction to be a number between minus infinity and infinity that's why I keep that linear neuron. Now the picture says it all that we're doing regression from x to y, any point is distrobuted according to a Gaussian. If any point distributed according to a Gaussian then the pictures as it show tthere you have a specific $y_i$ if you evaluate the height of the point, the is point that's $y_i$, the value at the line it to fit neural network fit is just $\hat y_i$, that's the function $y_i$, the difference between $\hat y_i$ and $y_i$ is what I 'm showing in red, that's the cost. Each point has Gaussian distribution and by each point I'm in each y, there is no noise in the axis again, just the only thing is we change the line for an arbitrary curve instead of line we now putting basis functions together doing an arbitrary curve.  In effect neural networks that's the only thing they bring into linear regression that they're allow us to go nonlinear, but the probabilistic model that is saying the cross-validation is the same. Now we have a nonlinear functioin, so i'm just saying that $\hat y_i$ is the function which is the neural network that depends on $\theta$ and the index i. 

### MLP - Multiclass 

![MLP multiclass](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/MLP%20multiclass.PNG)

Now let's look at another more interesting case, you want to classify data not into 1 or 2 classes, you want to classify into more classes. We're going to do a 1 of K encoding,  if a data point is in class 2, we are going to label the $y_s$ in encoding in three classes, using this method I can present to the output. I will get in a decision boundary or the discriminant boundary, which is also called that separates in three classes. And also we want probabilities.

![Softmax function](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/MLP%20multiclass-softmax%20function.PNG)

We will define the probability that the label says of class 2 which is 010 given the input  and $\theta$, using this function because its gonna give us something postitive, and because I normalized the outputs, I take the $e^{\hat y_2}$ that guarantees that its positive, if I normalized for all the three classes, then I am ensuring that I will get them sum up to 1. This function is called as softmax function. It's used with neural networks but it's also used with a lot of other machine learning techniques.

First if $y_i$ is negative, I'm using  an exponent function which makes everything positive andsince I normalize its making this term between zero and one. If I put sigmoid there, actually it should still work.

![MLP Multiclass with likelihood](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/MLP-Multiclass%20with%20likelihood.PNG)

So the likelihood for a single data point, it's like the Bernoulli but now for three classes. It's just like Bernoulli trick, it's a categorical distribution for three variables. 

If you want the cost function, it is again cross entropy error function, but with three classes. So you minimizing uncertainty, it is equivalent to maximizing likelihood. 

Once we know the cost functions, or equivalently we know the likelihood, we know how to regularize them by adding $\theta^T\theta$, and we know how to proceed to get an estimate of $\theta$, we just differentiate the log likelihood, we equate to zero and solve for $\theta$. In this case we can't solve $\theta$ exactly, so instead we will do gradient descent or Newton's method, and if you do gradient descent you can do mini batches or search online or the full batch of data. 

### Backpropagation

![backpropagation](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/backpropagation.PNG)

We're going to derive expressions for the gradient of neural network of those cost functions, those equations are telling us how to update a new $\theta$. In the first case,batch case, at iteration, it will true for each $theta$, so we do the gradient for each $\theta_j$, the old value $\theta$ plus the gradient with respect to that $\theta_j$, that's how we minimize the cost function of the gradient. for online we wouldn't take the sum over all the data, we will just take the current data. 

You will see the patterns here, all these derivatives look to same, so you also can put all these as a single matrix vector multiplication squash that ector through is multivariate, it's asigmoid that just gives you an output the vector in python and then the derivative computation.

![backpropagation with derivative](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/Backpropagation%20with%20derivative.PNG)

I'm going to do regression, using a quadratic error function and looking at the online problem, so only looking at the current error at time i, firstly doing the differentiation, then derivative with respect to $\theta_j$, then following the chain rule. The first ternm is done, it just the difference between the prediction and the actual y. The next term is the derivative of the ouput with respect to the $\theta_j$, so in roder to computer the gradient in your neural network, it is respect to each parameter. That's the chain rule, for neural netwoks it ht x. as a special name, it's called backpropagation. So what we get the derivative is using forward pass to get all the O's.

![Chainrule derivative](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/backpropagation%20with%20chainrule%20derivative.PNG)

If I want take derivative of $\hat y$ with respect to $\theta_3$, you need following the path from the output to the parameter by doing the chain rule. Sigmoid basically that sigmoid times 1 minus the sigmoid, so you get the parameter $\theta_7$ times sort of term times input x. 

# ML 30 - Deep learning

## Regularization

Just like we did for linear regression, our cost function for  all the parameters, the prediction $\hat y$ is the function of the input and its a function of the parameter. It will get good result if add a regularizer on cost fucntion for the linear case, $\lambda R(\theta)$ is the regularizer.  The $ R(\theta)$ could be the L2 norm square, so that would give rise to the ridge, and the ridge in neural network has a name, weight decay. Or using the L1 norm, this one forces two stages to go precisely to zero, if we have several inputs, we can arrange for some of those inputs to disappear, there will be effectively we can decide which inputs are relevant. We can do 2, because something more smarter,we can introdup. ce regularizes that only look at a subset of neurons. ping all the parameters to go with tone neuron and 
The regularizer $(\theta_1^2+\theta_2^2)$ is a sum over the neurons for each neuron in the hidden layer, it's the sum of the L2 norm of the weights corresponding to neuro, so it's an L1 of L2 because the sum of something positive is just L1 norm, essentially I'm grouping terms, grouping all parameters with one neuro, and I computer the L2 norm of that parameter. My penalty now it's gonna be $\lambda$ times each group. That way I will try to force a whole neuro to be switched on instead of just switching a single link, this is called L1 L2 norm. In general, L1 and L2 norm is just the sum, L2 norm of the vector $\theta$,  in group j, there's still a group of vectors $\theta_j$. 

### Unsupervised learning

![NN for edge detect](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/NN%20for%20image%20edge%20detect.PNG)

The cost function for the PCA, was essentially given a matrix of images X find two matrix B and C that approximate X, just find a reconstruction of X using two matrix B and C, for PCA we discovered that those matrix where B would be equal to use $\sigma$ and C would be equal to $V^T$, in other words the SVD is the optimal L2 reconstruction, and add an extra term which penalizes the magnitude of the V vectors, that we would then get something called sparse coding which looks like left side image. Each of these guys is one of the filters that gets active when you look at an image patch since that the interpretation. Because we take a lot og image patches and we convert the image patches two vectors, we computer the SVD and we have the V matrix what we do is, we  go from a vector to an image and that each of those plots is one of the V's eigenvectors. If you apply regularization, you get something that looks completely different, now each eigenvector starts looking like an edge detector. Because it's an image that on one side is there's light white and on the other side there's darkness. So it's looking for things in the image where there's strong contrast. Now we argued that we want machine learning systems that look like this. Each neuron detects a different edge, so being able to construct a machine learning technique that allows us to detect edges is quite useful.

## Autoencoders (and sparsity)

![Autoencoders](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/Autoencoders.PNG)

Here is the solution for training neural networks in an unsupervised way. The idea is as an input you take the image and you make the NN predict the same image as the output. So what I see is my training data. You create hidden layer and you want that hidden layer to have less neurons than the size of the image, so you're able to compress the image and from the compressed image, you're able to expand again ad you still get the image. In other words you're able to do the reconstruction, so if you could encode and decode the image which means you learned the image. 

### Unsupervised Feature Learning/Self-taught learning

![unsupervised feature learning](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/unsupervised%20feature%20learning.PNG)

There's another advantage of having an architecture like this. I can take billions of images of Africa, and I can train the NN, when I move to Canada that NN still useful. If I learn all the types of images that exist in Africa, and when I move to Canada, then I introduced to a new thing in Canada that I need to learn to recognize. I just need take through outputs of the nerual net activations, and then all I have to learn is three parameters, so I just do logistic regression, the idea is I pre-learned all the features. This is called transfer learning or self-taught learning. The idea you learn some theories, and then you're able to transport it to a different world. And this is essential we learn features in a supervised way, we learn detectors, sort of high-level features, and move to another place in order to be able to quickly recognize. 

### Sparse autoencoders

![sparse autoencoders](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/sparse%20autoencoders.PNG)

There are two ways in which we could train a neural network like to reconstruct. If we had a neural network with linear units not sigmoidal units, we would do is going from the input to the output $Wx^{(i)}$, in other words, we have the input, hidden layer, then we go back to the input. $\hat x$ is our reconstruction.  we are minimized the difference between our reconstruction and x and then we add to it a regularizer. The regularizer is very important, and  one of the thinkgs you can do is low-cost function that actually a smooth way of approximating a L1 regularizer and you could also use sothing like the group L1 regularizer. Instead of using linear neurons $W^TWx^{(i)} - x^{(i)}$, you could use a sigmoid neuron, $\sigma(Wx^{(i)}+b)$, so you feed the input through a set of sigmoids to encode and then you decode, $\sigma(W^T\sigma(Wx^{(i)}+b)$, so we have two stages, encoding and decoding. It turns out that in corder to get filter detectors. Each of the images on the left side is the weights that go from each of the inputs to a single neuron. $P_1$ means pixel 1. It turns out if you don't add a regularizer then you would just get PCA basis. You will go with a linear network, you will go back to basically having sinusoid and that the sort of global asis don't match what we observe in $V_1$.  So in order to have H detectors, the key is to add the regularized, in fact we don't even need the nonlineaer functions we could just use linear neurons and just a good regularizer.

## Deep learning : more layers

![NN with multiple layers](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/NN%20with%20multiple%20layers.PNG)

Everything since we discussed so far are single layers, how do we deal with more layers? We do this, what we call the greedy layer wise training and this was sort of what gave rise to this whole field of deep learning that's become very popular lately. The idea is we are going to start with the image has the input and we try to predict the image and what that allows us to do is to learn what the hidden layer neurons which people often call the features. Humans do not have the capacity to introspect and be able to actually come up with the rules. 

We have the first layer of features and then what we do essentially is we take these features and we do a further encoding to get a second layer of features, so that first layer of features is trying to predict itself and so on. So now if we take the first layer plus the second layer plus the third layer with we've now created a sort of a deep architecture, we start getting neurons embedded within other neurons, so when you build then a classifier, in the end you would just stored this, these are your features given the features, you can just learn a classifier just like we did before for transfer learning.

![Deep NN](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/Deep%20NN.PNG)

You learn the first features, the second features, you learn the final classifier and if you put them all together you get a deep neural network. The reason why we train them like this, because if you just start by writing the full neural network and run back propagation on it, it's actually  very expensive, your error terms often have these expressions that are the output of the logistic times 1 minus the output of a logistic. So a lot of these error terms as you doing the chain rule and propagating, they tend to go to 0. So as phenomenon that's called its vanishing gradients. So te gradients give you a very little signal, so the optimization is extremely slow, instead by doing this geedy layer wise training you can get algorithms that they quickly get a good solution and then right at the end when you're done, you can still run back propagation on the full network and just fine-tune the weights.

## Google autoencoder

![google autoencoder](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/google%20autoencoder.PNG)

Google network made the news has had billions of parameters, it essentially was following this idea, so each auto eoncoder started with an image of size 200 pixels, color images are RGB channels, so you will have 3 channels, the first they only allow each group of 8 neurons to be connected to a group of about 18 neuros, that's to decrease the size of the image and that first stage of weights is what they call W and then they will introduce a further compression using weight H and then they do operation called local contrast normalization (LCN) which is essentially standardization. The overall training objective for this neural network is, once again this is just reconstruction error encoding ($W_1^Tx^{(i)}$)and decoding ($W_2W_1^Tx^{(i)}$), they allow $W_1$ $W_2$ to be different,  but they only did this so that they could run it on their 16,000 course efficiently. the L1 regularizer they use is this the su of L2 norms, $H_j(W_1^Tx^{(i)})^2$ is of the essentially L2 norm of the inputs times x squared, so the input goes through 2 operations of W when we do encoding and decoding, and then we multiply it times the matrix of weights H in order to get any further compression. 

### Pooling

![Pooling](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/Pooling.PNG)

Pooling has the objective of trying images are huge and so in order to be able to deal you know if you take a image so its just a thousand by a thousand , and you assign one parameter to each pixel you already have 1 million parameters, so you're very quickly getting to very tough computational problems. What pooling tries to do is just tries to pick an average of a group of nerons in order to compress the image, so we trying to just average or pick the largest level of firting of the largest unit, in order to go from a window of several units to a single unit. 

### Pooling & LCN

![Pooling & LCN](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/Pooling%20&%20LCN.PNG)

The form was used in Google and in fact these operations by the way, they actually were invented in the 70s, none of this was new scientific work for Google, what google was implement this with billions of parameters and 16,000 courses, this sort of thing that only companies with lots of computers and money are capable of doing. The idea of pooling is essentially at the next layer is of neurons as you go from 2nd layer to 3rd layer,  you essentially take a window of pixels and you just average their squarred values and take the square root of that, so it's again sum of L1 norm of L2 values, the next operation the Google guys apply in order to go from 3rd layer to 4th (last) layer is local contrast normalization which is something that we've already learned to to which is subtract the mean of data and divide by the standard deviation of the data and that's essentially make sure that all the data is comparable.

### L2 Pooling

![L2 Pooling](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/L2%20pooling.PNG)

In more details this is what pooling does, if neurons are very similar, if you just put up an image slightly you wouldn't like neurons at deeper if I'm seeing of a hand and I move it slightly at the higher level in the layers I would still want my hand neuron to be firing, so I don't want little perturbations to affect it. And the idea of pooling is to get rid of these perturbations because if I move it I want nearby neurons to still fire. So if I have a group of nearby neurons  and suppose this neuron is most similar to this guy- the first neuron so that and it's anit-correlated, so it fires. 

![L2 pooling activation](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/L2%20pooling%20activation.PNG)

If I have another nearby neuron that fires I still want my output to remian at 1. So as long as it's my activation is in this same group, the output neuron will still be activde. This is essential what is does. So small distortions do not affect the output.

**L2 Pooling helps learning representations more robust to local distortions!**

### Local contrast Normalization (LCN)

![LCN](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/LCN.PNG)

Contract normalization is to do with scale, so I mentioned that when we did linear regression we made all the inputs have the same scale and tha's because we are applyng the same $\lambda$ value of regularizer to allthe inputs if you have an input that is if you're trying to classify predict the prices of cars and one input is the number of doors and one input is number of miles per gallon then one input has a very will tend to be a large number where is another just two or four, so if you do regularizatino, you'll get rid of one variable much more easily than the other and if you change the units to instead of going from gallons or from kilometers to miles then you would have to re-learn the network, what contrast normalization does it makes by subtracting the mean and divide by standard variable, the standard varies makes all the input be between sort of being the range with high probability between  zero and one, so that they are comparable. And that's idea.

![LCN by constant](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/LCN%20by%20constant.PNG)

So if you are doing the same, the neuron would fire the same if you just scale the input by a constant. So in other words, if the image gets brighter, if I look at my hand and I switch on the light or switch it off I will fit my hand neuron will still be firing. I would still be able to recognize a hand.

### Unsupervised Learning With 1B Parameters

![unsupervised learning](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/Unsupervised%20learning%20-%20Google%20Paper.PNG)

The network consisted of three layers and with three stages which is the sort of subsampling, then basically have the contrast and pooling and in contrast normalization and then they trained it to reconstruct in the news sparse regularizes.

![Unsupervised learning Layers](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/Unsupervised%20learning%20-%20Google%20Paper%20-%20Layers.PNG)

It's basically each unit consists of this filtering or compression of L2 pooling and local contrast normalization and you just got just one stage and then you have three of these stages. In other words you have three layers.

# ML 31 - Decision Trees

## Motivation example 1: object detection

The algorithm has classify the box.

## Motivation example 2: Kinect

The connect is the sensor, it's a camera that senses depth. from the depth image it learns to predict whether a pixel is an arm or a leg. 

## Image Classification example

![Image classification example](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/Image%20classification%20example.PNG)

Trees are nice because they're very easy to interpret. Random forest willdo essetially what Lasso did for us.

## Classification tree

![classification tree](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/classification%20tree.PNG)

In this case, there are four classes, you have input vector v, then you have class C, four classes represent blue, green, yellow, and red. The classifiers to just have to be $\epsilon$ better than random. If you have many slightly better than random decisions, your final decision will be a very good one.  When you go down a branch , that when you get to the leaf most of the points in one leaf will be of one class, so each leaf unfortunately will not be exactly. Just like the random forest just give your the probabilistic classification. So when you have a new input, you go through the leaves and when it reaches a leaf, what your output is those four bars. 

## Dataset

![Dataset](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/Dataset.PNG)

You have see 12 examples and what the example shows customers going to a restaurant and different things make the decide whether to wait in line or to leave. Each attribute will become a node in the tree. But we don't know which of those nodes are more useful.

### How do we construct the tree

![How do we construct the tree](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/How%20do%20we%20construct%20the%20tree.PNG)

How to pick attribute (nodes) ? Do example start from Patrons as top node or Type? Ask Patrons is better for asking because that quickly splits the data, green to one side, red to another side. 
All we need is a measuring, it measured the left hand side of the node is better than the right hand side. It is by looking at the entroy of the point, the class on the left has higher entroy. And further we check how do we go in any branch how much would have entropy have reduced, or would have gained more.  Would I have gain more information? So entory is a reminder for a binary variable wwe went over this, if the total number of data is p+n, the P log is just $\theta log\theta$ plus $(1-\theta)log(1-\theta)$

## How to pick nodes?

![How to pick nodes](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/How%20to%20pick%20nodes.PNG)

Let assume the node has K values, the problem is before we do the decision we don't know what the children will look like, we can do is computer the expectation what we think it should look like. The child of a node A has the $i_s$ child which has $p_i$ positives and $n_i$ negatives. The normalization just make sure we have the probabilities that they add up to one. $\frac{p_i+n_i}{p+n}$ is just the probability that you'll pick the $i_s$ leaves. And then we we do is we look at the entorpy of the root minues the expected entropy of the children and so this is called the infromation gain or the entropy reduction. so the nodes that reduce the entropy the most are the ones that we should pick or equivalently the node that maximizes information the most is the one that you want to choose.

### Example

![Decision Tree example](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/Decision%20Tree%20example.PNG)

$\frac{2}{12}H(0,1)$ is means, there are two of red over 12 total training set, there is no green in None class, so it is 0, two reds over total 2 set in None class,so it is 1. $\frac{6}{12}H(\frac{2}{6},\frac{4}{6})$ is means, there are 6 sets over 12 total training set in Full class, there is two greens over total 6 sets in Full class, so it is $\frac{2}{6}$, four reds over total 6 set in Full class, so it is $\frac{4}{6}$.

# ML 32 - Random Forests

## Use information gain to decide splits

![User information gain to decide splits](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/User%20information%20gain%20to%20decide%20splits.PNG)

How do we deal with continuous data?  If we had the option of a vertical line and a horizontal line in the middle. Compare with the horizontal line, the vertical line did better job because the vertical line put all the blue and greens in one side, red and yellow on the other side. The vertical boundary seems to split twll us apart two classes quit well. 

We computer the entory to decide which line is well do the split. If wwe can know the entropy, then we compute the information gain, and that decides whether we use horizontal or vertical. 

## Adanced: Gaussian information gain to decide splits

![Advanced Gaussian information gain ](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/Advanced%20Gaussian%20information%20gain%20to%20decide%20splits.PNG)

It is possible to do this for Gaussian, if you have gaussians, currently we're using splits as the decisions but it's possible to just use actually full Gaussian distributions and you can also compute information gained for Gaussian distributions. 

![Multiple lines to decide split](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/Multiple%20lines%20to%20decide%20splits.PNG)

We know that your decisions,  attributes are horizontal or vertical line, but obviously you would think there could be more than one horizontal or vertical line. Essentially what we're doing is we're now going to have a huge family of such line and the parameter $\theta$ that determines exactly where that line is. The $\theta$ is a threshold, and the decision node is a function, $h(v,\theta_j) \in {0,1}$. In this case, it is a line which has different heights, different features, and based on whether you're above that line or below, youget 1 or 0.  In order to train my node, I will try all my possible value of $\theta$, eg, I start with five value and then I check which is the best one in terms of splitting the data, which means the best one in terms of maximizing the information gain, this is the way not just for one line, but for many lines. A typical strategy for example used in the Kinect sensor is just pick a bunch of random $\theta$ and then you have a big dictionary, you just go over each of them and you pick the best. 

![Alternative node decisions](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/Alternative%20node%20decisions.PNG)

So far we see the horizontal and vertical line, the other thing is you might choose to use a line with an angle, so you might want to use a logistic classify as the decision in each node, so each node now is basically classify. If you just building a single tree it would be important to use more and ore fancy ones, but as  we will see by using a technique of using many trees essentially and averageing them, it will be enough to use simple trees.

The central limit theorem tells us that if you averaged many numbers, we end up with a Gaussian distribution, and it says the more points you add, the less variance of that Gaussian. so in other words, just by using a survey simple argument, if you take many simple decision, and you average them, you will knock off the variance, and you willend up with a very good technique.

## Random Forests for classification or regression 

How do we average? If the blue line, red line , green line is the same, then averaging them would not have helped. But the blue line or black line are in the different position, so that I can actually get something meaningful out of averaging the lines, if all my lines were like the blue line, then averaging would not allow me to get something that would be like the dashed line. Now how do we make the classify as different from each other. 

![Random Forests for Classification](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/Random%20Forests%20for%20classification.PNG)

Here is the strategy, first of all, let each classifier have access to a different subset of the data. So the first step in the algorithmm is called in statistics a bootstrap which is you draw some sample from training data with replacement, you pick a new data set. Each tree will be looked at a different set of the data,  Each tree looks at the portion of the data and each node in the tree only looks at the the portion of the features.

Each tree learns from a subset of the data that's called bagging if you use the full dataset and you can actually get confidence estimates and that's called bootstrapping. The theory behind is, each tree learned from a little bit of data, moreover for each tree we don't consider all the possible features, we only consider subset of the features, so we randomized the data, so each tree on this is a little bit of the data and moreover each tree only has a subset of the features. 

 ##  Randomization
 ![Randomization](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/Randomization.PNG)
 
We pick a subset of the parameters only at each node and for the sebset of the parameters,  we find the best $\theta$ by maximizing information gain. 

## Building a forest (ensable)

![Building a forest](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/build%20a%20forest.PNG)

This is answering the question, how do we combine the trees. Suppose we have three trees, for each tree you computer, eg you have new test data, then follow the decisions in the tree until you each leaf, each leaf has associated with its a histogram, then we do that (the same new test data) for all the trees (three trees), so we have three histograms, if you want a combined histogram, you just average those.  So we average three green cells and the blue cells and the red celles, yellow cells. You just average all the classes. Little t is the average tree, big T is the all trees.

The first and second tree is quite confident it will be green, the last tree is not very sure, it could be red or blue but when you sum  them all up, the point will be prevalently green. So if one tree is not very good, it does not matter. There are many techniques to do the average, can also base on geometry, or on the profits called stacking. And if you only have few data, eg patient with cancer, you want to know are my estimates good enough based on among those data, and bootstrapping is the technique that allows you to access that. 

# ML 33 - Random forests, face detection and Kinect

## Effect of forest size

![Effect of forest size](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/Effect%20of%20forest%20size.PNG)

Here is a dataset with two classes, so the input is two-dimensional for $x_1$ and $x_2$ say height and weight of a person and there's two trees there on top, if only use one tree (T=1) the tree will display like bottom-left side. If use 8 trees (T=8), I start getting shaded confort zone, I can get estimates of the probability of class yellow and class red, and as you go for more trees that you can see you start getting a really nice estimate of the probability of points being either red or yellow, so for any point there, the color intensity tell you how yellow dense would tell you the probability of the point being yellow or red. So single tree is just a binary decision but when  you start averaging.

### Effect of more classes and noise

![Effect of more classes and noise](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/Effect%20of%20more%20classes%20and%20noise.PNG)

If you have more classes, random forests work on 2 or 4 classes pretty well, if there's oise it does work as well.

### Effect of tree depth (D)

![Effect of tree depth](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/Effect%20of%20tree%20depth.PNG)

How big/deep should the trees be? This is quite big question in researching, here som result.  If your trees are too small (D=3), you might have some underfitting.  If you are using the trees too deep, so the tree start becoming complex, the same with neural networks, if your model has too many parameters, it has the ability of overfitting. So how we find? Using Cross validation.

## Effect of bagging

![Effect of bagging](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/Effect%20of%20bagging.PNG)

The previous results all were based on each classifier looks at of the data, but here I gonna compare two things, on top each tree looks at all the data, so the only randomization is on the other features. The bottom plot , the randomization is also over data, so each tree only looks at 50% of the data, so each tree is looking at less data. What I observed? The points that are closest to the boundary, the points of both classes that are closest to each other, they seem to be the ones that matter the most and what we actually find if you just do random forest, these distances from the decision boundary that running for spines where the points are is the same, this is called the max margin property, the key idea that is used to build things called support vector machines. If you have many points and you want to do linear classify, there are many lines you can draw to separate these points,  some lines will be more robust than others, what support vector machine or max margin tells you, if you look at the points in the boundary then you draw the line that tries to maximize the gap between the points in the boundary.

If you bagging with following thing, the decision boundary moves, the moves to be more in the middle of the two big clumps, its not the max margin decision, on


### Object detection

![object detection](https://github.com/yanzhang422/Machine-Learning/blob/main/UCB-ML-Undergraduate/Theory%20Course/IMG/object%20detection.PNG)

You are going to  take a very simple window, these windows will be of size 24 by 24 pixels, you sum the intenstiy of the pixel in the middle and minus the intensity of the pixlel that fall on the left and on the right, that's your feature, and because you have many locations images, there is index $x_i$ tell you which location you're placing in the patch. Then you pick a bunch of thresholds $\theta$, and you just check the intensity greater than the tresholds, this is the weak learner, a very basic decision. You put the window, you look at the difference between white and black, and then you check is greater than treshold, if it is , left is not right. Then you average all your weak learners, $\alpha_th_t(x)$, normally using the trees, look e a bunch of random featuresfor the one node what is give you the better split, the plot "relevant feature" is good one which give you more information gain, the  "Irrelevant feature" is bad one. It's a simple decision tree, 






































































