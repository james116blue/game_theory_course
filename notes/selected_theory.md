![alt text](image.png)

![alt text](image-1.png)

<ins>Normal-form game (game in strategic form) </ins>
1. A finite set of $N$ players.
2. Action set for the players: $\{\mathcal{A}_1, \mathcal{A}_2, \dots \mathcal{A}_N\}$
3. Reward functions for the k player: $R_n : \mathcal{A}_1 \times \mathcal{A}_2 \dots \times \mathcal{A}_N \to \mathbb{R}$
   (can be represented as matrix, two players -> two matrices bn )

**Types of games**

![img_4.png](img_4.png)

**example**  the Prisoner’s Dilemma

![alt text](image-2.png)

*Reward is closely related to preference*
![alt text](image-3.png)

**example**
![alt text](image-4.png)

**example**
![img.png](img.png)

**example**
Rock-scisor-paper game



### Mixed strategies

$$\pi=(\pi_1,..\pi_i..,\pi_N)=(\pi_i,\pi_{-i})$$

Expected return for mixed strategies (utility) $$ J_n(\pi) = \mathbb{E}_\pi[R_n]=\sum_{(a_1,..,a_N) \in A_1 \times ... \times A_N}\pi_1(a_1)\cdot...\cdot\pi_n(a_n) \cdot R_n(a_1, ..., a_N) $$

*For two players game reward vector will be equal* $R_k((\pi_1, \pi_2)) = \pi_1R_k\pi_2$

### Solution concepts

<ins> Nash equilibrium</ins>

$$\forall n R_n(\pi_n^*, \pi_{-n}^*) \geq R(\pi_n, \pi_{-n}^*)$$

![alt text](image-5.png)

**proof**

![img_5.png](img_5.png)

![img_6.png](img_6.png)

function f

 <ins>Best responce function</ins> 

set-valued function 
![img_1.png](img_1.png)

for  mixed strategies $B_n(\pi_{-n})=\underset{\pi_n}{\arg \max} J_n(\pi_n, \pi_{-n})$

<ins>Weak domination<\ins>

<ins>$\epsilon$-Nash equlibrium <\ins>

![img_2.png](img_2.png)

### Learning solution in game

Ficticious play (store as memory opponents moves statistic)

### Sources
1. MULTI-AGENT REINFORCEMENT LE ARNING FOUNDATIONS AND MODERN APPROACHE S Stefano V. Albrecht
2. An Introduction to Game Theory by
Martin J. Osborne
3. https://nashpy.readthedocs.io/en/stable/text-book
4. An overview of Multi-agent reinforcement learning from Game theoretic persspective