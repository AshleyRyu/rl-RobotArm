## Data-Efficient HRL

- Off-policy TD
- state-action-reward





### Q함수 형성

deterministic NN policy 는 상응하는 state-action Q function ```Q_theta``` 를 통해 배웁니다.

파이와 세타에 대한 그래디언트 업데이트를 통해.

샘플된 데이터를 통해 bellman error를 줄이는 방향으로 학습한다.

```
xxxxxxxxxx 	
```



### HIRO(HIerarchical Reinforcement learning with Off-policy correction)

Lower-level의 무한한 정책을 특정짓기 위해 parameterized reward function을 이용한다.

Parameterized reward function은 desired goal을 위해 observed state s_t에 상응한다.





instantiation 



### 방식

!image-20190112143433401](/Users/ryujiwon/Library/Application Support/typora-user-images/image-20190112143433401.png)

![image-20190112143525085](/Users/ryujiwon/Library/Application Support/typora-user-images/image-20190112143525085.png)



1. the higher-level policy takes in state observations from the environment and produces high-level actions (**goals**).

![image-20190112143716040](/Users/ryujiwon/Library/Application Support/typora-user-images/image-20190112143716040.png)

2. High level actions specify desired relative changes in the state observation.



![image-20190112143639937](/Users/ryujiwon/Library/Application Support/typora-user-images/image-20190112143639937.png)

3. the lower-level policy takes in goal and state observation

![image-20190112143841873](/Users/ryujiwon/Library/Application Support/typora-user-images/image-20190112143841873.png)

4. the lower-level policy produces an atomic action and directly applies it to the environment.

![image-20190112143957337](/Users/ryujiwon/Library/Application Support/typora-user-images/image-20190112143957337.png)

5. The environment produces a reward and a new state observation

![image-20190112144048069](/Users/ryujiwon/Library/Application Support/typora-user-images/image-20190112144048069.png)

6. While the environment reward will be used by the higher-level policy for training, the lower-level policy will be rewarded for satisfying the higher-level policy goals.

![image-20190112144214246](/Users/ryujiwon/Library/Application Support/typora-user-images/image-20190112144214246.png)

7. The lower-level policy will be rewarded for satisfying the higher-level policy goals via a parameterized reward function.

![image-20190112144312264](/Users/ryujiwon/Library/Application Support/typora-user-images/image-20190112144312264.png)

8. New relative state goal is derived using goal transition function, designed to maintain the same absolute state goal.

![image-20190112144413981](/Users/ryujiwon/Library/Application Support/typora-user-images/image-20190112144413981.png)

9. Lower-level policy chooses another action to apply to the environment, and a new environment reward is yielded.

![image-20190112144521238](/Users/ryujiwon/Library/Application Support/typora-user-images/image-20190112144521238.png)



10. The process continues for c steps (fixed c), at which point the higher-level policy chooses a new goal. And so on ...



#### Training

![image-20190112144709853](/Users/ryujiwon/Library/Application Support/typora-user-images/image-20190112144709853.png)

Both policies are trained in an **off-policy** manner concurrently.



The lower-level policy stores individual step transitions and is trained to maximize **parameterized reward r**.



Higher-level policy stores temporarily extended transitions and is trained to maximize environment reward **R**



We are able to handle a non-stationary lower-level policy by using an **off-policy correction**.

어떻게 ==off-policy==가 가능할까요?

과거에는 behavior of the lower-level policy

본 논문에서는 상위 레이어 policy에 의해 수집된 transition tuples(s,g,a,r,s for c step)을  state-action-reward transition으로 바꿉니다. (s,g,sigma(r),s). 그리고 그것을 리플레이 버퍼에 넣습니다. 하지만 ㅇ하위 레벨 컨트롤러에서 얻은 변화들은 액션들을 정확히 반영하지 않기 때문에(and therefore resultant states s_t+1:s_t+c) 

그래서 반드시 오래된 변화들을 현재 하위레벨 컨트롤러에 맞게 코렉션을 해야한다.

그것이 re-labeling이다. 상위 레벨 변화들(s_t, g_t, sigma(R_t:t+c-1),s_t_c)을 다른 상위레벨 엑션(~g_t)과(이 액션은 maximize the probability policy(lo)) 리라벨링을 해야한다. 이 때, 중간 골(~g_t)은 fixed goal trnsition function h를 사용하여 계산된다.

