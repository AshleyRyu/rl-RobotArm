## 준형이 개발하기

MDP을 통해 자신의 상태를 인식함.

![1.png](http://www.modulabs.co.kr/files/attach/images/334/136/002/9864ef6a012bcbff9249a3805b06035d.png)

![Screenshot from 2016-07-12 16:19:31.png](http://www.modulabs.co.kr/files/attach/images/334/192/003/b256481449d77879cff9109fbecb08d1.png)



그럼, Agent는 어떻게 정책을 펼칠까? 그것은 agent가 value function을 가지고 판단.

value function에는 두가지가 있음

1. state-value function

에피소드가 끝났을 때 받는 리턴값, 

![4.png](http://www.modulabs.co.kr/files/attach/images/334/136/002/2f32323a0ff14183c045cfb04744ab73.png)

이 return의 기댓값이 state-value function임. 기댓값은 기대하는 값의 평균임. 예를들어 주사위의 수 기댓값 같은. return G_t = reward*감가율의 합임.

![Screenshot from 2016-07-08 14:47:02.png](http://www.modulabs.co.kr/files/attach/images/334/136/002/4885d4877f3115bb054016dbd00e14ea.png)

즉, 어떤 상태  s의 가치이다. agent는 다음으로 갈 수 있는 state들의 가치를 보고서 높은 가치가 있는 state로 가는 것을 지향한다. (*다음상태 v(s)로만 판단하진 않음)

2. value function

1.에서는 s에 대한 정보를 다 알아야지 기댓값을 설정할 수 있고, 그 상태로 가려면 어떻게 해야하는지, 그에 따라서 행동을 결정함.  즉, 정보 다 알아야함. 모델을 몰라도 학습을 할 수 있어햐 진정한 강화학습 아니겠노? 그래서 2.가 등장함.

![5.png](http://www.modulabs.co.kr/files/attach/images/334/136/002/e7b067d294a64c295cd120d1cdf33e20.png)

어떤 state s에서 action a를 취할 경우 받을 return에 대한 기댓값으로서 어떤 행동을 했을 때 얼마나 좋을 것인가에 대한 값이다. 우리가 흔히 말하는 Q함수가 바로 이것.



### Bellman eqn.

현재 s의 value func은 다음 s의 discounted value func에 지금 어떠한 행동을 해서 받는 r을 합한 것.

agent 초기 입장에서 각 state의 value func을 알 수 없음. 경험을 통해 state의 value  function을 배워가게 되는데 , 각 state의 value function값이 true라면 다음 식이 만족하게 된다.

![Screenshot from 2016-07-12 17:35:18.png](http://www.modulabs.co.kr/files/attach/images/334/237/003/18eba72dcfeafa6e6280055a95078ffa.png)

![Screenshot from 2016-07-12 17:42:04.png](http://www.modulabs.co.kr/files/attach/images/334/237/003/276f2082eb0ce52b5479f0678bdc24e0.png)

![Screenshot from 2016-07-12 17:51:33.png](http://www.modulabs.co.kr/files/attach/images/334/237/003/acc6587c0d50511c8c21a32ce2d67d8a.png)



### 최적의 action-value func을 찾기

![Screenshot from 2016-07-12 17:58:59.png](http://www.modulabs.co.kr/files/attach/images/334/237/003/5354ff8754b2bc1a491db64374d12536.png)

강화학습의 목표를 다시 정의하자면, action-value function을 최대로 만들어주는 optimal policy를 찾는 것입니다. action-value function의 optimal한 값을 안다면 단순히 q값이 높은 action을 선택해주면 되므로 MDP가 풀렸다고 말할 수 있습니다. 그러면 어떻게 optimal한 action-value function의 값을 알 수 있을까요?



#### Dynamic Programming

- policy iteration

![Screenshot from 2016-07-12 18:09:21.png](http://www.modulabs.co.kr/files/attach/images/334/237/003/1601b1e72a52c39d2fc6447597f0ff3b.png)





### On-policy

- MC

  ![Screenshot from 2016-07-13 00:06:43.png](http://www.modulabs.co.kr/files/attach/images/334/237/003/e80db19335830c54364e777338f123fb.png)

1) 에피소드 단위로 정책 평가

2) action-value function, model-free를 위해

3) epsilon greedy

단점 : 에피소드 끝이 있어야 하고, 오프라인 학습이다.

- TD

![Screenshot from 2016-07-13 00:19:37.png](http://www.modulabs.co.kr/files/attach/images/334/237/003/90c80cf356a95548c5fac0702e528280.png)

-> 온라인 학습으로, 한 스텝마다 업데이트

1) 스텝 단위로 정책 평가

TD error: 한 스텝 샘플을 통해 estimate된 값과 현재 Q function 값의 error

alpha: 어떠한 비율로 업뎃할건지 (즉 ???)



==> 이 두 러닝은 epsilon이 0으로 수렴하는 구조여야 한다. 탐험을 보장할 수 없으므로 off-policy



### off-policy

쉽게말해, policy가 두가지이다.

1. 학습(learning)에 사용되는 policy
2. 이터레이션(실제 움직임)에 사용되는 policy

![Screenshot from 2016-07-13 00:38:14.png](http://www.modulabs.co.kr/files/attach/images/334/237/003/237a2ce24902623c635a2455b72cf209.png)

TD랑 다른건 max 임. Q값들 중에서 최댓값을 정해 learning을 하고, 실제 움직일 때는 여러가지 경험을 위해 가끔 다른 행동을 취한다.

![Screenshot from 2016-07-13 00:42:39.png](http://www.modulabs.co.kr/files/attach/images/334/237/003/ae69117d4537c6d2e960db12f754a3e4.png)

sudo코드임.