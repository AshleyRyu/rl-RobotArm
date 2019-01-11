HRL

every timestep t 마다 





H->L : fixed parameterized reward function r 을 써서 intrinsic reward r_t = r(s_t,g_t,a_t,s_t+1)

lower level policy는 off-policy training을 위해 experience( s, g, a, r, s_t+1, h(s,g,s_t+1))을 저장한다.(이 때 h는 fixed goal transition function, g_t+1 = h(s,g,s_t+1)이다.)

higher level policy는 환경 리워드(R_t)를 저장하고, off policy training을 위해 매 c time step마다 higher-level transition 을 저장한다.



higher level policy : g_t를 생성, 그 골은 state 과ㄴ찰결과에서 "desired relative changes"를 나타낸다.

즉, step t에서 high level policy는 goal g_t를 생성한다. 이것의 하위레벨 에이전트가 s_t + g_t에 가까운 관찰 s_t+c 를 산출하려는 동작을 취하려는 목표를 나타내는  g_t 목표를 생성한다.



즉, t 단계에서 높은 수준의 정책은 낮은 수준의 에이전트가 st + gt에 가까운 관찰 st + c를 산출하는 동작을 취하려는 목표를 나타내는 gt 목표를 생성합니다



+ goal transition model h

``` h(st, gt, st+1) = st + gt − st+1.```

intrinsic reward : sparse reward인 상황에서, 보상을 얻기위한 트릭. 최종 reward 얻기 전 짜잘한 reward얻기