```python
#input 값
#return 값 -line7 from humanoid.py
return self.__get__obs(), reward, done, dict(reward_linvel=lin_vel_cost, reward_quadctrl=-quad_ctrl_cost, reward_alive=alive_bonus, reward_impact=-quad_impact_cost)
```

Input 

```python
self.sim = mujoco_py.MjSim(self.model)
self.data = self.sim.data
#mujoco_py에서 sim 가져옵니다.
```

```python
class mujoco_py.MjSim(model, data=None, nsubsteps=1, udd_callback=None)
```

[mujoco_py.MjSim API해당문서](https://openai.github.io/mujoco-py/build/html/reference.html#mujoco_py.MjSim)

MjSim represents a running simulation including its state.

Similar to Gym’s `MujocoEnv`, it internally wraps a `PyMjModel` and a `PyMjData`.

[PyMjData  해당문서](https://openai.github.io/mujoco-py/build/html/reference.html#pymjdata-time-dependent-data)

```python

self.init_qpos = self.sim.data.qpos.ravel().copy() # 다차원 배열을 1차원 배열로
self.init_qvel = self.sim.data.qvel.ravel().copy()
observation, _reward, done, _info = self.step(np.zeros(self.model.nu)) #
assert not done
self.obs_dim = observation.size

bounds = self.model.actuator_ctrlrange.copy()
low = bounds[:, 0]
high = bounds[:, 1]
self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

high = np.inf*np.ones(self.obs_dim)
low = -high
self.observation_space = spaces.Box(low, high, dtype=np.float32)

self.seed() # random위해 seed 생성

def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]
```

data에 들어있는 것은

- `act`

- `act_dot`

- `active_contacts_efc_pos`

- `actuator_force`

- `actuator_length`

- `actuator_moment`

- `actuator_velocity`

- `body_jacp`

- `body_jacr`

- `body_xmat`

- `body_xpos`

- `body_xquat`

- `body_xvelp`

- `body_xvelr`

- `cacc`

- `cam_xmat`

- `cam_xpos`

- `cdof`

- `cdof_dot`

- `cfrc_ext`

- `cfrc_int`

- `cinert`

- `contact`

- `crb`

- `ctrl`

- `cvel`

- `efc_AR`

- `efc_AR_colind`

- `efc_AR_rowadr`

- `efc_AR_rownnz`

- `efc_D`

- `efc_J`

- `efc_JT`

- `efc_JT_colind`

- `efc_JT_rowadr`

- `efc_JT_rownnz`

- `efc_J_colind`

- `efc_J_rowadr`

- `efc_J_rownnz`

- `efc_R`

- `efc_aref`

- `efc_b`

- `efc_diagApprox`

- `efc_force`

- `efc_frictionloss`

- `efc_id`

- `efc_margin`

- `efc_solimp`

- `efc_solref`

- `efc_state`

- `efc_type`

- `efc_vel`

- `energy`

- `geom_jacp`

- `geom_jacr`

- `geom_xmat`

- `geom_xpos`

- `geom_xvelp`

- `geom_xvelr`

- `light_xdir`

- `light_xpos`

- `maxuse_con`

- `maxuse_efc`

- `maxuse_stack`

- `mocap_pos`

- `mocap_quat`

- `nbuffer`

- `ncon`

- `ne`

- `nefc`

- `nf`

- `nstack`

- `pstack`

- `qLD`

- `qLDiagInv`

- `qLDiagSqrtInv`

- `qM`

- `qacc`

- `qacc_unc`

- `qacc_warmstart`

- `qfrc_actuator`

- `qfrc_applied`

- `qfrc_bias`

- `qfrc_constraint`

- `qfrc_inverse`

- `qfrc_passive`

- `qfrc_unc`

- `qpos`

- `qvel`

- `sensordata`

- `set_joint_qpos`

- `set_joint_qvel`

- `set_mocap_pos`

- `set_mocap_quat`

- `site_jacp`

- `site_jacr`

- `site_xmat`

- `site_xpos`

- `site_xvelp`

- `site_xvelr`

- `solver`

- `solver_fwdinv`

- `solver_iter`

- `solver_nnz`

- `subtree_angmom`

- `subtree_com`

- `subtree_linvel`

- `ten_length`

- `ten_moment`

- `ten_velocity`

- `ten_wrapadr`

- `ten_wrapnum`

- `time`

- `timer`

- `userdata`

- `warning`

- `wrap_obj`

- `wrap_xpos`

- `xanchor`

- `xaxis`

- `xfrc_applied`

- `ximat`

- `xipos`