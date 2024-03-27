import gym

env_name = 'CartPole-v0'
env = gym.make("CartPole-v1", render_mode='human')

env.reset()

terminated = False
total_reward = 0
while not terminated:
   env.render()
   observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
   total_reward += reward
   print(f"{observation} -> {reward}")
print(f"Total reward: {total_reward}")

# Model training
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

env = gym.make(env_name)
env = DummyVecEnv([lambda: env])
model = PPO('MlpPolicy', env, verbose=1)

model.learn(total_timesteps=20000)

# save the model
model.save('ppo model')

evaluate_policy(model, env, n_eval_episodes=10, render=True)

env.close()

# Alt implementation
for episode in range(1, 11):
    score = 0
    obs = env.reset()
    done = False
    
    while not done:
        env.render()
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        score += reward
        
    print('Episode:', episode, 'Score:', score)
env.close()