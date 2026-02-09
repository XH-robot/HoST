import sys
from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import copy
import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger, export_policy_as_onnx

import torch
import time

import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Value
def export_policy_as_jit(actor_critic, path, example_input=None, atol=1e-6, rtol=1e-5):
    """
    导出 actor_critic 的 actor 部分为 TorchScript 模型，并验证输出一致性。

    Args:
        actor_critic: 包含 actor 属性的策略网络（如 PPO 中的 ActorCritic 类）
        path (str): 保存目录路径
        example_input (torch.Tensor, optional): 用于验证的示例输入。
            若为 None，则自动生成符合 actor 输入要求的随机张量。
        atol (float): torch.allclose 的绝对容差
        rtol (float): torch.allclose 的相对容差
    """
    os.makedirs(path, exist_ok=True)
    save_path = os.path.join(path, 'policy_1.pt')

    # 深拷贝并移至 CPU
    model = copy.deepcopy(actor_critic.actor).to('cpu').eval()  # 推荐设为 eval 模式

    # 生成示例输入（如果未提供）
    if example_input is None:
        # 假设 actor 是一个接受 (batch_size, obs_dim) 的网络
        # 你可以根据实际情况调整 shape，例如从 actor_critic 获取 obs_shape
        try:
            # 尝试从 actor 获取输入维度（常见于有 .input_shape 或类似属性的情况）
            obs_shape = getattr(model, 'input_shape', None)
            if obs_shape is None:
                # 否则回退到默认：假设输入为 (1, *)，需要你根据实际调整
                # 更健壮的做法是从训练配置中传入 obs_shape
                obs_shape = (1, 456)  # 示例：替换为你的实际观测维度
            example_input = torch.randn(1, *obs_shape).to('cpu')
        except Exception as e:
            raise ValueError(
                "无法自动生成 example_input，请显式传入 example_input 参数。"
            ) from e
    else:
        example_input = example_input.to('cpu')

    # 原始模型推理
    with torch.no_grad():
        original_output = model(example_input)

    # 转换为 TorchScript
    traced_script_module = torch.jit.script(model)

    # TorchScript 模型推理
    with torch.no_grad():
        jit_output = traced_script_module(example_input)

    # 验证输出一致性
    if not torch.allclose(original_output, jit_output, atol=atol, rtol=rtol):
        raise RuntimeError(
            "TorchScript 模型输出与原始模型不一致！\n"
            f"原始输出: {original_output}\n"
            f"JIT 输出: {jit_output}\n"
            "请检查模型是否包含动态控制流、非 Scriptable 操作或随机性。"
        )

    # 保存模型
    traced_script_module.save(save_path)
    print(f"✅ JIT 模型已成功导出并验证，保存至: {save_path}")


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 100)
    env_cfg.terrain.num_rows = 4
    env_cfg.terrain.num_cols = 4
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.control.action_scale = 0.25
    env_cfg.curriculum.pull_force = False
    env_cfg.env.test = True

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()

    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, env_cfg=env_cfg, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        export_policy_as_onnx(ppo_runner.alg.actor_critic, path, "policy1")
        print('Exported policy as jit script to: ', path)


    logger = Logger(env.dt)
    for i in range(10*int(env.max_episode_length)):

        result = env.gym.fetch_results(env.sim, True)
        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())


if __name__ == '__main__':
    EXPORT_POLICY = True
    args = get_args()
    play(args)
