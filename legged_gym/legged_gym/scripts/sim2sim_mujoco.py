
import time

import mujoco.viewer
import mujoco
import numpy as np
import torch
import yaml
import os
from pathlib import Path
from collections import deque
import matplotlib.pyplot as plt
import csv
import pickle
import argparse
import imageio
##counter不对， 没有保存更改obs_history
def quat_rotate_inverse(q, v):
    """
    Rotate a vector v from world frame to the local frame defined by quaternion q.
    
    Parameters:
        q (np.ndarray): shape (4,), quaternion in [x, y, z, w] format
        v (np.ndarray): shape (3,), vector in world coordinates
    
    Returns:
        np.ndarray: shape (3,), vector in body/local coordinates
    """
    q = np.asarray(q, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)

    q_w = q[0]          # scalar
    q_vec = q[1:4]       # (3,)

    a = v * (2.0 * q_w ** 2 - 1.0)
    b = 2.0 * q_w * np.cross(q_vec, v)
    c = 2.0 * q_vec * np.dot(q_vec, v)

    return a - b + c

def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation

def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd


    # body & hand
    self.data.qpos[:] = mujoco_dof_pos
    mujoco.mj_forward(self.model, self.data)


if __name__ == "__main__":
# get config file name from command line
    
    mujoco_default_dof_pos = np.concatenate([
            np.array([0, 0, 0]),
            # np.array([1,0,0,0]),
            np.array([1,0,-1,0]),
            np.array([
                        0.2, 0.0, 0.0, -0.4, -0.2, 0.0,  # left leg (6)
                        -0.2 , 0.0, 0.0, 0.4, 0.2, 0.0,  # right leg (6)
                        0.0, # torso (1)
                        0.0, 1.4, 0.0, 0.0, 0.0, # left arm (6)
                        0.0, -1.4, 0.0, 0.0, 0.0,# right arm (6)
                    ])
    ])
    torque_limits = np.array([
                88, 88, 88, 140, 50, 50,
                88, 88, 88, 140, 50, 50,
                88,
                25, 25, 25, 25, 25, 
                25, 25, 25, 25, 25,          
    ])
    current_path = os.path.dirname(__file__)

    config_path =os.path.join(current_path, "configs/ths_23dof.yaml")
    print("config_path: ", config_path)

    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

        policy_path = os.path.join(current_path, config["policy_name"])
        xml_path    = os.path.join(current_path, config["xml_name"])

        print("policy_path: ", policy_path)
        print("xml_path   : ", xml_path)

        simulation_duration = config["simulation_duration"]
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]

        kps = np.array(config["kps"], dtype=np.float32)
        kds = np.array(config["kds"], dtype=np.float32)

        default_angles = np.array(config["default_angles"], dtype=np.float32)
        # print("****** default_angles ******", default_angles)

        ang_vel_scale = config["ang_vel_scale"]
        dof_pos_scale = config["dof_pos_scale"]
        dof_vel_scale = config["dof_vel_scale"]
        action_scale = config["action_scale"]
        

        num_actions = config["num_actions"]
        num_obs = config["num_obs"]
        # cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)
        # cmd = np.array(config["cmd_init"], dtype=np.float32)

        # num_mass_params_tensor = config["num_mass_params_tensor"]
        # num_friction_coeffs_tensor = config["num_friction_coeffs_tensor"]
        # num_lin_vel = config["num_lin_vel"]

        policy_mode = config["policy_mode"]

    # define context variables
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)

    counter = 0
    # delay_buffer = np.zeros((5, 1, 23))

    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt         
    d.qpos[:] = mujoco_default_dof_pos
    mujoco.mj_forward(m, d)
    # load policy
    policy = torch.jit.load(policy_path)

    if policy_mode =="unitree":
        obs_buf = np.zeros(num_obs, dtype=np.float32)
    elif  policy_mode == "host":
        obs_buf = np.zeros(num_obs * 6, dtype=np.float32)
    
    record_video = False
    video_filename = "simulation_output.mp4"
    fps = int(1.0 / m.opt.timestep)  # 假设 timestep 是固定的，例如 0.01 -> fps=100
    # 注意：实际帧率可能受 control_decimation 影响，可调整为 render_fps = fps // control_decimation
    render_every = 100  # 每多少仿真步渲染一帧（可设为 control_decimation 或 1）
    video_fps =fps/ render_every 
    frame_buffer = []
    renderer = None
    if record_video:
        try:
            renderer = mujoco.Renderer(m, height=480, width=640)
        except Exception as e:
            print("Failed to create Renderer:", e)
            print("Video recording disabled.")
            record_video = False

    with mujoco.viewer.launch_passive(m, d) as viewer:
        
        # Close the viewer automatically after simulation_duration wall-seconds.
        start = time.time()
        step_count = 0
        while viewer.is_running() and time.time() - start < simulation_duration:


            step_start = time.time()
            tau = pd_control(action, np.zeros_like(kps), kps, np.zeros_like(kds), d.qvel[6:], kds)
            torques = np.clip(tau,-torque_limits,torque_limits )
            # torques *= 0
            if counter / control_decimation < 30.0:
                 torques *= 0
                # print("torques:",torques)
            d.ctrl[:] = torques

            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            mujoco.mj_step(m, d)
            scene = viewer.user_scn
            counter += 1 

            if counter % control_decimation == 0:
                # Apply control signal here.

                # create observation
                qj = d.qpos[7:]      # 关节角度
                dqj = d.qvel[6:]     # 关节速度
                quat = d.qpos[3:7]   # 四元数

                # omega = d.qvel[3:6]  # 角速度w
                ang_vel = quat_rotate_inverse(quat, d.qvel[3:6])
                qj = qj * dof_pos_scale
                dqj = dqj * dof_vel_scale
                gravity_orientation = get_gravity_orientation(quat)

                ang_vel = ang_vel * ang_vel_scale


                if policy_mode == "host":
                    current_obs = np.concatenate((
                                            ang_vel,
                                            gravity_orientation,
                                            qj,
                                            dqj,
                                            action,
                                            np.array([0.25]) 
                                        ), axis=-1)

                    if counter / control_decimation <= 30.0:
                        current_obs = current_obs * 0

                    obs_buf = np.concatenate(( obs_buf[76:76*6],current_obs), axis=-1, dtype=np.float32)

                obs_tensor = torch.from_numpy(obs_buf).unsqueeze(0)


                action = policy(obs_tensor).detach().numpy().squeeze()
                action = np.clip(action,-100,100)*0.25
                
                if counter / control_decimation <= 30.0:
                    action = np.zeros(num_actions, dtype=np.float32)

                # print("action:",action)
                # target_dof_pos = d.qpos[7:]  + action
                # delay_idx = 2
                # action_scaled = action * 1.0
                # action_scaled_expanded = np.expand_dims(np.expand_dims(action_scaled, axis=0), axis=1)  # [1,1,23]
                # delay_buffer = np.concatenate((delay_buffer[1:], action_scaled_expanded), axis=0)

                # target_dof_pos = default_angles + delay_buffer[delay_idx, 0, :]


            # 获取关节 ID
            jid2 = m.joint('lleg_joint2').id
            jid3 = m.joint('lleg_joint3').id

            # 获取 qpos/qvel 索引
            qpos2_idx = m.jnt_qposadr[jid2]
            qvel2_idx = m.jnt_dofadr[jid2]  # hinge 是 1 DOF

            qpos3_idx = m.jnt_qposadr[jid3]
            qvel3_idx = m.jnt_dofadr[jid3]

            # 获取 actuator 索引（可选）
            aid2 = m.actuator('M_hip_l_roll').id  # 应为 7
            aid3 = m.actuator('M_hip_l_yaw').id  # 应为 8
            # aid2 = m.actuator('lleg_joint2').id  # 应为 7
            # aid3 = m.actuator('lleg_joint3').id  # 应为 8

            # 每步打印
            print(f"ctrl[1]={d.ctrl[1]:.1f}, ctrl[2]={d.ctrl[2]:.1f}")
            print(f"qpos2={d.qpos[qpos2_idx]:.4f}, qvel2={d.qvel[qvel2_idx]:.6f}")
            print(f"qpos3={d.qpos[qpos3_idx]:.4f}, qvel3={d.qvel[qvel3_idx]:.6f}")
            print(f"act force: {d.actuator_force[aid2]:.2f}, {d.actuator_force[aid3]:.2f}")

            if record_video and (step_count % render_every == 0):
                # 必须先 sync physics state to renderer
                renderer.update_scene(d)  # 或指定 camera="track", "egocentric" 等
                rgb_arr = renderer.render()
                frame_buffer.append(rgb_arr)
            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

            step_count += 1
    
    # 关闭渲染器
    if renderer:
        renderer.close()

    # 保存视频
    if record_video and frame_buffer:
        print(f"Saving video with {len(frame_buffer)} frames to {video_filename}")
        imageio.mimwrite(video_filename, frame_buffer, fps=video_fps)
        print("✅ Video saved.")
