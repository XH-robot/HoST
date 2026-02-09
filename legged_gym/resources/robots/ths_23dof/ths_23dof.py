

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from legged_lab.assets import ISAAC_ASSET_DIR


# 定义一个名为 THS_23DOF 的关节体（Articulation）配置对象
THS_23DOF_CFG = ArticulationCfg(                                         
    spawn=sim_utils.UsdFileCfg(                                     # 配置生成方式：从 USD 文件加载模型
        usd_path=f"{ISAAC_ASSET_DIR}/ths_23dof/usd/ths_23dof.usd",  # USD 模型文件的完整路径，使用 ISAAC_ASSET_DIR 环境变量拼接
        activate_contact_sensors=True,                              # 激活接触传感器，用于检测碰撞接触信息
        # 刚体物理属性配置
        rigid_props=sim_utils.RigidBodyPropertiesCfg(               
            disable_gravity=False,                                  # 不禁用重力，物体受重力影响
            retain_accelerations=False,                             # 不保留加速度缓存，节省内存
            linear_damping=0.0,                                     # 线性阻尼为 0，运动不受线性阻力衰减
            angular_damping=0.0,                                    # 角阻尼为 0，旋转不受阻力衰减
            max_linear_velocity=1000.0,                             # 最大线速度限制为 1000.0 m/s
            max_angular_velocity=1000.0,                            # 最大角速度限制为 1000.0 rad/s
            max_depenetration_velocity=1.0,                         # 最大脱离穿透速度设为 1.0，防止碰撞解决过快
        ),
        # 关节体根节点属性配置
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(  
            enabled_self_collisions= True, #False,                          # 禁用自碰撞检测，关节体各部分不会相互碰撞
            solver_position_iteration_count=8,                      # 位置求解器迭代次数为 8，提高位置精度
            solver_velocity_iteration_count=4,                      # 速度求解器迭代次数为 4，提高速度精度 
        ),
    ),
 
    # 定义关节体的初始状态配置
    init_state=ArticulationCfg.InitialStateCfg( 
        pos=(0.0, 0.0, 0.752),                                    # 初始世界坐标位置 (x, y, z)
        # 设置各关节的初始角度（位置），单位为弧度
        joint_pos={      
            "left_hip_pitch_joint": 0.3,                           # 左髋俯仰关节初始角度                        
            "left_hip_roll_joint": 0.0,                            # 左髋横滚关节初始角度
            "left_hip_yaw_joint": 0.0,                             # 左髋偏航关节初始角度
            "left_knee_joint": -0.6,                               # 左膝俯仰关节初始角度 
            "left_ankle_pitch_joint": -0.3,                        # 左脚踝俯仰关节初始角度 
            "left_ankle_roll_joint": 0.0,                          # 左脚踝横滚关节初始角度 
            "right_hip_pitch_joint": -0.3,                         # 右髋俯仰关节初始角度 
            "right_hip_roll_joint": 0.0,                           # 右髋横滚关节初始角度
            "right_hip_yaw_joint": 0.0,                            # 右髋偏航关节初始角度 
            "right_knee_joint": 0.6,                               # 右膝俯仰关节初始角度 
            "right_ankle_pitch_joint": 0.3,                        # 右脚踝俯仰关节初始角度 
            "right_ankle_roll_joint": 0.0,                         # 右脚踝横滚关节初始角度 
            "torso_joint": 0.0,                                     # 躯干(腰部)
            "left_shoulder_pitch_joint": 0.0,                      # 左肩俯仰关节初始角度 
            "left_shoulder_roll_joint": 1.3,                       # 左肩横滚关节初始角度 
            "left_shoulder_yaw_joint": -0.6,                       # 左肩偏航关节初始角度 
            "left_elbow_joint": 0.3,                               # 左肘俯仰关节初始角度 
            "left_wrist_roll_joint": 0.0,                           # 左手腕横滚
            "right_shoulder_pitch_joint": 0.0,                      # 右肩俯仰关节初始角度 
            "right_shoulder_roll_joint": -1.3,                      # 右肩横滚关节初始角度 
            "right_shoulder_yaw_joint": 0.6,                        # 右肩偏航关节初始角度 
            "right_elbow_joint": -0.3,                              # 右肘俯仰关节初始角度
            "right_wrist_roll_joint": 0.0,
        },
        joint_vel={".*": 0.0},                                      # 使用正则表达式 ".*" 匹配所有关节，初始速度设为 0.0（静止启动）
    ),


    soft_joint_pos_limit_factor=0.9,                                      # 关节位置软限制因子设为 0.9，限制范围略小于物理极限以保护仿真稳定
    # 定义驱动器配置字典，用于设置不同关节组的驱动参数
    actuators={                                                         
        "legs": ImplicitActuatorCfg(                            # 为腿部关节组创建隐式驱动器配置
            # 使用正则表达式匹配腿部相关关节名称
            joint_names_expr=[    
                ".*_hip_pitch_joint",                                     # 匹配左右髋俯仰关节 
                ".*_hip_roll_joint",                                      # 匹配左右髋横滚关节                           
                ".*_hip_yaw_joint",                                       # 匹配左右髋偏航关节
                ".*_knee_joint",                                          # 匹配左右膝俯仰关节
            ],
            # 仿真中的力矩限制
            effort_limit_sim={              
                ".*_hip_pitch_joint":  50,                                # 髋俯仰关节                               
                ".*_hip_roll_joint":   50,                                # 髋横滚关节
                ".*_hip_yaw_joint":    20,                                # 髋偏航关节
                ".*_knee_joint":       50,                                # 膝俯仰关节
            },
            # 仿真中的最大速度限制
            velocity_limit_sim={         
                ".*_hip_pitch_joint":  10,                              # 髋俯仰关节                               
                ".*_hip_roll_joint":   10,                              # 髋横滚关节
                ".*_hip_yaw_joint":    10,                              # 髋偏航关节
                ".*_knee_joint":       10,                              # 膝俯仰关节
            },
            
            # 关节刚度系数
            stiffness={                                                 
                ".*_hip_pitch_joint":  100,                             # 俯仰关节                                    
                ".*_hip_roll_joint":   100,                             # 髋横滚关节
                ".*_hip_yaw_joint":    50,                              # 髋偏航关节
                ".*_knee_joint":       100,                             # 关节

            },
            # 关节阻尼系数
            damping={                                                    
                ".*_hip_pitch_joint":  5,                               # 髋俯仰关节                                    
                ".*_hip_roll_joint":   5,                               # 髋横滚关节  
                ".*_hip_yaw_joint":    3,                             # 髋偏航关节
                ".*_knee_joint":       5,                               # 膝俯仰关节
            },

        ),
        "feet": ImplicitActuatorCfg(                           # 为脚踝部关节组创建隐式驱动器配置
            # 使用正则表达式匹配脚部相关关节名称
            joint_names_expr=[                                              
                ".*_ankle_pitch_joint",                                     # 匹配左右脚踝俯仰关节
                ".*_ankle_roll_joint",                                      # 匹配左右脚踝横滚关节
            ],
            # 仿真中的力矩 限制
            effort_limit_sim={                                              
                ".*_ankle_pitch_joint":  15,                                # 脚踝俯仰关节
                ".*_ankle_roll_joint":   12,                                # 脚踝横滚关节
            },
            # 仿真中的速度限制
            velocity_limit_sim={                                            
                ".*_ankle_pitch_joint":  10,                                # 脚踝俯仰关节
                ".*_ankle_roll_joint":   10,                                # 脚踝横滚关节
            },
            # 关节刚度系数
            stiffness={                                                     
                ".*_ankle_pitch_joint":  20,                                # 脚踝俯仰关节
                ".*_ankle_roll_joint":   20,                                # 脚踝横滚关节
            },
            # 关节阻尼系数
            damping={                                                      
                ".*_ankle_pitch_joint":  1.5,                                 # 脚踝俯仰关节
                ".*_ankle_roll_joint":   1.5,                                # 脚踝横滚关节
            },
        ),
        "torso":ImplicitActuatorCfg(                                 # 为躯干创建隐式驱动器配置   
            # 使用正则表达式匹配手臂相关关节名称                             
            joint_names_expr=[                                             
                "torso_joint",                                               # 匹配腰部偏航关节
            ],
            # 仿真中的力矩限制
            effort_limit_sim={                                            
                "torso_joint":           20,                                 # 躯干腰部关节
            },
            # 仿真中的速度限制
            velocity_limit_sim={                                           
                "torso_joint":           10,                                 # 躯干腰部关节
            },
            # 关节刚度系数
            stiffness={                                                   
                "torso_joint":           20,                                 # 躯干腰部关节
            },
            # 关节阻尼系数
            damping={                                                     
                "torso_joint":           1.5,                                # 躯干腰部关节
            },
        ),
        "arms": ImplicitActuatorCfg(                               # 为手臂关节组创建隐式驱动器配置   
            # 使用正则表达式匹配手臂相关关节名称                             
            joint_names_expr=[                                             
                ".*_shoulder_pitch_joint",                                  # 匹配左右肩俯仰关节
                ".*_shoulder_roll_joint",                                   # 匹配左右肩横滚关节
                ".*_shoulder_yaw_joint",                                    # 匹配左右肩偏航关节
                ".*_elbow_joint",                                           # 匹配左右肘俯仰关节
                ".*_wrist_roll_joint"                                       # 匹配左右手腕横滚
            ],
            # 仿真中的力矩限制
            effort_limit_sim={                                            
                ".*_shoulder_pitch_joint":  15,                             # 肩俯仰关节
                ".*_shoulder_roll_joint":   12,                             # 肩横滚关节
                ".*_shoulder_yaw_joint":    12,                             # 肩偏航关节
                ".*_elbow_joint":           12,                             # 肘俯仰关节
                ".*_wrist_roll_joint":      12,
            },
            # 仿真中的速度限制
            velocity_limit_sim={                                           
                ".*_shoulder_pitch_joint":  10,                             # 肩俯仰关节
                ".*_shoulder_roll_joint":   10,                             # 肩横滚关节
                ".*_shoulder_yaw_joint":    10,                             # 肩偏航关节
                ".*_elbow_joint":           10,                             # 肘俯仰关节
                ".*_wrist_roll_joint":      10,
            },
            # 关节刚度系数
            stiffness={                                                   
                ".*_shoulder_pitch_joint":  15,                             # 肩俯仰关节
                ".*_shoulder_roll_joint":   15,                             # 肩横滚关节
                ".*_shoulder_yaw_joint":    10,                             # 肩偏航关节
                ".*_elbow_joint":           10,                             # 肘俯仰关节
                ".*_wrist_roll_joint":      10,
            },
            # 关节阻尼系数
            damping={                                                     
                ".*_shoulder_pitch_joint":  0.8,                             # 肩俯仰关节
                ".*_shoulder_roll_joint":   0.8,                             # 肩横滚关节
                ".*_shoulder_yaw_joint":    0.5,                             # 肩偏航关节
                ".*_elbow_joint":           0.5,                             # 肘俯仰关节
                ".*_wrist_roll_joint":      0.5,
            },
        ),

    },
)
