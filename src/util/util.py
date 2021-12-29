import robosuite as suite
from robosuite import load_controller_config
from robosuite.devices import Keyboard
from robosuite.wrappers import GymWrapper, VisualizationWrapper

from constants import REACH2D_ACT_MAGNITUDE
from envs import CustomWrapper
from models import MLP, Ensemble, LinearModel


def get_model_type_and_kwargs(args, obs_dim, act_dim):
    if args.arch == "LinearModel":
        model_type = LinearModel
        if args.environment == "Reach2D":
            model_kwargs = dict(obs_dim=obs_dim, act_dim=act_dim, scale=REACH2D_ACT_MAGNITUDE, normalize=True)
        else:
            model_kwargs = dict(obs_dim=obs_dim, act_dim=act_dim)
    elif args.arch == "MLP":
        model_type = MLP
        model_kwargs = dict(obs_dim=obs_dim, act_dim=act_dim, hidden_size=args.hidden_size)
    else:
        raise NotImplementedError(f"The architecture {args.arch} has not been implemented yet!")

    return model_type, model_kwargs


def init_model(model_type, model_kwargs, device, num_models):
    if num_models > 1:
        model_kwargs = dict(model_kwargs=model_kwargs, device=device, num_models=num_models, model_type=model_type)
        model = Ensemble(**model_kwargs)
    elif num_models == 1:
        model = model_type(**model_kwargs)
    else:
        raise ValueError(f"Got {num_models} for args.num_models, but value must be an integer >= 1!")

    return model


def setup_robosuite(args, max_traj_len):
    render = not args.no_render
    controller_config = load_controller_config(default_controller="OSC_POSE")
    config = {
        "env_name": args.environment,
        "robots": "UR5e",
        "controller_configs": controller_config,
    }

    if args.environment == "NutAssembly":
        env = suite.make(
            **config,
            has_renderer=render,
            has_offscreen_renderer=False,
            render_camera="agentview",
            single_object_mode=2,  # env has 1 nut instead of 2
            nut_type="round",
            ignore_done=True,
            use_camera_obs=False,
            reward_shaping=True,
            control_freq=20,
            hard_reset=True,
            use_object_obs=True,
        )
    elif args.environment == "PickPlace":
        env = suite.make(
            **config,
            has_renderer=render,
            has_offscreen_renderer=False,
            render_camera="agentview",
            single_object_mode=2,
            object_type="cereal",
            ignore_done=True,
            use_camera_obs=False,
            reward_shaping=True,
            control_freq=20,
            hard_reset=True,
            use_object_obs=True,
        )
    else:
        env = suite.make(
            **config,
            has_renderer=render,
            has_offscreen_renderer=False,
            render_camera="agentview",
            ignore_done=True,
            use_camera_obs=False,
            reward_shaping=True,
            control_freq=20,
            hard_reset=True,
            use_object_obs=True,
        )

    env = GymWrapper(env)
    env = VisualizationWrapper(env, indicator_configs=None)
    env = CustomWrapper(env, render=render)

    input_device = Keyboard(pos_sensitivity=0.5, rot_sensitivity=3.0)

    if render:
        env.viewer.add_keypress_callback("any", input_device.on_press)
        env.viewer.add_keyup_callback("any", input_device.on_release)
        env.viewer.add_keyrepeat_callback("any", input_device.on_press)

    arm_ = "right"
    config_ = "single-arm-opposed"
    active_robot = env.robots[arm_ == "left"]
    robosuite_cfg = {
        "max_ep_length": max_traj_len,
        "input_device": input_device,
        "arm": arm_,
        "env_config": config_,
        "active_robot": active_robot,
    }

    return env, robosuite_cfg
