from models import LinearModel, MLP
from util import ACT_MAGNITUDE

def get_model_type_and_kwargs(args, obs_dim, act_dim):
    if args.arch == 'LinearModel':
        model_type = LinearModel
        if args.environment == 'Reach2D':
            model_kwargs = dict(obs_dim=obs_dim, act_dim=act_dim, 
                                scale=ACT_MAGNITUDE, normalize=True)
        else:
            model_kwargs = dict(obs_dim=obs_dim, act_dim=act_dim)
    elif args.arch == 'MLP':
        model_type = MLP
        model_kwargs = dict(obs_dim=obs_dim, act_dim=act_dim, 
                            hidden_size=args.hidden_size)
    else:
        raise NotImplementedError(f'The architecture {args.arch} has not been implemented yet!')
    
    return model_type, model_kwargs