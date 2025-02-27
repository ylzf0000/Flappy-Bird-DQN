import argparse
import dataclasses
from dataclasses import dataclass, field, fields
from email.policy import default
from optparse import OptionParser
from typing import Type, Any

# ACTIONS = 2  # number of valid actions
# GAMMA = 0.99  # decay rate of past observations
# # 前OBSERVE轮次，不对网络进行训练，只是收集数据，存到记忆库中
# # 第OBSERVE到OBSERVE+EXPLORE轮次中，对网络进行训练，且对epsilon进行退火，逐渐减小epsilon至FINAL_EPSILON
# # 当到达EXPLORE轮次时，epsilon达到最终值FINAL_EPSILON，不再对其进行更新
# OBSERVE = 200.
# EXPLORE = 5000000.
# FINAL_EPSILON = 0.0001
# INITIAL_EPSILON = 1.0
# REPLAY_MEMORY = 10000
# BATCH_SIZE = 128
# # 每隔FRAME_PER_ACTION轮次，就会有epsilon的概率进行探索
# FRAME_PER_ACTION = 1
# # 每隔UPDATE_TIME轮次，对target网络的参数进行更新
# UPDATE_TIME = 1000
# width = 80
# height = 80

@dataclass
class RLArguments:
    mode: str = "train" # 训练还是玩
    model_class: str = "SiglipBertMMMoCoV2" # 网络类
    load_model_checkpoint: str = None
    save_model_checkpoint: str = r"D:\Github\Flappy-Bird-DQN\model.pth"
    mixed_precision: str = None
    gamma: float = 0.99  # decay rate of past observations
    train_steps: int = 5000000 # 训练步数
    batch_size: int = 64
    replay_memory_size: int = 10000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.001
    epsilon_decay = 0.999999
    update_steps:int = 1000
    learning_rate: float = 1e-4




def get_argparser_for_model_arguments():
    return generate_argparser_from_dataclass(RLArguments)


def generate_argparser_from_dataclass(dataclass_type: Type[Any]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f'Parser for {dataclass_type.__name__}')

    for field in fields(dataclass_type):
        field_name = field.name
        field_type = field.type
        default = field.default
        default_factory = field.default_factory if field.default_factory is not dataclasses.MISSING else None

        # Determine argument type for argparse
        arg_type = field_type
        if hasattr(field_type, '__origin__') and field_type.__origin__ is list:
            arg_type = field_type.__args__[0]
            nargs = '+'
        elif field_type is list:
            arg_type = type(default_factory()[0])
            nargs = '+'
        else:
            nargs = None

        # Add argument to parser
        if default is dataclasses.MISSING and default_factory is None:
            parser.add_argument(f'--{field_name}', type=arg_type, nargs=nargs, required=True)
        else:
            parser.add_argument(f'--{field_name}', type=arg_type, nargs=nargs, default=default_factory() if default_factory else default)

    return parser


# Usage
if __name__ == '__main__':
    # print(ModelArguments().__class__.__name__)
    parser = get_argparser_for_model_arguments()
    args = parser.parse_args()
    # print(args)
    config = RLArguments(**vars(args))
    print(config)
