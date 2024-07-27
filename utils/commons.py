import numpy as np
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

# log class
# class Logger:
#     def __init__(self, log_name: str, log_path: str = './', print_in_terminal: bool = True):
#         if not os.path.exists(log_path):
#             os.makedirs(log_path)
#         # remove illegal characters in log_name that cannot in a path and add time stamp
#         log_name_ = log_name.replace('/', '-') + '_' + datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
#         self.log_file = open(os.path.join(log_path, log_name_ + '.log'), 'w+')
#         self.tb_writer = SummaryWriter(log_dir=os.path.join(log_path, log_name_))
#         self.print_in_terminal = print_in_terminal
#
#     def tb_scalar(self, *args):
#         """
#         :param args:
#         :return:
#         """
#         self.tb_writer.add_scalar(*args)
#
#     def __del__(self):
#         self.log_file.close()
#
#     def msg(self, info: str):
#         """
#         :param info:
#         :return:
#         """
#         time_strip = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
#         complete_info = '{time_strip:<10}: {info:<10}'.format(time_strip=time_strip, info=info)
#         self.log_file.write(complete_info + '\n')
#         if self.print_in_terminal:
#             print(complete_info)


class Logger:
    def __init__(self,  log_path: str, exp_name: str, print_in_terminal: bool = True):
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        self.exp_name = exp_name
        self.log_path = log_path
        self.print_in_terminal = print_in_terminal
        self.log_file_name = f"{self.exp_name}.log"
        self.log_file_path = os.path.join(self.log_path, self.log_file_name)
        self.tb_writer = None

    def get_tb_writer(self):
        if self.tb_writer is None:
            self.tb_writer = SummaryWriter(log_dir=os.path.join(self.log_path, self.exp_name))
        return self.tb_writer

    def tb_scalar(self, tag, scalar_value, global_step):
        """
        Logs a scalar variable to tensorboard
        :param tag: Name of the scalar
        :param scalar_value: Value of the scalar
        :param global_step: Global step value to record
        """
        writer = self.get_tb_writer()
        writer.add_scalar(tag, scalar_value, global_step)

    def tb_hparams(self, hparam_dict):
        writer = self.get_tb_writer()
        writer.add_hparams(hparam_dict, {'hparam/avg_reward':0, 'hparam/avg_step':0})

    def __del__(self):
        if self.tb_writer:
            self.tb_writer.close()

    def msg(self, info: str):
        """
        Logs a message to the log file and optionally to the terminal.
        :param info: Message to log
        """
        time_stamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        complete_info = f"{time_stamp}: {info}"
        with open(self.log_file_path, 'a') as file:
            file.write(complete_info + '\n')
        if self.print_in_terminal:
            print(complete_info)


class EnvNotExist(Exception):
    def __str__(self):
        return 'environment name or id not exist'


class PolicyNotImplement(Exception):
    def __str__(self):
        return 'Policy Has Not Been Implemented'



if __name__ == '__main__':
    logger_ = Logger(log_name='test_log.txt', log_path='./')
    # logger_('test logger...')
