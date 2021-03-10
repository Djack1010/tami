from utils import config

# GLOBAL VAR
progr_bar_lenght = 20


def print_log(string, print_on_screen=False, print_on_file=True):
    if print_on_screen:
        print(string)
    if print_on_file:
        with open(config.main_path + 'results/exec_logs/' + config.timeExec + ".results", 'a') as logfile:
            logfile.write(string + "\n")
