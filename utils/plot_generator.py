import matplotlib.pyplot as plt
import os


APP_ROOT = os.path.dirname(os.path.abspath(__file__))
results_folder = os.path.join(APP_ROOT, '../results')
plot_folder = os.path.join(results_folder, 'plot_images')
log_folder = os.path.join(results_folder, 'exec_logs')

def build_plot(file_name, first_metric_list, second_metric_list, metric_name):

    plt.rcParams["figure.autolayout"] = True

    file_name = file_name.replace('.results', '')
    plot_name = os.path.join(plot_folder, f'{metric_name}_{file_name}')

    # plot creation
    num_epochs = len(first_metric_list) + 1
    epochs = range(1, num_epochs)

    fig, ax = plt.subplots()

    ax.plot(epochs, first_metric_list, color='#BF2A15', linestyle=':', label=f'Training {metric_name}')
    ax.plot(epochs, second_metric_list, 'o-', label=f'Validation {metric_name}')
    ax.set_xlabel('Epochs')
    ax.set_ylabel(f'{metric_name}')
    ax.set_xticks(range(1, len(first_metric_list) + 1))
    ax.legend()

    # set plot colors

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_facecolor('#e5e5e5')
    ax.tick_params(color='#797979', labelcolor='#797979')
    ax.patch.set_edgecolor('black')
    ax.grid(True, color='white')

    if num_epochs >= 20:

        plt.rcParams["figure.figsize"] = [(num_epochs/3.22), 5.50]

    else:

        plt.rcParams["figure.figsize"] = [8.50, 5.50]

    plt.draw()

    fig.savefig(plot_name, dpi=180)

    print(f"{metric_name}'s plot created and stored at following path: {plot_folder}.")

def generate_plot(choice, result_file):

    file_name = result_file

    # check if file contains .results extension. If not add it.

    if not result_file.endswith('.results'):
        result_file = os.path.join(log_folder, result_file + '.results')
    else:
        result_file = os.path.join(log_folder, result_file)

    # check if file is valid or not

    if not os.path.isfile(result_file):

        print(f'Error! {result_file} does not exists. Check the name of the file!')

    # check if the plot_folder already exits, if not create it.

    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)

    # defining lists

    accuracy = list()
    accuracy_validation = list()
    loss = list()
    loss_validation = list()

    # get values in file

    with open(result_file, "r") as txt_file:
        for line in txt_file:
            if 'train_acc' in line:
                acc_train = line
            elif 'val_acc' in line:
                acc_val = line
            elif 'train_loss' in line:
                loss_train = line
            elif 'val_loss' in line:
                loss_val = line

    # accuracy

    for acc in acc_train.split(','):
        accuracy.append(float(acc.replace('\ttrain_acc:[', '').replace(' ', '').replace(']\n', '')))

    for acc_val in acc_val.split(','):
        accuracy_validation.append(float(acc_val.replace('\tval_acc:[', '').replace(' ', '').replace(']\n', '')))

    # loss

    for loss_t in loss_train.split(','):
        loss.append(float(loss_t.replace('\ttrain_loss:[', '').replace(' ', '').replace(']\n', '')))

    for loss_v in loss_val.split(','):
        loss_validation.append(float(loss_v.replace('\tval_loss:[', '').replace(' ', '').replace(']\n', '')))

    if choice == 'accuracy':

        build_plot(file_name, first_metric_list=accuracy, second_metric_list=accuracy_validation, metric_name='Accuracy')

    elif choice == 'loss':

        build_plot(file_name, first_metric_list=loss, second_metric_list=loss_validation,
                   metric_name='Loss')

    elif choice == 'both':

        build_plot(file_name, first_metric_list=accuracy, second_metric_list=accuracy_validation,
                   metric_name='Accuracy')
        build_plot(file_name, first_metric_list=loss, second_metric_list=loss_validation,
                   metric_name='Loss')

