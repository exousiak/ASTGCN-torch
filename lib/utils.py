import os
import numpy as np
import torch
import torch.utils.data
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from .metrics import masked_mape_np
from scipy.sparse.linalg import eigs
from .metrics import masked_mape_np,  masked_mae,masked_mse,masked_rmse,masked_mae_test,masked_rmse_test, masked_wmape_np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from .metrics import masked_mape_np, masked_mae_test, masked_rmse_test, masked_wmape_np


def re_normalization(x, mean, std):
    x = x * std + mean
    return x


def max_min_normalization(x, _max, _min):
    x = 1. * (x - _min)/(_max - _min)
    x = x * 2. - 1.
    return x


def re_max_min_normalization(x, _max, _min):
    x = (x + 1.) / 2.
    x = 1. * x * (_max - _min) + _min
    return x


def get_adjacency_matrix(distance_df_filename, num_of_vertices, id_filename=None):
    '''
    Parameters
    ----------
    distance_df_filename: str, path of the csv file contains edges information

    num_of_vertices: int, the number of vertices

    Returns
    ----------
    A: np.ndarray, adjacency matrix

    '''
    if 'npy' in distance_df_filename:

        adj_mx = np.load(distance_df_filename)

        return adj_mx, None

    else:

        import csv

        A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                     dtype=np.float32)

        distaneA = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                            dtype=np.float32)

        if id_filename:

            with open(id_filename, 'r') as f:
                id_dict = {int(i): idx for idx, i in enumerate(f.read().strip().split('\n'))}  # 把节点id（idx）映射成从0开始的索引

            with open(distance_df_filename, 'r') as f:
                f.readline()
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    A[id_dict[i], id_dict[j]] = 1
                    distaneA[id_dict[i], id_dict[j]] = distance
            return A, distaneA

        else:

            with open(distance_df_filename, 'r') as f:
                f.readline()
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    A[i, j] = 1
                    distaneA[i, j] = distance
            return A, distaneA


def scaled_Laplacian(W):
    '''
    compute \tilde{L}

    Parameters
    ----------
    W: np.ndarray, shape is (N, N), N is the num of vertices

    Returns
    ----------
    scaled_Laplacian: np.ndarray, shape (N, N)

    '''

    assert W.shape[0] == W.shape[1]

    D = np.diag(np.sum(W, axis=1))

    L = D - W

    lambda_max = eigs(L, k=1, which='LR')[0].real

    return (2 * L) / lambda_max - np.identity(W.shape[0])


def cheb_polynomial(L_tilde, K):
    '''
    compute a list of chebyshev polynomials from T_0 to T_{K-1}

    Parameters
    ----------
    L_tilde: scaled Laplacian, np.ndarray, shape (N, N)

    K: the maximum order of chebyshev polynomials

    Returns
    ----------
    cheb_polynomials: list(np.ndarray), length: K, from T_0 to T_{K-1}

    '''

    N = L_tilde.shape[0]

    cheb_polynomials = [np.identity(N), L_tilde.copy()]

    for i in range(2, K):
        cheb_polynomials.append(2 * L_tilde * cheb_polynomials[i - 1] - cheb_polynomials[i - 2])

    return cheb_polynomials


def load_graphdata_channel1(graph_signal_matrix_filename, num_of_hours, num_of_days, num_of_weeks, DEVICE, batch_size, shuffle=True):
    '''
    这个是为PEMS的数据准备的函数
    将x,y都处理成归一化到[-1,1]之前的数据;
    每个样本同时包含所有监测点的数据，所以本函数构造的数据输入时空序列预测模型；
    该函数会把hour, day, week的时间串起来；
    注： 从文件读入的数据，x是最大最小归一化的，但是y是真实值
    这个函数转为mstgcn，astgcn设计，返回的数据x都是通过减均值除方差进行归一化的，y都是真实值
    :param graph_signal_matrix_filename: str
    :param num_of_hours: int
    :param num_of_days: int
    :param num_of_weeks: int
    :param DEVICE:
    :param batch_size: int
    :return:
    three DataLoaders, each dataloader contains:
    test_x_tensor: (B, N_nodes, in_feature, T_input)
    test_decoder_input_tensor: (B, N_nodes, T_output)
    test_target_tensor: (B, N_nodes, T_output)

    '''

    file = os.path.basename(graph_signal_matrix_filename).split('.')[0]

    dirpath = os.path.dirname(graph_signal_matrix_filename)

    filename = os.path.join(dirpath,
                            file + '_r' + str(num_of_hours) + '_d' + str(num_of_days) + '_w' + str(num_of_weeks)) +'_astcgn'

    print('load file:', filename)

    file_data = np.load(filename + '.npz')
    train_x = file_data['train_x']  # (10181, 307, 3, 12)
    # train_x = train_x[:, :, 0:1, :]
    train_target = file_data['train_target']  # (10181, 307, 12)

    val_x = file_data['val_x']
    val_x = val_x[:, :, 0:1, :]
    val_target = file_data['val_target']

    test_x = file_data['test_x']
    test_x = test_x[:, :, 0:1, :]
    test_target = file_data['test_target']

    mean = file_data['mean'][:, :, 0:1, :]  # (1, 1, 3, 1)
    std = file_data['std'][:, :, 0:1, :]  # (1, 1, 3, 1)

    # ------- train_loader -------
    train_x_tensor = torch.from_numpy(train_x).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
    train_target_tensor = torch.from_numpy(train_target).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)

    train_dataset = torch.utils.data.TensorDataset(train_x_tensor, train_target_tensor)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

    # ------- val_loader -------
    val_x_tensor = torch.from_numpy(val_x).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
    val_target_tensor = torch.from_numpy(val_target).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)

    val_dataset = torch.utils.data.TensorDataset(val_x_tensor, val_target_tensor)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # ------- test_loader -------
    test_x_tensor = torch.from_numpy(test_x).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
    test_target_tensor = torch.from_numpy(test_target).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)

    test_dataset = torch.utils.data.TensorDataset(test_x_tensor, test_target_tensor)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # print
    print('train:', train_x_tensor.size(), train_target_tensor.size())
    print('val:', val_x_tensor.size(), val_target_tensor.size())
    print('test:', test_x_tensor.size(), test_target_tensor.size())

    return train_loader, train_target_tensor, val_loader, val_target_tensor, test_loader, test_target_tensor, mean, std


def compute_val_loss_mstgcn(net, val_loader, criterion,  masked_flag,missing_value,sw, epoch, limit=None):
    '''
    for rnn, compute mean loss on validation set
    :param net: model
    :param val_loader: torch.utils.data.utils.DataLoader
    :param criterion: torch.nn.MSELoss
    :param sw: tensorboardX.SummaryWriter
    :param global_step: int, current global_step
    :param limit: int,
    :return: val_loss
    '''

    net.train(False)  # ensure dropout layers are in evaluation mode

    with torch.no_grad():

        val_loader_length = len(val_loader)  # nb of batch

        tmp = []  # 记录了所有batch的loss

        for batch_index, batch_data in enumerate(val_loader):
            encoder_inputs, labels = batch_data
            outputs = net(encoder_inputs)
            if masked_flag:
                loss = criterion(outputs, labels, missing_value)
            else:
                loss = criterion(outputs, labels)

            tmp.append(loss.item())
            if batch_index % 100 == 0:
                print('validation batch %s / %s, loss: %.2f' % (batch_index + 1, val_loader_length, loss.item()))
            if (limit is not None) and batch_index >= limit:
                break

        validation_loss = sum(tmp) / len(tmp)
        sw.add_scalar('validation_loss', validation_loss, epoch)
    return validation_loss


# def evaluate_on_test_mstgcn(net, test_loader, test_target_tensor, sw, epoch, _mean, _std):
#     '''
#     for rnn, compute MAE, RMSE, MAPE scores of the prediction for every time step on testing set.
#
#     :param net: model
#     :param test_loader: torch.utils.data.utils.DataLoader
#     :param test_target_tensor: torch.tensor (B, N_nodes, T_output, out_feature)=(B, N_nodes, T_output, 1)
#     :param sw:
#     :param epoch: int, current epoch
#     :param _mean: (1, 1, 3(features), 1)
#     :param _std: (1, 1, 3(features), 1)
#     '''
#
#     net.train(False)  # ensure dropout layers are in test mode
#
#     with torch.no_grad():
#
#         test_loader_length = len(test_loader)
#
#         test_target_tensor = test_target_tensor.cpu().numpy()
#
#         prediction = []  # 存储所有batch的output
#
#         for batch_index, batch_data in enumerate(test_loader):
#
#             encoder_inputs, labels = batch_data
#
#             outputs = net(encoder_inputs)
#
#             prediction.append(outputs.detach().cpu().numpy())
#
#             if batch_index % 100 == 0:
#                 print('predicting testing set batch %s / %s' % (batch_index + 1, test_loader_length))
#
#         prediction = np.concatenate(prediction, 0)  # (batch, T', 1)
#         prediction_length = prediction.shape[2]
#
#         for i in range(prediction_length):
#             assert test_target_tensor.shape[0] == prediction.shape[0]
#             print('current epoch: %s, predict %s points' % (epoch, i))
#             mae = mean_absolute_error(test_target_tensor[:, :, i], prediction[:, :, i])
#             rmse = mean_squared_error(test_target_tensor[:, :, i], prediction[:, :, i]) ** 0.5
#             mape = masked_mape_np(test_target_tensor[:, :, i], prediction[:, :, i], 0)
#             print('MAE: %.2f' % (mae))
#             print('RMSE: %.2f' % (rmse))
#             print('MAPE: %.2f' % (mape))
#             print()
#             if sw:
#                 sw.add_scalar('MAE_%s_points' % (i), mae, epoch)
#                 sw.add_scalar('RMSE_%s_points' % (i), rmse, epoch)
#                 sw.add_scalar('MAPE_%s_points' % (i), mape, epoch)

def predict_and_save_results_mstgcn(net, data_loader, data_target_tensor, global_step, metric_method, _mean, _std, params_path, type):
    net.train(False)  # Ensure dropout layers are in test mode

    with torch.no_grad():
        data_target_tensor = data_target_tensor.cpu().numpy()
        loader_length = len(data_loader)
        prediction_list = []
        input_list = []

        for batch_index, batch_data in enumerate(data_loader):
            encoder_inputs, labels = batch_data
            input_list.append(encoder_inputs[:, :, 0:1].cpu().numpy())
            outputs = net(encoder_inputs)
            prediction_list.append(outputs.detach().cpu().numpy())
            if batch_index % 100 == 0:
                print('Predicting data set batch %s / %s' % (batch_index + 1, loader_length))

        input_all = np.concatenate(input_list, axis=0)
        input_all = re_normalization(input_all, _mean, _std)
        prediction = np.concatenate(prediction_list, axis=0)

        print('input:', input_all.shape)
        print('prediction:', prediction.shape)
        print('data_target_tensor:', data_target_tensor.shape)

        if metric_method == 'mask':
            mae = masked_mae_test(data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1), 0.0)
            rmse = masked_rmse_test(data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1), 0.0)
            wmape = masked_wmape_np(data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1), 0)
        else:
            mae = mean_absolute_error(data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1))
            rmse = mean_squared_error(data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1)) ** 0.5
            wmape = masked_wmape_np(data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1), 0)

        # Print results
        print('all MAE: %.2f' % (mae))
        print('all RMSE: %.2f' % (rmse))
        print('all WMAPE: %.4f' % (wmape))
        
        output_filename = os.path.join(params_path, 'output_epoch_%s_%s' % (global_step, type))
        np.savez(output_filename, input=input_all, prediction=prediction, data_target_tensor=data_target_tensor)

        # Daily performance evaluation (last 15 days)
        sensor_id=222
        samples_per_day = 288  # 5-minute intervals (24 * 12)
        num_days = 15  # Last 15 days
        start_index = max(0, data_target_tensor.shape[0] - num_days * samples_per_day)

        daily_metrics = []
        minutes_in_day = np.arange(0, 24*60, 5)  # 5-minute intervals
        time_points = (minutes_in_day) / 60  # Convert to hours
        time_ticks = np.arange(0, 24, 2)
        time_labels = [f'{int(h):02d}:00' for h in time_ticks]

        for day in range(num_days):
            day_start = start_index + day * samples_per_day
            day_end = day_start + samples_per_day
            day_true = np.mean(data_target_tensor[day_start:day_end, sensor_id, :].reshape(samples_per_day, -1), axis=1)
            day_pred = np.mean(prediction[day_start:day_end, sensor_id, :].reshape(samples_per_day, -1), axis=1)


            if metric_method == 'mask':
                day_mae = masked_mae_test(day_true, day_pred, 0.0)
                day_wmape = masked_wmape_np(day_true, day_pred, 0)
            else:
                day_mae = mean_absolute_error(day_true, day_pred)
                day_wmape = np.sum(np.abs(day_true - day_pred)) / np.sum(day_true) * 100

            # Find peak time
            peak_index = min(np.argmax(day_true), len(time_points) - 1)
            peak_time = time_points[peak_index]

            # ±1 hour around peak time
            samples_per_hour = 12
            start_idx = max(0, peak_index - samples_per_hour)
            end_idx = min(len(day_true), peak_index + samples_per_hour + 1)

            peak_predictions = day_pred[start_idx:end_idx]
            peak_actuals = day_true[start_idx:end_idx]
            peak_mae = np.mean(np.abs(peak_predictions - peak_actuals))
            peak_wmape = (np.sum(np.abs(peak_predictions - peak_actuals)) / (np.sum(peak_actuals) + 1e-10)) * 100

            plt.figure(figsize=(15, 6))
            plt.plot(time_points, day_true, 'b-', label='Ground Truth', linewidth=2, alpha=0.7)
            plt.plot(time_points, day_pred, 'r--', label='Prediction', linewidth=2, alpha=0.7)
            plt.axvline(peak_time, color='green', linestyle='--', label='Peak Time', alpha=0.8)
            plt.axvspan(time_points[start_idx], time_points[end_idx-1], color='yellow', alpha=0.2)

            plt.text(peak_time, plt.ylim()[0] + (plt.ylim()[1] - plt.ylim()[0]) * 0.05,
                     f'±1h MAE: {peak_mae:.2f}\n±1h WMAPE: {peak_wmape:.2f}%',
                     color='black', fontsize=10, ha='center', bbox=dict(facecolor='white', alpha=0.8))

            plt.title(f'Day {day+1} Prediction (Node {sensor_id})', fontsize=14)
            plt.xlabel('Time (KST)', fontsize=12)
            plt.ylabel('Crowd Count', fontsize=12)
            plt.legend(fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.xticks(time_ticks, time_labels)
            plt.savefig(os.path.join(params_path, f'{sensor_id}_prediction_day_{day+1}_epoch_{global_step}.png'), dpi=300, bbox_inches='tight')
            plt.close()
            

            plt.tight_layout()
            print(f'Day {day+1} - MAE: {day_mae:.2f}, WMAPE: {day_wmape:.2f}%')
            print(f'Peak Time: {int(peak_time):02d}:{int((peak_time % 1) * 60):02d}')
            print(f'±1 Hour Around Peak MAE: {peak_mae:.2f}, WMAPE: {peak_wmape:.2f}%')
            
            daily_metrics.append([day_mae, day_wmape, peak_mae, peak_wmape])

        daily_metrics = np.array(daily_metrics)
        np.savez(os.path.join(params_path, f'daily_metrics_epoch_{global_step}.npz'), daily_metrics=daily_metrics)
        print("Daily metrics saved.")
    