
from pkgutil import get_data
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import CDFM
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
from torch.optim import lr_scheduler 
import copy
import random

warnings.filterwarnings('ignore')
def seed_everything(seed=2021):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args) 

    # for SIN normalization
    def _get_U_V(self):
        train_data, train_loader = self._get_data('train')
        X = []
        Y = []
        for data in train_loader:
            lookback_window = data[0]  # [B, L, N] 
            forcast_window = data[1]   # [B, H, N] 
            X.append(lookback_window)
            Y.append(forcast_window)
        X = np.concatenate(X, axis=0) # [8448, 96, 7]
        Y = np.concatenate(Y, axis=0) # (8448, 96, 7)
        selected_U = []
        selected_V = []
        for c in range(self.args.enc_in):
            X_c = X[:,:,c].copy()
            Y_c = Y[:,:,c].copy()
            XtY = np.matmul(X_c.T, Y_c)          # [L,H]
            U, S, VT = np.linalg.svd(XtY)    # U [L,L] S [L,H] V [H,H]
            V = VT.T
            tau = self.args.tau
            selected_U_c = U[:,:tau] # [L,tau]
            selected_U.append(selected_U_c.reshape(1, -1, self.args.tau))
            selected_V_c = V[:,:tau] # [H,tau]
            selected_V.append(selected_V_c.reshape(1, -1, self.args.tau))

        selected_U = np.concatenate(selected_U, axis=0) #[N, L, tau]
        selected_V = np.concatenate(selected_V, axis=0) #[N, H, tau]
        print("selected_U and V done!!")
        return selected_U, selected_V
    
    def _get_metric_G(self):
        train_data, train_loader = self._get_data(flag='train')

        # Calculate the non-stationarity of channels
        all_sigma_t = []
        for data in train_loader:
            x = data[0]
            y = data[1]
            x_y = torch.cat([x,y], dim=1)
            sigma_t = torch.std(x_y, dim=1)
            all_sigma_t.append(sigma_t)

        all_sigma_t = torch.cat(all_sigma_t, dim=0)
        all_sigma_t = torch.mean(all_sigma_t, dim=0)
        all_x = torch.tensor(train_data.origin_data_x)

        # Remove outliers
        mean = torch.mean(all_x, axis=0)
        std = torch.std(all_x, axis=0)
        all_x[all_x > mean + 2*std] = 0
        all_x[all_x < mean - 2*std] = 0
        mean = torch.mean(all_x, axis=0)
        std = torch.std(all_x, axis=0)
        all_x = (all_x - mean) / (std + 1e-5)

        # Calculate correlation between channels
        cov_matrix = torch.mm(all_x.t(), all_x) / (all_x.shape[0] - 1)

        # Calculate metric G
        G = torch.mean(cov_matrix, dim=-1) + all_sigma_t
        # print(G)
        return G
     
    def _build_model(self):
        # if self.args.use_norm == 3:
        #     self.args.selected_U, self.args.selected_V = self._get_U_V()
        self.args.G = self._get_metric_G()     
        
        model_dict = {
            'CDFM': CDFM,
        }
        model = model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        
        # calculate memory usage
        total = sum([param.nelement() for param in model.parameters()])
        total += sum(p.numel() for p in model.buffers())
        print("Number of parameters: %.2fM" % (total/(1024*1024))) 
        # exit()
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader
    
    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        self.criterion = nn.MSELoss()
        self.criterion_l1 = nn.L1Loss()
    

    def vali(self, vali_data, vali_loader, criterion, epoch, flag, is_selecting):
        total_loss = []
        total_loss_s = []
        total_loss_fused = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                
                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_label = batch_x[:, -self.args.label_len:, :]
                dec_inp = torch.cat([dec_label, dec_inp], dim=1).float()
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'CDFM' in self.args.model:
                            outputs, outputs_s, outputs_ns, sel_chs = self.model(batch_x, self.args.is_shifted)
                        elif 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, self.args.is_shifted)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, self.args.is_shifted)
                else:
                    if 'CDFM' in self.args.model:
                        outputs, outputs_s, outputs_ns, sel_chs = self.model(batch_x, self.args.is_shifted)
                    elif 'Linear' in self.args.model or 'TST' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, self.args.is_shifted)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, self.args.is_shifted)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                loss = criterion(outputs, batch_y) 
                
                if flag == 'val' and is_selecting == True:
                    loss_s = []
                    loss_fused = []
                    for i in range(self.args.enc_in):
                        loss_s.append((self.criterion(outputs_s[:,:,i], batch_y[:,:,i])).item())
                        loss_fused.append((self.criterion(outputs[:,:,i], batch_y[:,:,i])).item())
                    loss_s = np.array(loss_s)
                    loss_fused = np.array(loss_fused)
                    total_loss_s.append(loss_s.reshape(1,-1))
                    total_loss_fused.append(loss_fused.reshape(1,-1))

                total_loss.append(loss.cpu().item())
        total_loss = np.average(total_loss)

        if flag == 'val' and is_selecting == True:
            total_loss_s = np.concatenate(total_loss_s, axis=0)
            total_loss_fused = np.concatenate(total_loss_fused, axis=0)
            total_loss_s = np.mean(total_loss_s, axis=0)
            total_loss_fused = np.mean(total_loss_fused, axis=0)
            # print(total_loss_s )
            # print(total_loss_fused)
            
            
            threshold = 0.1*total_loss_s
            is_shifted = (total_loss_fused >= total_loss_s + threshold) 
            self.args.is_shifted = torch.tensor(is_shifted)

            # print(is_shifted)

        if epoch == self.args.pre_epochs and is_selecting == True:
            return total_loss, self.args.is_shifted
        

        self.model.train()
         
        return total_loss, None

    def train(self, setting, is_selecting=False):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        self._select_criterion() 
        
        scheduler = lr_scheduler.OneCycleLR(optimizer = model_optim,
                                            steps_per_epoch = train_steps,
                                            pct_start = self.args.pct_start,
                                            epochs = self.args.train_epochs,
                                            max_lr = self.args.learning_rate)
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            seed_everything()

            self.model.train()
            

            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_label = batch_x[:, -self.args.label_len:, :]
                dec_inp = torch.cat([dec_label, dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'CDFM' in self.args.model:
                            outputs, outputs_s, outputs_ns, sel_chs = self.model(batch_x, self.args.is_shifted)
                        elif 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, self.args.is_shifted)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, self.args.is_shifted)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = self.criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if 'CDFM' in self.args.model:
                        outputs, outputs_s, outputs_ns, sel_chs = self.model(batch_x, self.args.is_shifted)
                    elif 'Linear' in self.args.model or 'TST' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, self.args.is_shifted)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, self.args.is_shifted)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)


                    loss = self.criterion(outputs, batch_y)
                    loss += self.criterion(outputs_s, batch_y) + self.criterion(outputs_ns, batch_y) 
                    loss = loss.mean()
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * (
                            (self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
                    model_optim.zero_grad()
   
                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()
                    
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss, is_shifted = self.vali(vali_data, vali_loader, self.criterion, epoch, flag='val', is_selecting=is_selecting)
            test_loss = self.vali(test_data, test_loader, self.criterion, epoch, flag='test', is_selecting=is_selecting)[0]

            print(
                "Backbone Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            
            if early_stopping.early_stop:
                print("Early stopping")
                break 

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))


            if epoch == self.args.pre_epochs and is_selecting == True:
                return is_shifted 
        
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'), map_location=torch.device('cpu')))
        
        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                input_x = batch_x
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)  
                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_label = batch_x[:, -self.args.label_len:, :]
                dec_inp = torch.cat([dec_label, dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'CDFM' in self.args.model:
                            outputs, outputs_s, outputs_ns, _ = self.model(batch_x, self.args.is_shifted)
                        elif 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, self.args.is_shifted)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, self.args.is_shifted)
                else:
                    if 'CDFM' in self.args.model:
                        outputs, outputs_s, outputs_ns, _ = self.model(batch_x, self.args.is_shifted)
                    elif 'Linear' in self.args.model or 'TST' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, self.args.is_shifted)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, self.args.is_shifted)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
            
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                pred = outputs  
                true = batch_y               
                preds.append(pred)
                trues.append(true)
                # if i % 20 == 0:
                #     input = input_x.detach().cpu().numpy()
                #     gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                #     pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                #     visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        if self.args.test_flop:
            test_params_flop((batch_x.shape[1], batch_x.shape[2]))
            exit()
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
                    
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # for i in range(self.args.enc_in):
        #     mae, mse, rmse, mape, mspe, rse, corr = metric(preds[:,:,i], trues[:,:,i])
        #     print('mse:{}, mae:{}'.format(mse, mae))

        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rse:{}, corr:{}'.format(mse, mae, rse, corr))
        f.write('\n')
        f.write('\n')
        f.close()
       
        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe,rse, corr]))
        # np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)
        return mse, mae

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(
                    batch_y.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'CDFM' in self.args.model:
                            outputs, outputs_s, outputs_ns, _ = self.model(batch_x, self.args.is_shifted)
                        elif 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, self.args.is_shifted)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, self.args.is_shifted)
                else:
                    if 'CDFM' in self.args.model:
                        outputs, outputs_s, outputs_ns, _ = self.model(batch_x, self.args.is_shifted)
                    elif 'Linear' in self.args.model or 'TST' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, self.args.is_shifted)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, self.args.is_shifted)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return

