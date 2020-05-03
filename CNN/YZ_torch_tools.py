# -*- coding: utf-8 -*-
##############################################################################
#                                                                            #
#                                                                            #
#Created on Sat May  2 17:04:38 2020                                         #
#                                                                            #
#@author: yluea                                                              #
#                                                                            #
#                                                                            #
##############################################################################

import sys;
import torch;
import torch.nn as nn;

import numpy as np;
import matplotlib.pyplot as plt;

import time;

## building model
#==============================================================================
class YZ_nn_model(nn.Module):
    def __init__(self, seq_list):
        super(YZ_nn_model,self).__init__();
        self.Seq = seq_list;
        self.layers = nn.ModuleList();
        for i in range(len(self.Seq)):
            #if self.Seq[i][0] == "CNN":
                
                #self.layers.append(self.Seq[i][1]);
            #else:
                #self.layers.append(self.Seq[i][1]); 
            self.layers.append(self.Seq[i][1]); 

    def forward(self, X):        
        for i in range(len(self.layers)):
            #if self.Seq[i][0] == "CNN":
                #X = X.permute(0,2,1);
                #X = self.layers[i](X.permute(0,2,1)).permute(0,2,1);
                #X = X.permute(0,2,1);
            #else:
                #X = self.layers[i](X); 
            X = self.layers[i](X);
        return X;
#==============================================================================
class Permute(nn.Module):
    def __init__(self):
        super(Permute, self).__init__();
        #self.Permute = args;
    def forward(self,x):
        return x.permute(0,2,1);
    
class Torch_max(nn.Module):
    def __init__(self, dim = 1):
        super(Torch_max, self).__init__();
        #self.Permute = args;
        self.dim = dim
    def forward(self, x):
        out,_ = torch.max(x, self.dim);
        return out;
    
class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__();
        self.shape = shape;
    def forward(self, x):
        return x.view(self.shape);   
    
    
#==============================================================================    

class YZ_nn_layer:
    def __init__(self, N_data):
        self.n_nodes = [N_data];
        self.Seq = [];
        
    def YZ_nn_sequential(self,layer:list):
        seq = [];    
        for i in range(len(layer)):
            if layer[i][0] == 'ReLU':
                seq.append(nn.ReLU());

            elif layer[i][0] == 'Linear':
                seq.append(nn.Linear(self.n_nodes[-1], layer[i][1]));
                self.n_nodes.append(layer[i][1]);
                
            elif layer[i][0] == 'Torch_max':
                if len(layer[i]) > 1:
                    seq.append(Torch_max(layer[i][1]));
                else:
                    seq.append(Torch_max());
            elif layer[i][0] == 'Permute':
                seq.append(Permute());

            elif layer[i][0] == 'Sigmoid':
                seq.append(nn.Sigmoid());
            elif layer[i][0] == 'Tanh':
                seq.append(nn.Tanh());
            elif layer[i][0] == 'Softmax':
                seq.append(nn.Softmax(dim = 1));
            elif layer[i][0] == 'Dropout':
                seq.append(nn.Dropout(layer[i][1]));
            elif layer[i][0] == 'Conv1d':
                if len(layer[i]) == 3:
                    seq.append(nn.Conv1d(self.n_nodes[-1], layer[i][1], layer[i][2]));
                elif len(layer[i]) == 4:
                    seq.append(nn.Conv1d(self.n_nodes[-1], layer[i][1], layer[i][2], 
                                        padding=layer[i][3]));
                self.n_nodes.append(layer[i][1]);

            elif layer[i][0] == "MaxPool1d":
                seq.append(nn.MaxPool1d(layer[i][1]));

            elif layer[i][0] == "Embedding":
                seq.append(nn.Embedding(self.n_nodes[-1], layer[i][1]));
                self. n_nodes.append(layer[i][1]);

            else:
                print("Error!\nPlease check your input format!");
                return False;
            
        Seq_item = []; 
        ## Seq_item[0] record if this is a CNN due to permute reason
        ## Seq_item[1] is the nn.Sequential
        #layer_type = int(input('The %s layer!\nIs this a CNN?\n1.[y] 2.[n]\n'\
                               #%len(self.Seq)));
        #if layer_type == 1:
            #Seq_item.append('CNN');
            #seq = [Permute()] + seq + [Permute()];
            
        #else:
            #Seq_item.append('NN');     
        Seq_item.append('YIZHOU_layer'); 

        Seq_item.append(nn.Sequential(*seq));
        self.Seq.append(Seq_item);
                
    def Show(self):
        print(self.Seq);
        
#==============================================================================
class YZ_nn_optimize:
    def __init__(self, model,  train_iter, test_iter = False, device = False,
                 job = 'classification', K_class = 2):
        self.model = model;
        self.device = device;
        if self.device == False:
            self.device = torch.device("cuda:0" \
                                       if torch.cuda.is_available() else "cpu");
        self.train_iter = train_iter;
        self.test_iter = test_iter;

        self.job = job;
        self.K_class = K_class;

        if self.job == 'regression':
            self.K_class = 1;

        print("Doing a " + job + " with " + str(self.K_class) + " label(s)!");

        self.model.to(self.device);

        self.train_losses = [];
        if self.test_iter != False:
            self.test_losses = [];
        

    ## make a process bar
    def YZ_process_bar(self, ratio, comments = False, overwrite = True, length = 50):
        bar = 'Yizhou said 欲速则不达，施主稍安勿躁: | ';
        i = 0;
        while i < ratio * length:
            bar += '▒';
            i += 1;
        while i < length:
            bar += '░';
            i += 1;        
        bar += (' | %s%%'%(int(ratio*1000)/10));
        if ratio == 1:
            bar += ' (^_^)/ Done!'
        if comments != False:
            bar += ('\n' + str(comments));
        if overwrite == True:
            print('\r', end='');
        else:
            print('\n',end = '');
        print(bar, end='');
        sys.stdout.flush();

    ## GD
    def Step_gradient_descent(self, data_iter, process = 'testing'):
        loss_list = [];
        for inputs, targets in data_iter:
            if self.job == 'classification':
                targets = torch.nn.functional.one_hot(targets, self.K_class).float();   
            elif self.job == 'regression':
                targets = targets.view(-1,1).float();
            inputs, targets  =  inputs.to(self.device), targets.to(self.device);
            self.optimizer.zero_grad();
            outputs = self.model(inputs);
            loss = self.criterion(outputs, targets);

            if process == 'training':
                loss.backward();
                self.optimizer.step();

            loss_list.append(loss.item());
        return np.mean(loss_list);

    def Classification_rate(self, data_iter):
        n_correct = 0.0;
        n_total = 0.0;
        for inputs, targets in data_iter:
            targets = torch.nn.functional.one_hot(targets, self.K_class).float();
            outputs = self.model(inputs);
            prediction = (torch.argmax(outputs, dim=1));

            n_total += targets.shape[0];
            n_correct += (torch.argmax(targets, dim=1) == prediction).sum().item();
            rate = n_correct / n_total;
        print("The classification rate for this dataset is %s%%" %(int(rate * 1000)/10));
        return rate;


    def Optimizing(self, lr = 1e-3, criterion = False, optimizer = False, 
                   epochs = False, plot_epoch = False):
        if criterion == False:
            criterion = nn.BCEWithLogitsLoss();
        self.criterion = criterion;
        if optimizer == False:
            optimizer = torch.optim.Adam(self.model.parameters(),lr);
        self.optimizer = optimizer;
        if epochs == False:
            epochs = 10;
        if plot_epoch == False:
            plot_epoch = int(epochs/10);
        if plot_epoch <= 1:
            plot_epoch = 1;

        self.performance = []; ## record time
        start = time.time();## set the timer starting!

        for it in range(epochs):
            train_loss = self.Step_gradient_descent(self.train_iter, process='training');
            if self.test_iter != False:
                test_loss = self.Step_gradient_descent(self.test_iter, process='testing');

            if it%plot_epoch  == 0:
                dt = time.time() - start;
                if self.test_iter != False:
                    nn_comments = "Epoch (%d / %d)...Train_Loss: %.3e...\
                        Test_loss: %.3e...Duration: %.3e sec"%(it+1, epochs,\
                        train_loss, test_loss, dt);
                else:
                    nn_comments = "Epoch (%d / %d)...Train_Loss: %.3e...\
                        ...Duration: %.3e sec"%(it+1, epochs,\
                        train_loss,  dt);
                
                self.YZ_process_bar((it+1)/epochs*1.0, comments=nn_comments, overwrite = False);

                self.train_losses.append(train_loss);
                if self.test_iter != False:
                    self.test_losses.append(test_loss);
                self.performance.append(dt);

            if it == epochs-1:                
                self.YZ_process_bar((it+1)/epochs*1.0, overwrite = False);
        
        plt.figure();
        plt.step(range(len(self.train_losses)), self.train_losses, c = 'r', 
                 label = 'training');
        if self.test_iter != False:
            plt.step(range(len(self.test_losses)), self.test_losses, c = 'b',
                 label = 'testing');
        plt.xlabel('Epochs/%s'%(plot_epoch));
        plt.ylabel('Loss');
        plt.legend();
        plt.title('Optimization Curve');

        plt.figure();
        plt.plot(self.performance, marker = 's');
        plt.xlabel('Epochs/%s'%(plot_epoch));
        plt.ylabel('Time used');
        #plt.legend();
        plt.title('Performance');
        
        plt.show();

        if self.job == 'classification':
            print("Training:");
            self.CR_train = self.Classification_rate(self.train_iter);
            if self.test_iter != False:
                print("Testing:");
                self.CR_test = self.Classification_rate(self.test_iter);


