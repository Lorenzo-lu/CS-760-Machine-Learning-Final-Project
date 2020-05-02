# -*- coding: utf-8 -*-
"""
Created on Fri May  1 19:45:07 2020

@author: yluea
"""


import sys;
import torch;
import torch.nn as nn;
#import torchtext.data as ttd;
#from torchtext.vocab import GloVe;

import numpy as np;
import matplotlib.pyplot as plt;
import pandas as pd;
pd.options.mode.chained_assignment = None; ## avoid warning
#from datetime import datetime;
import time;

class YZ_torch_std:
    def __init__(self, model,  device, train_iter, test_iter, 
                 job = 'classification', K_class = 2):
        self.model = model;
        self.device = device;
        self.train_iter = train_iter;
        self.test_iter = test_iter;

        self.job = job;
        self.K_class = K_class;

        if self.job == 'regression':
            self.K_class = 1;

        print("Doing a " + job + " with " + str(self.K_class) + " label(s)!");

        self.model.to(self.device);

        self.train_losses = [];
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
            test_loss = self.Step_gradient_descent(self.test_iter, process='testing');

            if it%plot_epoch  == 0:
                dt = time.time() - start;
                nn_comments = "Epoch (%d / %d)...Train_Loss: %.3e...Test_loss: %.3e...Duration: %.3e sec"\
                %(it+1, epochs, train_loss, test_loss, dt);
                self.YZ_process_bar((it+1)/epochs*1.0, comments=nn_comments, overwrite = False);

                self.train_losses.append(train_loss);
                self.test_losses.append(test_loss);
                self.performance.append(dt);

            if it == epochs-1:                
                self.YZ_process_bar((it+1)/epochs*1.0, overwrite = False);
        
        plt.figure();
        plt.step(range(len(self.train_losses)), self.train_losses, c = 'r', 
                 label = 'training');
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
            print("Testing:");
            self.CR_test = self.Classification_rate(self.test_iter);
