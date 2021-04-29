% Plots the lerning curves for the specified training runs from data in the
% file "lfw_result.txt" stored in the log directory for the respective
% model.

% MIT License
% 
% Copyright (c) 2016 David Sandberg
% 
% Permission is hereby granted, free of charge, to any person obtaining a copy
% of this software and associated documentation files (the "Software"), to deal
% in the Software without restriction, including without limitation the rights
% to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
% copies of the Software, and to permit persons to whom the Software is
% furnished to do so, subject to the following conditions:
% 
% The above copyright notice and this permission notice shall be included in all
% copies or substantial portions of the Software.
% 
% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
% IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
% FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
% AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
% LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
% OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
% SOFTWARE.

%%
addpath('/home/david/git/facenet/util/');
log_dirs = { '/home/david/logs/facenet' };
%%
res = { ...
{ '20180402-114759', 'vggface2, wd=5e-4, center crop, fixed image standardization' }, ...
};

%%
res = { ...
{ '20180408-102900', 'casia, wd=5e-4, pnlf=5e-4, fixed image standardization' }, ...
};

%%

colors = {'b', 'g', 'r', 'c', 'm', 'y', 'k'};
markers = {'.', 'o', 'x', '+', '*', 's', 'd' };
lines = {'-', '-.', '--', ':' };
fontSize = 6;
lineWidth = 2;
lineStyles = combineStyles(colors, markers);
lineStyles2 = combineStyles(colors, {''}, lines);
legends = cell(length(res),1);
legends_accuracy = cell(length(res),1);
legends_valrate = cell(length(res),1);
var = cell(length(res),1);
for i=1:length(res),
    for k=1:length(log_dirs)
        if exist(fullfile(log_dirs{k}, res{i}{1}), 'dir')
            ld = log_dirs{k};
        end
    end
    filename = fullfile(ld, res{i}{1}, 'stat.h5');
    
    var{i} = readlogs(filename,{'loss', 'reg_loss', 'xent_loss', 'lfw_accuracy', ...
        'lfw_valrate', 'val_loss', 'val_xent_loss', 'val_accuracy', ...
        'accuracy', 'prelogits_norm', 'learning_rate', 'center_loss', ...
        'prelogits_hist', 'accuracy'});
    var{i}.steps = 1:length(var{i}.loss);
    epoch = find(var{i}.lfw_accuracy,1,'last');
    var{i}.epochs = 1:epoch;
    legends{i} = sprintf('%s: %s', res{i}{1}, res{i}{2});
    start_epoch = max(1,epoch-10);
    legends_accuracy{i} = sprintf('%s: %s (%.2f%%)', res{i}{1}, res{i}{2}, mean(var{i}.lfw_accuracy(start_epoch:epoch))*100 );
    legends_valrate{i} = sprintf('%s: %s (%.2f%%)', res{i}{1}, res{i}{2}, mean(var{i}.lfw_valrate(start_epoch:epoch))*100 );
    
    arguments_filename = fullfile(ld, res{i}{1}, 'arguments.txt');
    if exist(arguments_filename)
        str = fileread(arguments_filename);
        var{i}.wd = getParameter(str, 'weight_decay', '0.0');
        var{i}.cl = getParameter(str, 'center_loss_factor', '0.0');
        var{i}.fixed_std = getParameter(str, 'use_fixed_image_standardization', '0');
        var{i}.data_dir = getParameter(str, 'data_dir', '');
        var{i}.lr = getParameter(str, 'learning_rate', '0.1');
        var{i}.epoch_size = str2double(getParameter(str, 'epoch_size', '1000'));
        var{i}.batch_size = str2double(getParameter(str, 'batch_size', '90'));
        var{i}.examples_per_epoch = var{i}.epoch_size*var{i}.batch_size;
        var{i}.mnipc = getParameter(str, 'filter_min_nrof_images_per_class', '-1');
        var{i}.val_step = str2num(getParameter(str, 'validate_every_n_epochs', '10'));
        var{i}.pnlf = getParameter(str, 'prelogits_norm_loss_factor', '-1');
        var{i}.emb_size = getParameter(str, 'embedding_size', '-1');

        fprintf('%s: wd=%s lr=%s, pnlf=%s, data_dir=%s, emb_size=%s\n', ...
            res{i}{1}, var{i}.wd, var{i}.lr, var{i}.pnlf, var{i}.data_dir, var{i}.emb_size);
    end
end;

timestr = datestr(now,'yyyymmdd_HHMMSS');

h = 1; figure(h); close(h); figure(h); hold on; setsize(1.5);
title('LFW accuracy');
xlabel('Steps');
ylabel('Accuracy');
grid on;
N = 1; flt = ones(1,N)/N;
for i=1:length(var),
    plot(var{i}.epochs*1000, filter(flt, 1, var{i}.lfw_accuracy(var{i}.epochs)), lineStyles2{i}, 'LineWidth', lineWidth);
end;
legend(legends_accuracy,'Location','SouthEast','FontSize',fontSize);
v=axis;
v(3:4) = [ 0.95 1.0 ];
axis(v);
accuracy_file_name = sprintf('lfw_accuracy_%s',timestr);
%print(accuracy_file_name,'-dpng')


if 0
    %%
    %h = 2; figure(h); close(h); figure(h); hold on; setsize(1.5);
    h = 1; figure(h); hold on;
    title('LFW validation rate');
    xlabel('Step');
    ylabel('VAL @ FAR = 10^{-3}');
    grid on;
    for i=1:length(var),
        plot(var{i}.epochs*1000, var{i}.lfw_valrate(var{i}.epochs), lineStyles{i}, 'LineWidth', lineWidth);
    end;
    legend(legends_valrate,'Location','SouthEast','FontSize',fontSize);
    v=axis;
    v(3:4) = [ 0.5 1.0 ];
    axis(v);
    valrate_file_name = sprintf('lfw_valrate_%s',timestr);
%    print(valrate_file_name,'-dpng')
end

if 0
    %% Plot cross-entropy loss
    h = 3; figure(h); close(h); figure(h); hold on; setsize(1.5);
    title('Training/validation set cross-entropy loss');
    xlabel('Step');
    title('Training/validation set cross-entropy loss');
    grid on;
    N = 500; flt = ones(1,N)/N;
    for i=1:length(var),
        var{i}.xent_loss(var{i}.xent_loss==0) = NaN;
        plot(var{i}.steps, filter(flt, 1, var{i}.xent_loss), lineStyles2{i}, 'LineWidth', lineWidth);
    end;
    legend(legends, 'Location', 'NorthEast','FontSize',fontSize);

    % Plot cross-entropy loss on validation set
    N = 1; flt = ones(1,N)/N;
    for i=1:length(var),
        v = var{i}.val_xent_loss;
        val_steps = (1:length(v))*var{i}.val_step*1000;
        v(v==0) = NaN;
        plot(val_steps, filter(flt, 1, v), [ lineStyles2{i} '.' ], 'LineWidth', lineWidth);
    end;
    legend(legends, 'Location', 'NorthEast','FontSize',fontSize);
    hold off
    xent_file_name = sprintf('xent_%s',timestr);
    %print(xent_file_name,'-dpng')
end

if 0
    %% Plot accuracy on training set
    h = 32; figure(h); clf; hold on; 
    title('Training/validation set accuracy');
    xlabel('Step');
    ylabel('Training/validation set accuracy');
    grid on;
    N = 500; flt = ones(1,N)/N;
    for i=1:length(var),
        var{i}.accuracy(var{i}.accuracy==0) = NaN;
        plot(var{i}.steps*1000, filter(flt, 1, var{i}.accuracy), lineStyles2{i}, 'LineWidth', lineWidth);
    end;
    legend(legends, 'Location', 'SouthEast','FontSize',fontSize);

    grid on;
    N = 1; flt = ones(1,N)/N;
    for i=1:length(var),
        v = var{i}.val_accuracy;
        val_steps = (1:length(v))*var{i}.val_step*1000;
        v(v==0) = NaN;
        plot(val_steps*1000, filter(flt, 1, v), [ lineStyles2{i} '.' ], 'LineWidth', lineWidth);
    end;
    legend(legends, 'Location', 'SouthEast','FontSize',fontSize);
    hold off
    acc_file_name = sprintf('accuracy_%s',timestr);
    %print(acc_file_name,'-dpng')
end

if 0
    %% Plot prelogits CDF
    h = 35; figure(h); clf; hold on; 
    title('Prelogits histogram');
    xlabel('Epoch');
    ylabel('Prelogits histogram');
    grid on;
    N = 1; flt = ones(1,N)/N;
    for i=1:length(var),
        epoch = var{i}.epochs(end);
        q = cumsum(var{i}.prelogits_hist(:,epoch));
        q2 = q / q(end);
        plot(linspace(0,10,1000), q2, lineStyles2{i}, 'LineWidth', lineWidth);
    end;
    legend(legends, 'Location', 'SouthEast','FontSize',fontSize);
    hold off
end

if 0
    %% Plot prelogits norm
    h = 32; figure(h); clf; hold on; 
    title('Prelogits norm');
    xlabel('Step');
    ylabel('Prelogits norm');
    grid on;
    N = 1; flt = ones(1,N)/N;
    for i=1:length(var),
        plot(var{i}.steps, filter(flt, 1, var{i}.prelogits_norm), lineStyles2{i}, 'LineWidth', lineWidth);
    end;
    legend(legends, 'Location', 'NorthEast','FontSize',fontSize);
    hold off
end

if 0
    %% Plot learning rate
    h = 42; figure(h); clf; hold on; 
    title('Learning rate');
    xlabel('Step');
    ylabel('Learning rate');
    grid on;
    N = 1; flt = ones(1,N)/N;
    for i=1:length(var),
        semilogy(var{i}.epochs, filter(flt, 1, var{i}.learning_rate(var{i}.epochs)), lineStyles2{i}, 'LineWidth', lineWidth);
    end;
    legend(legends, 'Location', 'NorthEast','FontSize',fontSize);
    hold off
end

if 0
    %% Plot center loss
    h = 9; figure(h); close(h); figure(h); hold on; setsize(1.5);
    title('Center loss');
    xlabel('Epochs');
    ylabel('Center loss');
    grid on;
    N = 500; flt = ones(1,N)/N;
    for i=1:length(var),
        if isempty(var{i}.center_loss)
            var{i}.center_loss = ones(size(var{i}.steps))*NaN;
        end;
        var{i}.center_loss(var{i}.center_loss==0) = NaN;
        plot(var{i}.steps/var{i}.epoch_size, filter(flt, 1, var{i}.center_loss), lineStyles2{i}, 'LineWidth', lineWidth);
    end;
    legend(legends, 'Location', 'NorthEast','FontSize',fontSize);
end

if 0
    %% Plot center loss with factor
    h = 9; figure(h); close(h); figure(h); hold on; setsize(1.5);
    title('Center loss with factor');
    xlabel('Epochs');
    ylabel('Center loss * center loss factor');
    grid on;
    N = 500; flt = ones(1,N)/N;
    for i=1:length(var),
        if isempty(var{i}.center_loss)
            var{i}.center_loss = ones(size(var{i}.steps))*NaN;
        end;
        var{i}.center_loss(var{i}.center_loss==0) = NaN;
        plot(var{i}.steps/var{i}.epoch_size, filter(flt, 1, var{i}.center_loss*str2num(var{i}.cl)), lineStyles2{i}, 'LineWidth', lineWidth);
    end;
    legend(legends, 'Location', 'NorthEast','FontSize',fontSize);
end

if 0
    %% Plot total loss
    h = 4; figure(h); close(h); figure(h); hold on; setsize(1.5);
    title('Total loss');
    xlabel('Epochs');
    ylabel('Total loss');
    grid on;
    N = 500; flt = ones(1,N)/N;
    for i=1:length(var),
        var{i}.loss(var{i}.loss==0) = NaN;
        plot(var{i}.steps/var{i}.epoch_size, filter(flt, 1, var{i}.loss), lineStyles2{i}, 'LineWidth', lineWidth);
    end;
    legend(legends, 'Location', 'NorthEast','FontSize',fontSize);
end

if 0
    %% Plot regularization loss
    h = 5; figure(h); close(h); figure(h); hold on; setsize(1.5);
    title('Regularization loss');
    xlabel('Epochs');
    ylabel('Regularization loss');
    grid on;
    N = 500; flt = ones(1,N)/N;
    for i=1:length(var),
        var{i}.reg_loss(var{i}.reg_loss==0) = NaN;
        plot(var{i}.steps/var{i}.epoch_size, filter(flt, 1, var{i}.reg_loss), lineStyles2{i}, 'LineWidth', lineWidth);
    end;
    legend(legends, 'Location', 'NorthEast','FontSize',fontSize);
end
