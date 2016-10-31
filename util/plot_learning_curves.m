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
log_dir = '/media/david/BigDrive/DeepLearning/logs/facenet';

% Weight decay
% res = { ...
% { '20161005-075946', 'wd=5e-6, cl=0.0' }, ...
% { '20161005-225802', 'wd=1e-5, cl=0.0' }, ...
% { '20161006-084144', 'wd=2e-5, cl=0.0' }, ...
% { '20161006-183744', 'wd=5e-5, cl=0.0' }, ...
% { '20161007-041018', 'wd=1e-4, cl=0.0' }, ...
% { '20161007-171453', 'wd=2e-4, cl=0.0' }, ...
% { '20161008-033853', 'wd=5e-4, cl=0.0' }, ...
% { '20161008-141202', 'wd=1e-3, cl=0.0' }, ...
% { '20161026-003440', 'wd=2e-4, cl=0.0' }, ...
% };

% Center loss
% res = { ...
% { '20161007-171453', 'wd=2e-4, cl=0.0' }, ...
% { '20161029-124259', 'wd=2e-4, cl=1e-5' }, ...
% { '20161030-023650', 'wd=2e-4, cl=2e-5' }, ...
% { '20161030-234243', 'wd=2e-4, cl=5e-5' }, ...
% };

% Weight decay and Center loss
res = { ...
{ '20161007-041018', 'wd=1e-4, cl=0.0' }, ...
{ '20161007-171453', 'wd=2e-4, cl=0.0' }, ...
{ '20161008-033853', 'wd=5e-4, cl=0.0' }, ...
{ '20161029-124259', 'wd=2e-4, cl=1e-5' }, ...
{ '20161030-023650', 'wd=2e-4, cl=2e-5' }, ...
{ '20161030-234243', 'wd=2e-4, cl=5e-5' }, ...
};

legends = cell(length(res),1);
legends_accuracy = {};
legends_valrate = {};
for i=1:length(res),
    filename = sprintf('%s/%s/lfw_result.txt', log_dir, res{i}{1});
    a=importdata(filename);
    steps{i} = a(:,1);
    accuracy{i} = a(:,2);
    valrate{i} = a(:,3);
    if ~isempty(res{i}{2})
        leg = res{i}{2};
    else
        leg = res{i}{1};
    end;
    legends_accuracy{i} = sprintf('%s (%.3f)', leg, accuracy{i}(end) );
    legends_valrate{i} = sprintf('%s (%.3f)', leg, valrate{i}(end) );
    
    filename = sprintf('%s/%s/revision_info.txt', log_dir, res{i}{1});
    fid = fopen(filename,'rt');
    str = fgetl(fid);
    fclose(fid);
    %fprintf('%s\n', str);
    split_args = strsplit(str);
    ix = find(~cellfun(@isempty,strfind(split_args,'weight_decay')));
    wd = '0.0';
    if ~isempty(ix)
        wd = split_args{ix+1};
    end;
    cl = '0.0';
    ix = find(~cellfun(@isempty,strfind(split_args,'center_loss')));
    if ~isempty(ix)
        cl = split_args{ix+1};
    end;
    disp(sprintf('%s: wd=%s cl=%s', res{i}{1}, wd, cl));
    xxx = 1;
end;

timestr = datestr(now,'yyyymmdd_HHMMSS');

figure(1); clf; hold on;
title('LFW accuracy');
xlabel('Step');
ylabel('Accuracy');
grid on;
for i=1:length(res),
    plot(steps{i}, accuracy{i}, 'LineWidth', 2);
end;
legend(legends_accuracy,'Location','SouthEast');
v=axis;
v(3:4) = [ 0.9 1.0 ];
axis(v);
accuracy_file_name = sprintf('lfw_accuracy_%s',timestr);
print(accuracy_file_name,'-dpng')

figure(2); clf; hold on;
title('LFW validation rate');
xlabel('Step');
ylabel('VAL @ FAR = 10^{-3}');
grid on;
for i=1:length(res),
    plot(steps{i}, valrate{i}, 'LineWidth', 2);
end;
legend(legends_valrate,'Location','SouthEast');
v=axis;
v(3:4) = [ 0.5 1.0 ];
axis(v);
valrate_file_name = sprintf('lfw_valrate_%s',timestr);
print(valrate_file_name,'-dpng')
