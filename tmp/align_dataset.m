# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

% LFW
% source_path = '/home/david/datasets/lfw/raw';
% target_path = '/home/david/datasets/lfw/lfw_mtcnnalign_160';
% image_size = 160 + 0;
% margin = round(image_size*0.2) + 0;

% FaceScrub
% source_path = '/home/david/datasets/facescrub/facescrub/';
% target_path = '/home/david/datasets/facescrub/facescrub_mtcnnalign_182_160';
% failed_images_list = '/home/david/datasets/facescrub/facescrub_mtcnnalign_182_160/failed_images.txt';
% image_size = 160 + 12;
% margin = round(image_size*0.2) + 12;

source_path = '/home/david/datasets/casia/CASIA-maxpy-clean/';
target_path = '/home/david/datasets/casia/casia_maxpy_mtcnnalign_182_160';
failed_images_list = '/home/david/datasets/casia/casia_maxpy_mtcnnalign_182_160/failed_images.txt';
image_size = 160 + 12;
margin = round(image_size*0.2) + 12;

image_extension = 'png';
minsize=20; %minimum size of face
use_new = 0;

caffe_path='/home/david/repo2/caffe/matlab';
pdollar_toolbox_path='/home/david/repo2/toolbox';
if use_new
    caffe_model_path='/home/david/repo2/MTCNN_face_detection_alignment/code/codes/MTCNNv2/model';
else
    caffe_model_path='/home/david/repo2/MTCNN_face_detection_alignment/code/codes/MTCNNv1/model';
end;
addpath(genpath(caffe_path));
addpath(genpath(pdollar_toolbox_path));

caffe.set_mode_gpu();
caffe.set_device(0);

%three steps's threshold
threshold=[0.6 0.7 0.7];

%scale factor
factor=0.709;

%load caffe models
if use_new
    prototxt_dir =  strcat(caffe_model_path,'/det4.prototxt');
    model_dir =  strcat(caffe_model_path,'/det4.caffemodel');
end;
%faces=cell(0);

k = 0;
classes = dir(source_path);
%classes = classes(randperm(length(classes)));
for i=1:length(classes),
    if classes(i).name(1)~='.'
        source_class_path = sprintf('%s/%s', source_path, classes(i).name);
        target_class_path = sprintf('%s/%s', target_path, classes(i).name);
        imgs = dir(source_class_path);
        %imgs = imgs(randperm(length(imgs)));
        if ~exist(target_class_path, 'dir'),
            mkdir(target_class_path);
        end;
        for j=1:length(imgs),
            if imgs(j).isdir==0
                [pathstr,name,ext] = fileparts(imgs(j).name);
                target_img_path = sprintf('%s/%s.%s', target_class_path, name, image_extension);
                if ~exist(target_img_path,'file') && any([ strcmpi(ext,'.jpg') strcmpi(ext,'.jpeg') strcmpi(ext,'.png') strcmpi(ext,'.gif') ])
                    if mod(k,1000)==0
                        fprintf('Resetting GPU\n');
                        caffe.reset_all();
                        caffe.set_mode_gpu();
                        caffe.set_device(0);
                        prototxt_dir = strcat(caffe_model_path,'/det1.prototxt');
                        model_dir = strcat(caffe_model_path,'/det1.caffemodel');
                        PNet=caffe.Net(prototxt_dir,model_dir,'test');
                        prototxt_dir = strcat(caffe_model_path,'/det2.prototxt');
                        model_dir = strcat(caffe_model_path,'/det2.caffemodel');
                        RNet=caffe.Net(prototxt_dir,model_dir,'test');
                        prototxt_dir = strcat(caffe_model_path,'/det3.prototxt');
                        model_dir = strcat(caffe_model_path,'/det3.caffemodel');
                        ONet=caffe.Net(prototxt_dir,model_dir,'test');
                        if use_new
                            prototxt_dir = strcat(caffe_model_path,'/det4.prototxt');
                            model_dir = strcat(caffe_model_path,'/det4.caffemodel');
                            LNet=caffe.Net(prototxt_dir,model_dir,'test');
                        end;
                    end;
                        
                    source_img_path = sprintf('%s/%s', source_class_path, imgs(j).name);
                    % source_img_path = '/home/david/datasets/facescrub/facescrub//Billy_Zane/095f83fefdf1dc493c013edb1ef860001193e8d9.jpg'
                    try
                        img = imread(source_img_path);
                    catch exception
                        fprintf('Unexpected error (%s): %s\n', exception.identifier, exception.message);
                        continue;
                    end;
                    fprintf('%6d: %s\n', k, source_img_path);
                    if length(size(img))<3
                        img = repmat(img,[1,1,3]);
                    end;
                    img_size = size(img); % [height, width, channels]
                    img_size = fliplr(img_size(1:2));  % [x,y]
                    if use_new
                        [boundingboxes, points]=detect_face_v2(img,minsize,PNet,RNet,ONet,LNet,threshold,false,factor);
                    else
                        [boundingboxes, points]=detect_face_v1(img,minsize,PNet,RNet,ONet,threshold,false,factor);
                    end;
                    nrof_faces = size(boundingboxes,1);
                    det = boundingboxes;
                    if nrof_faces>0
                        if nrof_faces>1
                            % select the faces with the largest bounding box
                            %  closest to the image center
                            bounding_box_size = (det(:,3)-det(:,1)).*(det(:,4)-det(:,2));
                            img_center = img_size / 2;
                            offsets = [ (det(:,1)+det(:,3))/2 (det(:,2)+det(:,4))/2 ] - ones(nrof_faces,1)*img_center;
                            offset_dist_squared = sum(offsets.^2,2);
                            [a, index] = max(bounding_box_size-offset_dist_squared*2.0); % some extra weight on the centering
                            det = det(index,:);
                            points = points(:,index);
                        end;
%                         if nrof_faces>0
%                             figure(1); clf;
%                             imshow(img);
%                             hold on;
%                             plot(points(1:5,1),points(6:10,1),'g.','MarkerSize',10);
%                             bb = round(det(1:4));
%                             rectangle('Position',[bb(1) bb(2) bb(3)-bb(1) bb(4)-bb(2)],'LineWidth',2,'LineStyle','-')
%                             xxx = 1;
%                         end;
                        det(1) = max(det(1)-margin/2, 1);
                        det(2) = max(det(2)-margin/2, 1);
                        det(3) = min(det(3)+margin/2, img_size(1));
                        det(4) = min(det(4)+margin/2, img_size(2));
                        det(1:4) = round(det(1:4));
                        
                        img = img(det(2):det(4),det(1):det(3),:);
                        img = imresize(img, [image_size, image_size]);
                        
                        imwrite(img, target_img_path);
                        k = k + 1;
                    else
                        fprintf('Detection failed: %s\n', source_img_path);
                        fid = fopen(failed_images_list,'at');
                        if fid>=0
                            fprintf(fid, '%s\n', source_img_path);
                            fclose(fid);
                        end;
                    end;
                    if mod(k,100)==0
                        xxx = 1;
                    end;
                end;
            end;
        end;
    end;
end;
