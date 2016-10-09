% MIT License
% 
% Copyright (c) 2016 Kaipeng Zhang
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

function [total_boxes, points] = detect_face_v2(img,minsize,PNet,RNet,ONet,LNet,threshold,fastresize,factor)
	%im: input image
	%minsize: minimum of faces' size
	%pnet, rnet, onet: caffemodel
	%threshold: threshold=[th1 th2 th3], th1-3 are three steps's threshold
	%fastresize: resize img from last scale (using in high-resolution images) if fastresize==true
	factor_count=0;
	total_boxes=[];
	points=[];
	h=size(img,1);
	w=size(img,2);
	minl=min([w h]);
    img=single(img);
	if fastresize
		im_data=(single(img)-127.5)*0.0078125;
    end
    m=12/minsize;
	minl=minl*m;
	%creat scale pyramid
    scales=[];
	while (minl>=12)
		scales=[scales m*factor^(factor_count)];
		minl=minl*factor;
		factor_count=factor_count+1;
	end
	%first stage
	for j = 1:size(scales,2)
		scale=scales(j);
		hs=ceil(h*scale);
		ws=ceil(w*scale);
		if fastresize
			im_data=imResample(im_data,[hs ws],'bilinear');
		else 
			im_data=(imResample(img,[hs ws],'bilinear')-127.5)*0.0078125;
		end
		PNet.blobs('data').reshape([hs ws 3 1]);
		out=PNet.forward({im_data});
		boxes=generateBoundingBox(out{2}(:,:,2),out{1},scale,threshold(1));
		%inter-scale nms
		pick=nms(boxes,0.5,'Union');
		boxes=boxes(pick,:);
		if ~isempty(boxes)
			total_boxes=[total_boxes;boxes];
		end
	end
	numbox=size(total_boxes,1);
	if ~isempty(total_boxes)
		pick=nms(total_boxes,0.7,'Union');
		total_boxes=total_boxes(pick,:);
		bbw=total_boxes(:,3)-total_boxes(:,1);
		bbh=total_boxes(:,4)-total_boxes(:,2);
		total_boxes=[total_boxes(:,1)+total_boxes(:,6).*bbw total_boxes(:,2)+total_boxes(:,7).*bbh total_boxes(:,3)+total_boxes(:,8).*bbw total_boxes(:,4)+total_boxes(:,9).*bbh total_boxes(:,5)];	
		total_boxes=rerec(total_boxes);
		total_boxes(:,1:4)=fix(total_boxes(:,1:4));
		[dy edy dx edx y ey x ex tmpw tmph]=pad(total_boxes,w,h);
	end
	numbox=size(total_boxes,1);
	if numbox>0
		%second stage
 		tempimg=zeros(24,24,3,numbox);
		for k=1:numbox
			tmp=zeros(tmph(k),tmpw(k),3);
			tmp(dy(k):edy(k),dx(k):edx(k),:)=img(y(k):ey(k),x(k):ex(k),:);
			tempimg(:,:,:,k)=imResample(tmp,[24 24],'bilinear');
		end
        tempimg=(tempimg-127.5)*0.0078125;
		RNet.blobs('data').reshape([24 24 3 numbox]);
		out=RNet.forward({tempimg});
		score=squeeze(out{2}(2,:));
		pass=find(score>threshold(2));
		total_boxes=[total_boxes(pass,1:4) score(pass)'];
		mv=out{1}(:,pass);
		if size(total_boxes,1)>0		
			pick=nms(total_boxes,0.7,'Union');
			total_boxes=total_boxes(pick,:);     
            total_boxes=bbreg(total_boxes,mv(:,pick)');	
            total_boxes=rerec(total_boxes);
		end
		numbox=size(total_boxes,1);
		if numbox>0
			%third stage
			total_boxes=fix(total_boxes);
			[dy edy dx edx y ey x ex tmpw tmph]=pad(total_boxes,w,h);
            tempimg=zeros(48,48,3,numbox);
			for k=1:numbox
				tmp=zeros(tmph(k),tmpw(k),3);
				tmp(dy(k):edy(k),dx(k):edx(k),:)=img(y(k):ey(k),x(k):ex(k),:);
				tempimg(:,:,:,k)=imResample(tmp,[48 48],'bilinear');
			end
			tempimg=(tempimg-127.5)*0.0078125;
			ONet.blobs('data').reshape([48 48 3 numbox]);
			out=ONet.forward({tempimg});
			score=squeeze(out{3}(2,:));
			points=out{2};
			pass=find(score>threshold(3));
			points=points(:,pass);
			total_boxes=[total_boxes(pass,1:4) score(pass)'];
			mv=out{1}(:,pass);
			bbw=total_boxes(:,3)-total_boxes(:,1)+1;
            bbh=total_boxes(:,4)-total_boxes(:,2)+1;
            points(1:5,:)=repmat(bbw',[5 1]).*points(1:5,:)+repmat(total_boxes(:,1)',[5 1])-1;
            points(6:10,:)=repmat(bbh',[5 1]).*points(6:10,:)+repmat(total_boxes(:,2)',[5 1])-1;
			if size(total_boxes,1)>0				
				total_boxes=bbreg(total_boxes,mv(:,:)');	
                pick=nms(total_boxes,0.7,'Min');
				total_boxes=total_boxes(pick,:);  				
                points=points(:,pick);
			end
		end
		numbox=size(total_boxes,1);
		%extended stage
		if numbox>0 
			tempimg=zeros(24,24,15,numbox);
			patchw=max([total_boxes(:,3)-total_boxes(:,1)+1 total_boxes(:,4)-total_boxes(:,2)+1]');
			patchw=fix(0.25*patchw);	
			tmp=find(mod(patchw,2)==1);
			patchw(tmp)=patchw(tmp)+1;
			pointx=ones(numbox,5);
			pointy=ones(numbox,5);
			for k=1:5
				tmp=[points(k,:);points(k+5,:)];
				x=fix(tmp(1,:)-0.5*patchw);
				y=fix(tmp(2,:)-0.5*patchw);
				[dy edy dx edx y ey x ex tmpw tmph]=pad([x' y' x'+patchw' y'+patchw'],w,h);
				for j=1:numbox
					tmpim=zeros(tmpw(j),tmpw(j),3);
					tmpim(dy(j):edy(j),dx(j):edx(j),:)=img(y(j):ey(j),x(j):ex(j),:);
					tempimg(:,:,(k-1)*3+1:(k-1)*3+3,j)=imResample(tmpim,[24 24],'bilinear');
				end
			end
			LNet.blobs('data').reshape([24 24 15 numbox]);
			tempimg=(tempimg-127.5)*0.0078125;
			out=LNet.forward({tempimg});
			score=squeeze(out{3}(2,:));
			for k=1:5
				tmp=[points(k,:);points(k+5,:)];
				%do not make a large movement
				temp=find(abs(out{k}(1,:)-0.5)>0.35);
				if ~isempty(temp)
					l=length(temp);
					out{k}(:,temp)=ones(2,l)*0.5;
				end
				temp=find(abs(out{k}(2,:)-0.5)>0.35);  
				if ~isempty(temp)
					l=length(temp);
					out{k}(:,temp)=ones(2,l)*0.5;
				end
				pointx(:,k)=(tmp(1,:)-0.5*patchw+out{k}(1,:).*patchw)';
				pointy(:,k)=(tmp(2,:)-0.5*patchw+out{k}(2,:).*patchw)';
			end
			for j=1:numbox
				points(:,j)=[pointx(j,:)';pointy(j,:)'];
			end
		end
    end 	
end

function [boundingbox] = bbreg(boundingbox,reg)
	%calibrate bouding boxes
	if size(reg,2)==1
		reg=reshape(reg,[size(reg,3) size(reg,4)])';
	end
	w=[boundingbox(:,3)-boundingbox(:,1)]+1;
	h=[boundingbox(:,4)-boundingbox(:,2)]+1;
	boundingbox(:,1:4)=[boundingbox(:,1)+reg(:,1).*w boundingbox(:,2)+reg(:,2).*h boundingbox(:,3)+reg(:,3).*w boundingbox(:,4)+reg(:,4).*h];
end

function [boundingbox reg] = generateBoundingBox(map,reg,scale,t)
	%use heatmap to generate bounding boxes
    stride=2;
    cellsize=12;
    boundingbox=[];
	map=map';
	dx1=reg(:,:,1)';
	dy1=reg(:,:,2)';
	dx2=reg(:,:,3)';
	dy2=reg(:,:,4)';
    [y x]=find(map>=t);
	a=find(map>=t); 
    if size(y,1)==1
		y=y';x=x';score=map(a)';dx1=dx1';dy1=dy1';dx2=dx2';dy2=dy2';
	else
		score=map(a);
    end   
	reg=[dx1(a) dy1(a) dx2(a) dy2(a)];
	if isempty(reg)
		reg=reshape([],[0 3]);
	end
    boundingbox=[y x];
    boundingbox=[fix((stride*(boundingbox-1)+1)/scale) fix((stride*(boundingbox-1)+cellsize-1+1)/scale) score reg];
end

function pick = nms(boxes,threshold,type)
	%NMS
	if isempty(boxes)
	  pick = [];
	  return;
	end
	x1 = boxes(:,1);
	y1 = boxes(:,2);
	x2 = boxes(:,3);
	y2 = boxes(:,4);
	s = boxes(:,5);
	area = (x2-x1+1) .* (y2-y1+1);
	[vals, I] = sort(s);
	pick = s*0;
	counter = 1;
	while ~isempty(I)
		last = length(I);
		i = I(last);
		pick(counter) = i;
		counter = counter + 1;  
		xx1 = max(x1(i), x1(I(1:last-1)));
		yy1 = max(y1(i), y1(I(1:last-1)));
		xx2 = min(x2(i), x2(I(1:last-1)));
		yy2 = min(y2(i), y2(I(1:last-1)));  
		w = max(0.0, xx2-xx1+1);
		h = max(0.0, yy2-yy1+1); 
		inter = w.*h;
		if strcmp(type,'Min')
			o = inter ./ min(area(i),area(I(1:last-1)));
		else
			o = inter ./ (area(i) + area(I(1:last-1)) - inter);
		end
		I = I(find(o<=threshold));
	end
	pick = pick(1:(counter-1));
end

function [dy edy dx edx y ey x ex tmpw tmph] = pad(total_boxes,w,h)
	%compute the padding coordinates (pad the bounding boxes to square)
	tmpw=total_boxes(:,3)-total_boxes(:,1)+1;
	tmph=total_boxes(:,4)-total_boxes(:,2)+1;
	numbox=size(total_boxes,1);
	
    dx=ones(numbox,1);dy=ones(numbox,1);
	edx=tmpw;edy=tmph;
	
	x=total_boxes(:,1);y=total_boxes(:,2);
	ex=total_boxes(:,3);ey=total_boxes(:,4);		
	
	tmp=find(ex>w);
	edx(tmp)=-ex(tmp)+w+tmpw(tmp);ex(tmp)=w;		
	
	tmp=find(ey>h);
	edy(tmp)=-ey(tmp)+h+tmph(tmp);ey(tmp)=h;	
	
	tmp=find(x<1);
	dx(tmp)=2-x(tmp);x(tmp)=1;	
	
	tmp=find(y<1);
	dy(tmp)=2-y(tmp);y(tmp)=1;
end

function [bboxA] = rerec(bboxA)
	%convert bboxA to square
	bboxB=bboxA(:,1:4);
    h=bboxA(:,4)-bboxA(:,2);
	w=bboxA(:,3)-bboxA(:,1);
    l=max([w h]')';
    bboxA(:,1)=bboxA(:,1)+w.*0.5-l.*0.5;
    bboxA(:,2)=bboxA(:,2)+h.*0.5-l.*0.5;
    bboxA(:,3:4)=bboxA(:,1:2)+repmat(l,[1 2]);
end


