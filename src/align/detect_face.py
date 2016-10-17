""" Hej
"""
import numpy as np
import cv2
from math import floor
#from scipy import misc
import scipy.io as sio

def detect_face(img, minsize, pnet, rnet, onet, threshold, factor):
    # im: input image
    # minsize: minimum of faces' size
    # pnet, rnet, onet: caffemodel
    # threshold: threshold=[th1 th2 th3], th1-3 are three steps's threshold
    # fastresize: resize img from last scale (using in high-resolution images) if fastresize==true
    factor_count=0
    total_boxes=np.empty((0,9))
    points=[]
    h=img.shape[0]
    w=img.shape[1]
    minl=np.amin([h, w])
    #img=single(img);
#     if fastresize
#         im_data=(single(img)-127.5)*0.0078125;
#     end
    m=12.0/minsize
    minl=minl*m
    # creat scale pyramid
    scales=[]
    while minl>=12:
        scales += [m*np.power(factor, factor_count)]
        minl = minl*factor
        factor_count += 1

    # first stage
    #for j = 1:size(scales,2)
    for j in range(len(scales)):
        scale=scales[j]
        hs=int(np.ceil(h*scale))
        ws=int(np.ceil(w*scale))
        #im_data=(imResample(img,[hs ws],'bilinear')-127.5)*0.0078125;
        #im_data = (misc.imresize(img, (hs, ws), interp='bilinear')-127.5)*0.0078125
        #im_data = (cv2.resize(img, (hs, ws), interpolation=cv2.INTER_LINEAR)-127.5)*0.0078125
        #im_data = (cv2.resize(img, (hs, ws), interpolation=cv2.INTER_NEAREST)-127.5)*0.0078125
        #im_data = (img[0:hs,0:ws,:]-127.5)*0.0078125
        im_data = imResample2(img, (hs, ws), 'bilinear')
        im_data = (im_data-127.5)*0.0078125
        
        # PNet.blobs('data').reshape([hs ws 3 1]);
        # out=PNet.forward({im_data});

        img_x = np.expand_dims(im_data, 0)
        img_y = np.transpose(img_x, (0,2,1,3))
        out = pnet(img_y)
        out0 = np.transpose(out[0], (0,2,1,3))
        #out0 = out[0]
        out1 = np.transpose(out[1], (0,2,1,3))
        #out1 = out[1]
        
        #boxes=generateBoundingBox(out{2}(:,:,2), out{1}, scale, threshold[0]);
        boxes, reg = generateBoundingBox(out1[0,:,:,1].copy(), out0[0,:,:,:].copy(), scale, threshold[0])
        
        # inter-scale nms
        pick = nms(boxes.copy(), 0.5, 'Union')
        boxes = boxes[pick,:]  # boxes=boxes(pick,:);
        if boxes.size>0: # if ~isempty(boxes)
            total_boxes = np.append(total_boxes, boxes, axis=0)  # total_boxes=[total_boxes;boxes];

    numbox = total_boxes.shape[0]  # numbox=size(total_boxes,1);
    if numbox>0:  #   if ~isempty(total_boxes)
        pick = nms(total_boxes.copy(), 0.7, 'Union')  #     pick=nms(total_boxes,0.7,'Union');
        total_boxes = total_boxes[pick,:] #     total_boxes=total_boxes(pick,:);
        regw = total_boxes[:,2]-total_boxes[:,0] #     regw=total_boxes(:,3)-total_boxes(:,1);
        regh = total_boxes[:,3]-total_boxes[:,1] #     regh=total_boxes(:,4)-total_boxes(:,2);
        # total_boxes=[total_boxes(:,1)+total_boxes(:,6).*regw   total_boxes(:,2)+total_boxes(:,7).*regh   total_boxes(:,3)+total_boxes(:,8).*regw   total_boxes(:,4)+total_boxes(:,9).*regh   total_boxes(:,5)];
        qq1 = total_boxes[:,0]+total_boxes[:,5]*regw
        qq2 = total_boxes[:,1]+total_boxes[:,6]*regh
        qq3 = total_boxes[:,2]+total_boxes[:,7]*regw
        qq4 = total_boxes[:,3]+total_boxes[:,8]*regh
        total_boxes = np.transpose(np.vstack([qq1, qq2, qq3, qq4, total_boxes[:,4]]))
        total_boxes = rerec(total_boxes.copy()) # total_boxes=rerec(total_boxes);
        total_boxes[:,0:4] = np.fix(total_boxes[:,0:4])  # total_boxes(:,1:4)=fix(total_boxes(:,1:4));
        dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph = pad(total_boxes.copy(), w, h)

    numbox = total_boxes.shape[0] # numbox=size(total_boxes,1);
    if numbox>0:
        # second stage
        tempimg = np.zeros((24,24,3,numbox)) # tempimg=zeros(24,24,3,numbox);
        for k in range(0,numbox):  # for k=1:numbox
            tmp = np.zeros((tmph[k],tmpw[k],3))  #       tmp=zeros(tmph(k),tmpw(k),3);
            tmp[dy[k]-1:edy[k],dx[k]-1:edx[k],:] = img[y[k]-1:ey[k],x[k]-1:ex[k],:]  # tmp(dy(k):edy(k),dx(k):edx(k),:)=img(y(k):ey(k),x(k):ex(k),:);
            if tmp.shape[0]>0 and tmp.shape[1]>0 or tmp.shape[0]==0 and tmp.shape[1]==0:  # if size(tmp,1)>0 && size(tmp,2)>0 || size(tmp,1)==0 && size(tmp,2)==0
                tempimg[:,:,:,k] = imResample2(tmp, (24, 24), 'bilinear')  #                 tempimg(:,:,:,k)=imResample(tmp,[24 24],'bilinear');
            else:
                return np.empty()  #  total_boxes = []; return;
        tempimg = (tempimg-127.5)*0.0078125 #         tempimg=(tempimg-127.5)*0.0078125;
        # RNet.blobs('data').reshape([24 24 3 numbox]);
        # out=RNet.forward({tempimg});
        tempimg1 = np.transpose(tempimg, (3,1,0,2))
        out = rnet(tempimg1)
        out0 = np.transpose(out[0])
        out1 = np.transpose(out[1])
        score = out1[1,:]  # score=squeeze(out{2}(2,:));
        ipass = np.where(score>threshold[1]) # pass=find(score>threshold(2));
        total_boxes = np.hstack([total_boxes[ipass[0],0:4].copy(), np.expand_dims(score[ipass].copy(),1)])   # total_boxes=[total_boxes(pass,1:4) score(pass)'];
        mv = out0[:,ipass[0]]  # mv=out{1}(:,pass);
        if total_boxes.shape[0]>0:  #     if size(total_boxes,1)>0    
            pick = nms(total_boxes, 0.7, 'Union')  # pick=nms(total_boxes,0.7,'Union');
            total_boxes = total_boxes[pick,:]  # total_boxes=total_boxes(pick,:);     
            total_boxes = bbreg(total_boxes.copy(), np.transpose(mv[:,pick])) # total_boxes=bbreg(total_boxes,mv(:,pick)');  
            total_boxes = rerec(total_boxes.copy()) # total_boxes=rerec(total_boxes);

    numbox = total_boxes.shape[0]  #     numbox=size(total_boxes,1);
    if numbox>0:
        # third stage
        total_boxes=np.fix(total_boxes)  # total_boxes=fix(total_boxes);
        dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph = pad(total_boxes.copy(), w, h) # [dy edy dx edx y ey x ex tmpw tmph]=pad(total_boxes,w,h);
        tempimg = np.zeros((48,48,3,numbox))  # tempimg=zeros(48,48,3,numbox);
        for k in range(0,numbox):  # for k=1:numbox
            tmp = np.zeros((tmph[k],tmpw[k],3))#         tmp=zeros(tmph(k),tmpw(k),3);
            tmp[dy[k]-1:edy[k],dx[k]-1:edx[k],:] = img[y[k]-1:ey[k],x[k]-1:ex[k],:]          # tmp(dy(k):edy(k),dx(k):edx(k),:)=img(y(k):ey(k),x(k):ex(k),:);
            if tmp.shape[0]>0 and tmp.shape[1]>0 or tmp.shape[0]==0 and tmp.shape[1]==0:  # if size(tmp,1)>0 && size(tmp,2)>0 || size(tmp,1)==0 && size(tmp,2)==0
                tempimg[:,:,:,k] = imResample2(tmp, (48, 48), 'bilinear')  # tempimg(:,:,:,k)=imResample(tmp,[48 48],'bilinear');
            else:
                return np.empty()  #  total_boxes = []; return;
        tempimg = (tempimg-127.5)*0.0078125 #         tempimg=(tempimg-127.5)*0.0078125;
        tempimg1 = np.transpose(tempimg, (3,1,0,2))
        out = onet(tempimg1)  # out=ONet.forward({tempimg});
        out0 = np.transpose(out[0])
        out1 = np.transpose(out[1])
        out2 = np.transpose(out[2])
        score = out2[1,:]  # score=squeeze(out{3}(2,:));
        points = out1 # points=out{2};
        ipass = np.where(score>threshold[2]) #       pass=find(score>threshold(3));
        points = points[:,ipass[0]] # points=points(:,pass);
        total_boxes = np.hstack([total_boxes[ipass[0],0:4].copy(), np.expand_dims(score[ipass].copy(),1)])   # total_boxes=[total_boxes(pass,1:4) score(pass)'];
        mv = out0[:,ipass[0]]  # mv=out{1}(:,pass);

        w = total_boxes[:,2]-total_boxes[:,0]+1 # w=total_boxes(:,3)-total_boxes(:,1)+1;
        h = total_boxes[:,3]-total_boxes[:,1]+1   # h=total_boxes(:,4)-total_boxes(:,2)+1;
        points[0:5,:] = np.tile(w,(5, 1))*points[0:5,:] + np.tile(total_boxes[:,0],(5, 1))-1   # points(1:5,:)=repmat(w',[5 1]).*points(1:5,:)+repmat(total_boxes(:,1)',[5 1])-1;
        points[5:10,:] = np.tile(h,(5, 1))*points[5:10,:] + np.tile(total_boxes[:,1],(5, 1))-1 # points(6:10,:)=repmat(h',[5 1]).*points(6:10,:)+repmat(total_boxes(:,2)',[5 1])-1;
        if total_boxes.shape[0]>0:  #     if size(total_boxes,1)>0    
            total_boxes = bbreg(total_boxes.copy(), np.transpose(mv))  # total_boxes=bbreg(total_boxes,mv(:,:)');
            pick = nms(total_boxes.copy(), 0.7, 'Min')  # pick=nms(total_boxes,0.7,'Min');
            total_boxes = total_boxes[pick,:]  # total_boxes=total_boxes(pick,:);
            points = points[:,pick]  # points=points(:,pick);
                
    return total_boxes, points
            
 
# function [boundingbox] = bbreg(boundingbox,reg)
def bbreg(boundingbox,reg):
    # calibrate bouding boxes
    if reg.shape[1]==1:  # if size(reg,2)==1
        reg = np.reshape(reg, (reg.shape[2], reg.shape[3]))  #     reg=reshape(reg,[size(reg,3) size(reg,4)])';

    w = boundingbox[:,2]-boundingbox[:,0]+1  #   w=[boundingbox(:,3)-boundingbox(:,1)]+1;
    h = boundingbox[:,3]-boundingbox[:,1]+1  #   h=[boundingbox(:,4)-boundingbox(:,2)]+1;
    b1 = boundingbox[:,0]+reg[:,0]*w
    b2 = boundingbox[:,1]+reg[:,1]*h
    b3 = boundingbox[:,2]+reg[:,2]*w
    b4 = boundingbox[:,3]+reg[:,3]*h
    boundingbox[:,0:4] = np.transpose(np.vstack([b1, b2, b3, b4 ])) # boundingbox(:,1:4)=[boundingbox(:,1)+reg(:,1).*w boundingbox(:,2)+reg(:,2).*h boundingbox(:,3)+reg(:,3).*w boundingbox(:,4)+reg(:,4).*h];
    return boundingbox
 
def generateBoundingBox(map, reg, scale, t):
    # use heatmap to generate bounding boxes
    stride=2
    cellsize=12

    map = np.transpose(map)  #     map=map';
    dx1 = np.transpose(reg[:,:,0]) # dx1=reg(:,:,1)';
    dy1 = np.transpose(reg[:,:,1]) # dy1=reg(:,:,2)';
    dx2 = np.transpose(reg[:,:,2]) # dx2=reg(:,:,3)';
    dy2 = np.transpose(reg[:,:,3]) # dy2=reg(:,:,4)';
    y, x = np.where(map > t)  # [y x]=find(map>=t);
    if y.shape[0]==1:  # if size(y,1)==1
      # *** not checked ***
#         y=y';x=x';score=map(a)';dx1=dx1';dy1=dy1';dx2=dx2';dy2=dy2';
        y = np.transpose(y)
        x = np.transpose(x)
        dx1 = np.transpose(dx1)
        dy1 = np.transpose(dy1)
        dx2 = np.transpose(dx2)
        dy2 = np.transpose(dy2)
        score = map[(y,x)]
    else:
        score = map[(y,x)]

    reg = np.transpose(np.vstack([ dx1[(y,x)], dy1[(y,x)], dx2[(y,x)], dy2[(y,x)] ]))  #     reg=[dx1(a) dy1(a) dx2(a) dy2(a)];
    if reg.size==0:   #     if isempty(reg)
        # *** not checked ***
        reg = np.empty((0,3))    # reg=reshape([],[0 3]);
    bb = np.transpose(np.vstack([y,x]))  #     boundingbox=[y x];
    q1 = np.fix((stride*bb+1)/scale)
    q2 = np.fix((stride*bb+cellsize-1+1)/scale)
    boundingbox = np.hstack([q1, q2, np.expand_dims(score,1), reg])  # boundingbox=[fix((stride*(boundingbox-1)+1)/scale) fix((stride*(boundingbox-1)+cellsize-1+1)/scale) score reg];
    return boundingbox, reg
 
# function pick = nms(boxes,threshold,type)
def nms(boxes,threshold,type):
    if boxes.size==0:  # if isempty(boxes)
        return np.empty((0,3))   ###### check dimension  #######
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    s = boxes[:,4]
    area = (x2-x1+1) * (y2-y1+1)
    I = np.argsort(s)  # [vals, I] = sort(s);
    #vals = s[I]
    pick = np.zeros_like(s, dtype=np.int16)   # s*0;
    counter = 0
    while I.size>0:   # ~isempty(I)
        #last = I.size
        i = I[-1]
        pick[counter] = i
        counter += 1
        idx = I[0:-1]
        xx1 = np.maximum(x1[i], x1[idx]) # xx1 = max(x1(i), x1(I(1:last-1)));
        yy1 = np.maximum(y1[i], y1[idx]) # yy1 = max(y1(i), y1(I(1:last-1)));
        xx2 = np.minimum(x2[i], x2[idx]) # xx2 = min(x2(i), x2(I(1:last-1)));
        yy2 = np.minimum(y2[i], y2[idx]) # yy2 = min(y2(i), y2(I(1:last-1)));  
        w = np.maximum(0.0, xx2-xx1+1)   # w = max(0.0, xx2-xx1+1);
        h = np.maximum(0.0, yy2-yy1+1)   # h = max(0.0, yy2-yy1+1); 
        inter = w * h #     inter = w.*h;
        if type is 'Min':  #     if strcmp(type,'Min')
            o = inter / np.minimum(area[i], area[idx])  # o = inter ./ min(area(i),area(I(1:last-1)));
        else:
            o = inter / (area[i] + area[idx] - inter)   # o = inter ./ (area(i) + area(I(1:last-1)) - inter);
        I = I[np.where(o<=threshold)]  #     I = I(find(o<=threshold));
    pick = pick[0:counter]  #   pick = pick(1:(counter-1));
    return pick

# function [dy edy dx edx y ey x ex tmpw tmph] = pad(total_boxes,w,h)
def pad(total_boxes, w, h):
    # compute the padding coordinates (pad the bounding boxes to square)
    tmpw = total_boxes[:,2]-total_boxes[:,0]+1  # tmpw=total_boxes(:,3)-total_boxes(:,1)+1;
    tmph = total_boxes[:,3]-total_boxes[:,1]+1 # tmph=total_boxes(:,4)-total_boxes(:,2)+1;
    numbox=total_boxes.shape[0]

    dx = np.ones((numbox,1))   # dx=ones(numbox,1);
    dy = np.ones((numbox,1))   # dy=ones(numbox,1);
    edx=tmpw.copy()
    edy=tmph.copy()

    x = total_boxes[:,0].copy()  # x=total_boxes(:,1);
    y = total_boxes[:,1].copy()  # y=total_boxes(:,2);
    ex = total_boxes[:,2].copy() # ex=total_boxes(:,3);
    ey = total_boxes[:,3].copy()  # ey=total_boxes(:,4);    

    tmp = np.where(ex>w)  #   tmp=find(ex>w);
    edx[tmp] = np.expand_dims(-ex[tmp]+w+tmpw[tmp],1)  # edx(tmp)=-ex(tmp)+w+tmpw(tmp);ex(tmp)=w;
    ex[tmp] = w
    
    tmp = np.where(ey>h) # tmp=find(ey>h);
    edy[tmp] = np.expand_dims(-ey[tmp]+h+tmph[tmp],1) #   edy(tmp)=-ey(tmp)+h+tmph(tmp);ey(tmp)=h;  
    ey[tmp] = h

    tmp = np.where(x<1) #   tmp=find(x<1);
    dx[tmp] = np.expand_dims(2-x[tmp],1)    #   dx(tmp)=2-x(tmp);  
    x[tmp] = 1

    tmp = np.where(y<1) #   tmp=find(y<1);
    dy[tmp] = np.expand_dims(2-y[tmp],1) #   dy(tmp)=2-y(tmp);y(tmp)=1;
    y[tmp] = 1
    
    return dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph

# function [bboxA] = rerec(bboxA)
def rerec(bboxA):
    # convert bboxA to square
    #bboxB = bboxA[:,0:4]  #   bboxB=bboxA(:,1:4);
    h = bboxA[:,3]-bboxA[:,1] #   h=bboxA(:,4)-bboxA(:,2);
    w = bboxA[:,2]-bboxA[:,0] #   w=bboxA(:,3)-bboxA(:,1);
    l = np.maximum(w, h) # l=max([w h]')';
    bboxA[:,0] = bboxA[:,0]+w*0.5-l*0.5 # bboxA(:,1)=bboxA(:,1)+w.*0.5-l.*0.5;
    bboxA[:,1] = bboxA[:,1]+h*0.5-l*0.5 # bboxA(:,2)=bboxA(:,2)+h.*0.5-l.*0.5;
    bboxA[:,2:4] = bboxA[:,0:2] + np.transpose(np.tile(l,(2,1))) # bboxA(:,3:4)=bboxA(:,1:2)+repmat(l,[1 2]);
    return bboxA

def imResample2(img, sz, method):
    h=img.shape[0]
    w=img.shape[1]
    hs, ws = sz
    dx = float(w) / ws
    dy = float(h) / hs
    im_data = np.zeros((hs,ws,3))
    for a1 in range(0,hs):
        for a2 in range(0,ws):
            for a3 in range(0,3):
                im_data[a1,a2,a3] = img[floor(a1*dy),floor(a2*dx),a3]
    return im_data
