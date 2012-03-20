clear
databasePath = 'G:\research\SIFTflow\SIFTflow_original_code\demo';

dirs=dir([databasePath '\*.png']);   % 用你需要的目录以及文件扩展名替换。读取某个目录的指定类型文件列表，返回结构数组。
dircell=struct2cell(dirs)' ;    % 结构体(struct)转换成元胞类型(cell)，转置一下是让文件名按列排列。
filenames=dircell(:,1);   % 第一列是文件名

file_path = [databasePath '\'];
cd(file_path);

cachedir = 'results/';
mkdir(cachedir);

fid = fopen('result.txt','w');

[rows,cols] = size(filenames);

% init color wheel
colorWheel = makeColorwheel();

% init
addpath(fullfile(pwd,'mexDenseSIFT'));
addpath(fullfile(pwd,'mexDiscreteFlow'));

% init sift feature 
cellsize=3;
gridspacing=1;

% init sift flow params
SIFTflowpara.alpha=2*255;
SIFTflowpara.d=40*255;
SIFTflowpara.gamma=0.005*255;
SIFTflowpara.nlevels=4;
SIFTflowpara.wsize=2;
SIFTflowpara.topwsize=10;
SIFTflowpara.nTopIterations = 60;
SIFTflowpara.nIterations= 30;

% init the first image
fileName = char(filenames(1,1));
im1 = imread(fileName);
im1 = preprocessingImg(im1);
sift1 = mexDenseSIFT(im1,cellsize,gridspacing);

% loop
for i =2:rows
    fileName = char(filenames(i,1));
    fprintf(fid,'frame: %d ...\n',i);
    
    %get the next sift vimage
    im2 = imread(fileName);
    im2 = preprocessingImg(im2);
    sift2 = mexDenseSIFT(im2,cellsize,gridspacing);

    
    tic;[vx,vy,energylist]=SIFTflowc2f(sift1,sift2,SIFTflowpara);toc
    
     image = zeros(size(vx));  
     
     [height width] = size(vx);
     
     for j = 1:height
         for i = 1:width
             u = vx(j,i);
             v = vy(j,i);
             newX = min(max(1,i+u),width);
             newY = min(max(1,j+v),height);
             pixel = im1(newY,newX,:);
             image(j,i) = pixel;
         end
     end
     
     
%     rad = floor(sqrt(vx.^2+vy.^2));
%     minrad = min(rad(:));
%     maxrad = max(rad(:));
%     
%    
%     image = ones([size(vx),3],'uint8');  
%     image = image.*255;
%     ccrow = size(colorWheel,1)-1;
%     
%     dis = maxrad - minrad;
%     
%     a = (ccrow - 1)/(maxrad - minrad);
%     b = 1 - a*minrad;
%     
%     for radClass = minrad:maxrad
%         [xxrow, xxcol] = find(rad == radClass);
%         for i = 1:size(xxrow,1)
%             image(xxrow(i,1),xxcol(i,1),:) = colorWheel(floor(a*radClass + b)+ 1,:); 
%         end
%         
%         
%     end
%     
    
     
%   clear flow;
% flow(:,:,1)=vx;
% flow(:,:,2)=vy;
% a = zeros(1,1);
% figure;imshow(flowToColor(flow,a));
    
    
%     ouput the running time
%      fprintf(fid,'time: %s\n',num2str(toc));
%    
%      
%      
% %     get affine model
%     [height,width] = size(vx);
%     blockHalfW = 5;
%     blockHalfH = 5;
%     
%     errors = [];
%     params = [];
%     locations = [];
%     
%     
% %     构建struct X ，Y，vx，vy
%     X = ones(height,1)*(1:width);
%     
%     Y = zeros(size(X));
%     for i = 1:width
%         Y(:,i) = 1:height;
%     end
%  
%     s = struct('u',num2cell(vx),'v',num2cell(vy),'x',num2cell(X),'y',num2cell(Y),'class',num2cell(zeros(height,width)));
%     
% 
%     for j = 1 + blockHalfH : blockHalfH*2: height - blockHalfH
%         for i = 1 + blockHalfW :blockHalfW*2: width - blockHalfW
%             
% %             use the function : Vx(X,Y) = Axo + Axx(X) + Axy(Y),
% %                               Vy(X,Y) = Ayo+ Ayx(X) + Ayy(Y)
% 
% %             get Vx(X,Y)
%             vx_xy = vx( j - blockHalfH : j + blockHalfH , i - blockHalfW : i + blockHalfW);
%             vx_xy = vx_xy(:);
%             
% %             get Vy(X,Y)
%             vy_xy = vy( j - blockHalfH : j + blockHalfH , i - blockHalfW : i + blockHalfW);
%             vy_xy = vy_xy(:);
%          
% %             get Y
%             y = zeros(2*blockHalfH+1, 2*blockHalfW+1);
%             for yi = 1:2*blockHalfW+1;
%                 y(:,yi) = (j - blockHalfH : 1 : j + blockHalfH)';
%             end
%             y = y(:);
%          
%             % get X
%              x = zeros(2*blockHalfH+1, 2*blockHalfW+1);
%               for xi = 1:2*blockHalfH+1;
%                 x(xi,:) = (i - blockHalfW :1: i + blockHalfW);
%               end
%              x = x(:);          
%              [errorX,params1] = leastSquare(vx_xy,x,y);
%              [errorY,params2] = leastSquare(vy_xy,x,y);
%              
% %              X = [X x'];
% %              Y = [Y y'];
%              
%              param = [params1 params2];
%              params = [params param(:)];
%              errors = [errors [errorX,errorY]']; 
%              locations = [locations [j,i]'];
%              
%         end
%     end
%     
%     locations = locations';
%     params = params';   
% 
% %      iso data
% %  predictGroupNum = 3;
%      [centro, A, clustering] = provaisodata(params, params(1,:),100);
%      
% %    [clustering,centro] = kmeans(params, predictGroupNum);
%   
%     K = [8 3];
%      
%     for loop = 1:2
%         
%         params = [];
%         params_x = centro(:,1:3);
%         params_y = centro(:,4:6);
%         
%         % 遍历每一个点，看最适合的对应的param类是哪个,改变s中每一个像素最适合的类
%         for j = 1:height
%             for i = 1:width
%                 pointLocation = [1 i j]';
%                     candidate_u = params_x * pointLocation;
%                     candidate_v = params_y * pointLocation;
%                     
%                     current_u = vx(j,i);
%                     current_v  = vy(j,i);
%                     
%                     dist1 = abs(current_u^2 - candidate_u.^2);
%                     dist2 = abs(current_v^2 - candidate_v.^2);
%                     
%                     dist = dist1.^2 + dist2.^2;
%                     
%                     [m,indice] = min(dist);           
%                     s(j,i).class = indice; 
%             end
%         end
%               
% %         answers就是最终每一个像素点属于哪一类的一个说明
% %          根据answers找之前建的struct
% %          用find遍历每一类，找到所属的对应点，构建矩阵，用leastsquare
%         errors = [];
%         for class = 1:A(1,1)
%             criteria = @(e)(e.class == class);
%             [crow,ccol] = find(arrayfun(criteria, s));
%             
%             xs = zeros(size(crow));
%             ys = zeros(size(crow));
%             us = zeros(size(crow));
%             vs = zeros(size(crow));
%           
%             for i = 1:size(crow,1)
%                 xs(i,1) = s(crow(i,1),ccol(i,1)).x;
%                 ys(i,1) = s(crow(i,1),ccol(i,1)).y;
%                 us(i,1) = s(crow(i,1),ccol(i,1)).u;
%                 vs(i,1) = s(crow(i,1),ccol(i,1)).v;
%             end
%             [errorX,params1] = leastSquare(us,xs,ys);
%             [errorY,params2] = leastSquare(vs,xs,ys);
%             errors = [errors [errorX,errorY]']; 
%             
%              param = [params1 params2];
%              params = [params param(:)];       
%         end
%         
%             params = params';
% %         再次使用要缩小聚类中心的数目
%            [centro, A, clustering] = provaisodata(params, params(1,:),K(1,loop));
%     end
%     
%     
% %     记住当前一共有几类
%      predictGroupNum = A(1,1);
%      
%      image = ones([size(vx),3],'uint8');  
%     image = image.*255;
%      ccrow = size(colorWheel,1);
%      
%       params = [];
%         params_x = centro(:,1:3);
%         params_y = centro(:,4:6);
%      
%     for j = 1:height
%         for i = 1:width
%             pointLocation = [1 i j]';
%                 candidate_u = params_x * pointLocation;
%                 candidate_v = params_y * pointLocation;
% 
%                 current_u = vx(j,i);
%                 current_v  = vy(j,i);
% 
%                 dist1 = abs(current_u^2 - candidate_u.^2);
%                 dist2 = abs(current_v^2 - candidate_v.^2);
% 
%                 dist = dist1.^2 + dist2.^2;
% 
%                 [m,indice] = min(dist);           
%                 s(j,i).class = indice; 
%                 
%                 image(j,i,:) = colorWheel(floor(ccrow*indice/predictGroupNum),:);
%             
%         end
%     end
%     
%     
% %     give color
% 
%     
%    
%     
%     
%     
    
    
    
% %     ccc = [1 100 200];
%     
%     predictGroupNum = A(1,1);
%    for groupIndex = 1:predictGroupNum
%       [grows, gcols]=find(clustering == groupIndex);
%       relatedLoc = locations(gcols,:);
%       
%       for color = 1:3
%           result = ones(2*blockHalfH+1, 2*blockHalfW+1);
%            result = result.*colorWheel(floor(ccrow*groupIndex/predictGroupNum),color);
% %            result = result.*ccc(1,groupIndex);
%           for t = 1:size(relatedLoc,1)
%              image([relatedLoc(t,1)- blockHalfH : relatedLoc(t,1)+ blockHalfH],[relatedLoc(t,2) - blockHalfW : relatedLoc(t,2) + blockHalfW ],color) = result;
%           end  
%       end
%    end
%    
% %    figure;imshow(image);    
% %     array = struct('location',locations,'affineModelParams',clustering);
    
    
    
    clear flow;
    clear result;
    flow(:,:,1)=vx;
    flow(:,:,2)=vy;
    result = flowToColor(flow,fid);
    
%     the last sift equals to the next image
    im1 = im2;
    sift1 = sift2;
    
%     write result to given directory
    cd(cachedir);
%     imwrite(result,fileName,'JPG');
    imwrite(image,['reConstruct' fileName],'JPG');
    cd('../');
end


a = 1;
cd('../');
fprintf(fid,'successd\n');
fclose(fid);

