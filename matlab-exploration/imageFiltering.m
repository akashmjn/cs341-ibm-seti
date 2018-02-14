%% Image pre-processing

basePath = '~/Stanford/SpringQuarter/CS341/cs341-ibm-seti/matlab-exploration/';
cd(basePath)

% Colour images
% Noisy Narrowband drd (curve) signal
im003294 = imread(strcat(basePath,'p0c2_003294.jpg'));
% Noisy Straight line signal
im002024 = imread(strcat(basePath,'p0c1_002024.jpg'));
% Very faint Straight line signal
im012543 = imread(strcat(basePath,'p0c1_012543.jpg'));
% Clearer Straight line signal
im009484 = imread(strcat(basePath,'p0c1_009484.jpg'));
% Clearer narrowbanddrd signal
imnarrowbdrd = imread(strcat(basePath,'narrowbanddrd_0.jpg'));

% Grayscale images
% Clear, partially included curve
im015089 = rgb2gray(imread(strcat(basePath,'p0c1_015089.jpg')));
% Pulsed, moderately clear line
im014682 = rgb2gray(imread(strcat(basePath,'p1c0_014682.jpg')));
% Slightly visible line 
im015834 = rgb2gray(imread(strcat(basePath,'p1c0_015834.jpg')));
% Moderately clear squiggle
im015602 = rgb2gray(imread(strcat(basePath,'p1c2_015602.jpg')));

%% rough space
colIm = im012543;
greysImRaw = imcomplement(rgb2gray(im012543)); % Inverting the conversion to white-high
avgFilt = filter2(fspecial('average',3),greysImRaw)/255;
medFilt = medfilt2(greysImRaw);

% Normalizing column-wise between 0-1
imNorm = double(greysImRaw);
imNorm = imNorm - repmat(min(imNorm,[],1),length(imNorm(:,1)),1);
imNorm = imNorm./repmat(max(imNorm,[],1),length(imNorm(:,1)),1);
figure();imshow(imNorm,[])

% Trying a PCA reduction with columns as features
[U,S,V] = svd(imNorm,0);

% figure()
% for i=125:140
%     ncomp = i;
%     imTransform = U(:,1:ncomp)*S(1:ncomp,1:ncomp)*V(:,1:ncomp)';
%     pause(0.2)
%     imshow(imTransform)
% end
ncomp = 2;
imTransform = U(:,1:ncomp)*S(1:ncomp,1:ncomp)*V(:,1:ncomp)';

greysIm = imNorm-imTransform;
% Some basic stats-based thresholding
figure();
imhist(greysIm);
thresh = quantile(greysIm(:),0.9);
disp('Thresholding at:') 
disp(thresh)
figure();
binaryIm = imquantize(greysIm,thresh)-1;
imshow(binaryIm);


%% Other rough stuff

% Filtering normalized image
avgFiltNorm = filter2(fspecial('gaussian',[3 3],0.5),imNorm);
% NLM filter on column-normalized data
opts = struct('kernelratio',3,'windowratio',10,'filterstrength',0.2);
nlmFilt = NLMF(imNorm,opts);
figure();imshow(nlmFilt,[])

% Edge detection
[BW,thresh] = edge(binaryIm,'sobel');
[BW,thresh] = edge(binaryIm,'canny');
[BW,thresh] = edge(greysImRaw,'log');
% Trying out non-local means filtering
% NLM filter
opts = struct('kernelratio',10,'windowratio',3,'filterstrength',0.05);
nlmFilt = NLMF(double(greysImRaw)/255,opts);
imshow(nlmFilt)


% Subtracting row-wise average and normalizing between 0-1
imNormR = double(greysImRaw) - mean(greysImRaw,2)*ones(1,length(greysImRaw(1,:)));
imNormR = imNormR - min(imNormR(:));
imNormR = imNormR./max(imNormR(:));

opts = struct('kernelratio',3,'windowratio',10,'filterstrength',0.05);
nlmFiltBinary = NLMF(double(binaryIm),opts);
imshow(nlmFiltBinary)

%% rough - trying some autocorrelation stuff
greysImRaw = imcomplement(rgb2gray(im012543));

imDiffR = abs(diff(greysImRaw,1,1));
imDiffC = abs(diff(greysImRaw,1,2));
figure(); imshow(imDiffR,[])
figure(); imshow(imDiffC,[])

figure(); 
for i=1:128
    plot(greysImRaw(i,:))
    pause(0.05)
end

n = 4;
horzMAFilt = repmat([0 1/n 0],n,1);
avgFilt = filter2(horzMAFilt,greysImRaw)/255;
figure(); imshow(avgFilt,[])

%% Trying some basic filtering / thresholding

greysImRaw = imcomplement(rgb2gray(colIm)); % Inverting the conversion to white-high
% Subtracting column-wise average and normalizing between 0-1
imNormC = double(greysImRaw) - ones(length(greysImRaw(:,1)),1)*mean(greysImRaw,1);
imNormC = imNormC - min(imNormC(:));
imNormC = imNormC./max(imNormC(:));
% Subtracting row-wise average and normalizing between 0-1
imNormR = double(imNormC) - mean(imNormC,2)*ones(1,length(imNormC(1,:)));
imNormR = imNormR - min(imNormR(:));
imNormR = imNormR./max(imNormR(:));
%greysIm = binArray(greysIm,2);
avgFilt = filter2(fspecial('gaussian',[5 5],0.6),greysImRaw)/255;
medFilt = medfilt2(greysImRaw);
% Horizontal average filter
n = 4;
MAfilter = repmat([0 1/n 0],n,1);
horzMAFilt = filter2(MAfilter,greysImRaw)/255;
% non-local means filter
opts = struct('kernelratio',3,'windowratio',10,'filterstrength',0.05);
nlmFilt = NLMF(imNormR,opts);
%nlmFilt = NLMF(double(greysImRaw)/255,opts);
greysIm = greysImRaw;
%figure(); imshowpair(double(greysImRaw)/255,greysIm)
figure(); imshow(double(greysImRaw)/255,[])
% Some basic stats-based thresholding
figure();
imhist(greysIm);
thresh = quantile(greysIm(:),0.9);
disp('Thresholding at:') 
disp(thresh)
figure();
binaryIm = imquantize(greysIm,thresh)-1;
imshow(binaryIm,[]);

%% Hough transform
aspectAngle = floor(90-atand(size(binaryIm,1)/size(binaryIm,2))-1);
[H,theta,rho] = hough(binaryIm,'Theta',-aspectAngle:0.5:aspectAngle);
% finding peaks
P = houghpeaks(H,5);
%P = houghpeaks(H,1,'threshold',ceil(0.3*max(H(:))));
x = theta(P(:,2));
y = rho(P(:,1));
lines = houghlines(binaryIm,theta,rho,P);
%lines = houghlines(BW,theta,rho,P,'FillGap',5,'MinLength',7);

%% ploting hough transform
figure
imshow(imadjust(mat2gray(H)),[],...
       'XData',theta,...
       'YData',rho,...
       'InitialMagnification','fit');
xlabel('\theta (degrees)')
ylabel('\rho')
axis on
axis normal 
hold on
colormap(gca,hot)
plot(x,y,'s','color','black');

%% plotting marked out lines
figure, imshow(binaryIm), hold on
max_len = 0;
for k = 1:length(lines)
   xy = [lines(k).point1; lines(k).point2];
   plot(xy(:,1),xy(:,2),'LineWidth',2,'Color','green');

   % Plot beginnings and ends of lines
   plot(xy(1,1),xy(1,2),'x','LineWidth',2,'Color','yellow');
   plot(xy(2,1),xy(2,2),'x','LineWidth',2,'Color','red');

   % Determine the endpoints of the longest line segment
   len = norm(lines(k).point1 - lines(k).point2);
   if ( len > max_len)
      max_len = len;
      xy_long = xy;
   end
end
% highlight the longest line segment
plot(xy_long(:,1),xy_long(:,2),'LineWidth',2,'Color','red');

