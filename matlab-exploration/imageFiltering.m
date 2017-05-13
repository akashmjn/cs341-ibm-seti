%% Image pre-processing

cd('~/Stanford/SpringQuarter/CS341/matlab-exploration/')

% Noisy Narrowband drd (curve) signal
im003294 = imread('~/Stanford/SpringQuarter/CS341/matlab-exploration/p0c2_003294.jpg');
% Noisy Straight line signal
im002024 = imread('~/Stanford/SpringQuarter/CS341/matlab-exploration/p0c1_002024.jpg');
% Very faint Straight line signal
im012543 = imread('~/Stanford/SpringQuarter/CS341/matlab-exploration/p0c1_012543.jpg');
% Clearer Straight line signal
im009484 = imread('~/Stanford/SpringQuarter/CS341/matlab-exploration/p0c1_009484.jpg');

%% rough space

g002024 = imcomplement(rgb2gray(im002024)); % Inverting the conversion to white-high
avg002024 = filter2(fspecial('average',3),g002024)/255;
med002024 = medfilt2(g002024);
% Thresholding
figure();
imhist(g002024);
figure();
b002024 = imquantize(g002024,100)-1;
imshow(b002024);
% Edge detection
[BW,thresh] = edge(b002024,'sobel');
[BW,thresh] = edge(b002024,'canny');
[BW,thresh] = edge(g002024,'log');
% Trying out non-local means filtering
% NLM filter
opts = struct('kernelratio',10,'windowratio',3,'filterstrength',0.05);
nlmFilt = NLMF(double(greysImRaw)/255,opts);
imshow(nlmFilt)

% Subtracting column-wise average and normalizing between 0-1
imNorm = double(greysImRaw) - ones(length(greysImRaw(:,1)),1)*mean(greysImRaw,1);
imNorm = imNorm - min(imNorm(:));
imNorm = imNorm./max(imNorm(:));
% NLM filter on column-normalized data
opts = struct('kernelratio',3,'windowratio',10,'filterstrength',0.05);
nlmFilt = NLMF(imNorm,opts);
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

%% Trying some basic thresholding / filtering + hough transform

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
greysIm = horzMAFilt;
figure(); imshowpair(double(greysImRaw)/255,greysIm)
% Some basic stats-based thresholding
figure();
imhist(greysIm);
thresh = quantile(greysIm(:),0.9);
disp('Thresholding at:') 
disp(thresh)
figure();
binaryIm = imquantize(greysIm,thresh)-1;
imshow(binaryIm);

%% Hough transform
[H,theta,rho] = hough(binaryIm,'Theta',-80:0.5:80);
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

