%% Path tracing 

basePath = '~/Stanford/SpringQuarter/CS341/specdataimages_gray_512x256_8/';

imageMat = rgb2gray(imread(strcat(basePath,'014809.jpg')));
%imageMat = imcomplement(rgb2gray(im009484));

%imageMat = im014682;
%imageMat = im015834;
%imageMat = im015602;
%imageMat = im015089;

traceMat = imageMat - repmat(median(imageMat,1),size(imageMat,1),1);
%traceMat = imNorm3;
[f_vec1,loss1] = pathTrace(traceMat,0.1);
[f_vec2,loss2] = pathTrace(traceMat,0.5);

figure();imshow(imageMat);
figure();imshow(traceMat);
figure();
imshow(traceMat);
hold on
scatter(f_vec1,1:size(imageMat,1));
scatter(f_vec2,1:size(imageMat,1));
hold off

%% Building some features from extracted line

intensity_vals1 = zeros(length(f_vec1),1);
for i = 1:length(f_vec1)
    intensity_vals1(i) = traceMat(i,f_vec1(i));
end

figure();plot(intensity_vals1);

%% Experimenting with normalization

figure();imshow(imageMat);
hold on
plot(1:size(imageMat,2),size(imageMat,1)-median(imageMat,1));
plot(1:size(imageMat,2),size(imageMat,1)-mean(imageMat,1));
hold off

% Normalizing w.r.t. median value of each column
imNorm1 = imageMat - repmat(median(imageMat,1),size(imageMat,1),1);
figure();imshow(imNorm1);

% Normalizing w.r.t. median value of each row
%imNorm2 = imageMat - repmat(median(imageMat,2),1,size(imageMat,2));
%figure();imshow(imNorm2);

% Using deviation from median value of neighbourhood
%imNorm3 = imageMat - medfilt2(imageMat,[8 16]);
%figure();imshow(imNorm3);

%% 

gaussFilt = filter2(fspecial('gaussian',3),imageMat)/255;

% imNorm2 = imageMat - repmat(uint8(mean(imageMat,1)),size(imageMat,1),1);
% figure();imshow(imNorm2);

% % Some basic stats-based thresholding
% figure();
% imhist(imNorm);
% thresh = quantile(imNorm(:),0.98);
% disp('Thresholding at:') 
% disp(thresh)
% figure();
% binaryIm = imquantize(imNorm,thresh)-1;
% imshow(binaryIm);