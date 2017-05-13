function T = binArray(array,binFactor,dim)
% Bins an array horizontal/vertical by averaging binFactor columns 
arraySize = size(array);
newLen = round(arraySize(dim)/binFactor);
newSize = zeros(1,2);
T = zeros(arraySize(1),newLen);
for i=1:newLen
    T(:,i) = mean(array(:,(binFactor*(i-1)+1):binFactor*i),2);
end