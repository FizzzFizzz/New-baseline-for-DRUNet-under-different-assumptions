clear all;
close all;

inputpath = './trainsets/DIV2K_train_HR/';
outputpath= './trainsets/DIV2K_cut8/';

img_path_list = dir(strcat(inputpath,'*.png'));

for i = 1 : length(img_path_list)
    i
    name = img_path_list(i).name;
    I = double(imread([inputpath name]));
    [m,n,~] = size(I);
    cutm = floor(m/8).*8;
    cutn = floor(n/8).*8; f = [];
    f(1:cutm,1:cutn,:) = I(1:cutm,1:cutn,:);
    imwrite(uint8(f),[outputpath name]);    
end
