clc

image_directory = '../Data/McM_restored_Laplacian';
addpath([pwd, filesep, image_directory])
dinfo = dir(image_directory);
dinfo(ismember( {dinfo.name}, {'.', '..'})) = [];
shape = size(dinfo);
n_files = shape(1);
scores = [];
scores = ones(n_files,1);
for i = 1:n_files
    ps = dinfo(i).name;
    im = imread(ps); 
    [sharpnessScore map]= MLVSharpnessMeasure(im);
    scores(i)=sharpnessScore;
end
disp(scores);