
clc

ps='../Data/McM/1.tif';
im = imread(ps); 

figure(1);imagesc(im);title(i);colormap gray; drawnow;
        
[sharpnessScore map]= MLVSharpnessMeasure(im);
sharpnessScore

figure(2);imagesc(map);title('s2'); colormap gray; drawnow;





