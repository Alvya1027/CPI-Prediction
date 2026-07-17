clc
clear
t = 0:4e-11:2e-6;
load('D2data.mat');
y = D2(:,2);
y = 0.004*y;
ss = length(y);
t = t(1,1:ss)';
simin = [t,y];
