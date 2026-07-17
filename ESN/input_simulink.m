clc
clear
load('D1data.mat');
t = 0:4e-11:4e-6;
y = D1(:,2);
y = 0.004*y;%经过多次测试，注入的y值越大，仿真误差越大，噪声越多
ss = length(y);
t = t(1,1:ss)';
simin = [t,y];%simulink要求以simin的格式注入其包含注入时间和内容