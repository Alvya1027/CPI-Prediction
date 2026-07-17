clc
clear
%*****************  Train  *****************************
load('response1.mat');
response = ScopeData(:,2);
response1 = response(100001:200000);%在我的储备池中为了让反馈系统先稳定所以光注入的时候加入了延迟让其稳定后在注入
%一共得到100000个数据点（N*Ntrain）
NTrain=2000; %第一次信号中包含了2000个训练信号
nForgetPoints = 100; 
N=50;
for i=1:NTrain
    TrainStateMatrix(i,:)=response1((i-1)*N+1:i*N);
end
%将原始储备池得到的一维的信号转换成内部的状态节点一个（2000*50）的矩阵也就是储备池的Xi(n)
TrainStateMatrixM=TrainStateMatrix(nForgetPoints+1:end,:);%忽略掉一部分可能未稳定的点
load('trainout.mat'); %载入训练信号的标签
outputWeights = (pinv(TrainStateMatrixM)*output_train(nForgetPoints+1:end,:))';
%下面计算训练信号的NRMSE
%训练误差是整个储备池的上限，也就是说储备池最好能达到的结果
for i=1:NTrain-nForgetPoints
    predictedTrainOutput(i)=sum(outputWeights.*TrainStateMatrixM(i,:));  
end
predictedTrainOutput = predictedTrainOutput';
trainError=compute_NRMSE(predictedTrainOutput,output_train(nForgetPoints+1:end))
%画出部分的结果
figure(1);
stem(predictedTrainOutput(101:200),'g');
hold on
stem(output_train(201:300),'r');
hold on
legend('pred','trainoutput')
save outputWeights.mat outputWeights;
