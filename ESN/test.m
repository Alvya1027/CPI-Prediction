clear
clc
NMSE=zeros(1,1);
NTest = 1000;  %测试信号选取的是1000个信号点
nForgetPoints = 100; 
N=50;
load('response2.mat');
response2 = ScopeData(:,2);
response2 = response2(100001:150000);%由于是1000个所以进过储备池后得到的数据量就减少了50000个（N*Ntest）
%下面将一维系统转换成（Ntest*N）的内部节点矩阵
for i=1:NTest
testStateMatrix(i,:)=response2((i-1)*N+1:i*N);
end
testStateMatrixM=testStateMatrix(nForgetPoints+1:end,:);
load('testout.mat');%载入测试信号标签，用于计算测试误差
load('outputWeights.mat');
for i=1:NTest-nForgetPoints
    predictedtestOutput(i)=sum(outputWeights.*testStateMatrixM(i,:));
end
predictedtestOutput = predictedtestOutput';
testError=compute_NRMSE(predictedtestOutput,output_test(nForgetPoints+1:end))%计算NRMSE得到的结果
%画出预测的信号和原始的标签信号
figure(2);
stem(predictedtestOutput(401:500),'g');
hold on
stem(output_test(501:600),'r');
xlabel('dian');
ylabel('out');
legend(' pred','testOutput')