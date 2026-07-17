clc 
close all
clear
%生成的训练数据，训练数据设置长度为2000
u = rand(2000,1);
u = u./2;
figure(1);
stem(u);
N = 50;
sequenceLength= length(u);
%将输入信号输入到NARMA中得到输出，其输入信号是随机产生的并且保证它小于0.5（论文中说的）
output_train = zeros(sequenceLength,1);
NRAMA = 10; %10阶NRAMA
for i =NRAMA+1:sequenceLength
    output_train(i,1) = 0.3*output_train(i-1,1)+0.05*output_train(i-1,1)*sum(output_train(i-1-9:i-1-0,1))+0.1+1.5*u(i-1-9)*u(i-1); %NARMA10的生成公式
end
tau=50;%反馈环延迟时间=保持时间（因为这里要的只是一个比值，并且matlab在使用：做引索时要整数
%所以将这里的的反馈延迟由原来的2e-9归一化成50，也可以将其改成2e-9只是matlab会提示警告结果不变）
theta=1;%虚节点间隔θ（做了归一化，由原来的虚节点间隔的4e-11变成了1，为了保证tau/N结果等于50）
for index=1:1:sequenceLength;
    trainInput((index-1)*tau/theta+1:index*tau/theta) = u(index);%将输入信号u抽样保持以便于可以与mask相乘
end
%下面产生mask
m = rand(1,N);
for i =1:N
    if m(i)>0.5
        m(i) = 1;
    else 
        m(i) = -1;
    end
end
mask = m;
for index=1:1:sequenceLength
    Masked((index-1)*tau/theta+1:index*tau/theta) = mask;%每一小段masked持续时间为虚节点间隔theta,相当于将mask自身拓展了2000倍，
end
train_input = trainInput.*Masked; %输入信号加与mask相乘的结果

%生成的训练数据，训练数据设置长度为1000
test = rand(1000,1);
test = test./2;
testLength = length(test);
output_test = zeros(testLength,1);
for i =NRAMA+1:testLength
    output_test(i,1) = 0.3*output_test(i-1,1)+0.05*output_test(i-1,1)*sum(output_test(i-1-9:i-1-0,1))+0.1+1.5*test(i-1-9)*test(i-1); %NARMA10的生成公式   
end
for index=1:1:testLength
   Masked1((index-1)*tau/theta+1:index*tau/theta)=mask;%每一小段Lmask持续时间为虚节点间隔theta：相当于将mask自身拓展了1000倍，
end
for index=1:1:testLength;
    testInput((index-1)*tau/theta+1:index*tau/theta) = test(index);%将输入信号u抽样保持以便于可以与mask相乘
end
test_input = testInput.*Masked1;%输入信号加与mask相乘的结果


%%%%下面是保存信号，以方便注入simulink模块中。
Time1 = 0:4e-11:4e-6-0.01e-9;%在这里每隔40ps注入一个信息进入储备池
Time2 = 0:4e-11:2e-6-0.01e-9;
train_input =train_input';
test_input =test_input';
Time1=Time1';
Time2=Time2';
D1 = [Time1 train_input];
D2 = [Time2 test_input];
figure(2);
stem(output_train,'r');
axis([1,100,-1.5,1.5]);
xlabel('Times')
ylabel('Amplifer')
title('归一化后训练数据');

save trainInputSequence.mat u;
save testInputSequence.mat test;
save trainout.mat output_train;
save testout.mat output_test;
save D1data.mat D1;
save D2data.mat D2;
