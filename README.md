\# CPI-prediction



本项目面向 CPI 月度数据的小样本预测问题，计划依次实现传统基线模型、普通储备池模型、光储备池仿真模型和孪生光储备池模型。



\## 当前阶段



第 1 阶段：数据整理与基础预测框架搭建。



\## 仓库结构



\- data\_raw：原始 CPI 数据

\- data\_processed：清洗后的数据和训练集

\- notebooks：实验 notebook

\- src：可复用 Python 代码

\- matlab：光储备池 Matlab 仿真代码

\- results：实验结果、图片、表格

\- docs：项目文档、文献笔记、周报



\## 成员分工



\- A：项目管理与实验框架

\- B：数据收集与预处理

\- C：基线模型	

\- D：储备池与孪生网络预研



\## 注意事项

大家把 CPI-prediction 仓库 clone 到自己电脑任意位置都可以，不需要统一放在 C 盘或 D 盘。



但有两个要求：



不要改仓库内部文件夹名字，比如 data\_raw、data\_processed、docs、notebooks、results、src 这些保持一致。

代码里不要写自己电脑的绝对路径，比如 C:/Users/xxx/Desktop/...，统一通过 src/config.py 读取路径。



比如读取 CPI 数据时，不要写：



df = pd.read\_csv("C:/Users/xxx/Desktop/CPI-prediction/data\_processed/cpi\_monthly.csv")



而要写：



from src.config import DATA\_PROCESSED\_DIR

df = pd.read\_csv(DATA\_PROCESSED\_DIR / "cpi\_monthly.csv")

