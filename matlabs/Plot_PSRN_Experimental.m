clc;clear;close all;
Pulse = 40;
point = 71;
m = 1;
l = Pulse-60;
step = (10.92e9 - 10.78e9) / (point - 1);
sweepFreq = 10.78*10^9:step:10.92*10^9;     
% 加载数据
fileName0 = sprintf('./PSRN_Experimental_%dns%d_%dp_BGS.h5', Pulse,m,point);
h5disp(fileName0);  % 显示HDF5文件内容
loss_all = h5read(fileName0, '/TrainLoss');
loss_mse = loss_all(2,:);   
loss = loss_all(1,:);
BGS_phy = h5read(fileName0, '/BGS_recon');
BGS_phy = permute(BGS_phy,[3,2,1]);
BGS_raw = h5read(fileName0, '/BGS_raw');
BGS_raw = BGS_raw';
para_pre = h5read(fileName0, '/BGS_pre');
para_pre = permute(squeeze(para_pre),[3,2,1]);

%获取最佳Loss的索引
minValue = min(loss);% 找到最小值
[~, minIndex] = min(loss);% 找到最小值的索引（线性索引）
% 显示结果
fprintf('lossAll最小值是: %d\n', minValue);
fprintf('lossAll最小值的索引是: %d\n', minIndex);
minValue_mse = min(loss_mse);% 找到最小值
[~, minIndex_mse] = min(loss_mse);% 找到最小值的索引（线性索引）
% 显示结果
fprintf('lossMSE最小值是: %d\n', minValue_mse);
fprintf('lossMSE最小值的索引是: %d\n', minIndex_mse);

%获取高空间分辨率的重构BGS
epoch = minIndex;
BFS = squeeze(para_pre(epoch,1,:))';
SW = squeeze(para_pre(epoch,2,:))';
Intensity = squeeze(para_pre(epoch,3,:))';
BFS = (BFS * (10.89 - 10.81) + 10.81) * 1e9;
SW = (SW * (35 - 25) + 25) * 1e6;
Intensity = Intensity * (1 - 0.8) + 0.8;
% 构建 Freq_diff 矩阵
[SWEEP, BFS_MAT] = ndgrid(sweepFreq, BFS);
Freq_diff = SWEEP - BFS_MAT;
% 计算BGS
BGS_recon = Intensity.* (1 ./ (1 + (Freq_diff ./ (SW/2)).^2));

BGS_phy = squeeze(BGS_phy(epoch,:,end+l-500+1:end+l));



figure;
set(gcf,'Units','centimeter','Position',[5 5 8 6]);
semilogy(loss_mse, ...
    'LineWidth', 2, ...      % 线宽
    'Color', [0.9, 0.2, 0.2], ... % RGB颜色（红色）
    'LineStyle', '-' ...     % 实线
);
xlabel('Epoch', 'FontSize', 8);
ylabel('Loss', 'FontSize', 8);
ax = gca;
ax.YTick = [1e-4, 1e-3, 1e-2, 1e-1, 1e0]; % 示例：设置具体刻度值
%设置刻度标签格式（例如科学计数法）
ax.YAxis.TickLabelFormat = '%.0e'; % 显示为 10^{-4}, 10^{-3} 等
ylim([1e-5, 1e0]); % 确保刻度值在范围内


%%
colors = [
    0.00, 0.45, 0.74;   % 深蓝
    0.85, 0.33, 0.10;   % 橙红
    0.47, 0.67, 0.19;   % 橄榄绿
    0.49, 0.18, 0.56;   % 紫罗兰
    0.93, 0.69, 0.13;   % 金色
];
x_start = 4805;
a = 460;
BFS_plot = BFS(1,end-a+1:end);
x = x_start+0.1:0.1:a*0.1+x_start;
Fsize = 12;
% 创建图形窗口
figure;
set(gcf,'Units','centimeter','Position',[5 5 10 6]);
plot(x, BFS_plot/1e9,'Color', colors(2,:),'LineStyle', '-','LineWidth', 1.5);
xlim([4810 4850])
ylim([10.827 10.88])
% xticks(20:10:60);
% yticks(10.835:0.01:10.875);
xlabel(('Fiber length (m)'),'FontName','Arial','FontSize',Fsize);
ylabel(('BFS (GHz)'),'FontName','Arial','FontSize',Fsize);
set(gca,'FontName','Arial','FontSize',Fsize,'Box','on');
set(gca,'looseInset',[0 0 0.01 0.01])
filename = sprintf('./shiyan_%dns%d_BFS_duibi1', Pulse,m);
print(filename, '-dpng', '-r600');


%% BGS
Fsize = 12;
a = 460;
l=60;
x_start = 4805;
x = x_start+0.1:0.1:a*0.1+x_start;
BGS_recon = BGS_recon(:,end-l-a+1:end-l);
BGS_raw = BGS_raw(:,end-l-a+1:end-l);
[X1,Y1] = meshgrid(x,10.78:0.002:10.92);
figure
set(gcf,'Units','centimeter','Position',[5 5 10 6]);
% imagesc(BGS)
surf(X1,Y1,BGS_recon,'EdgeColor','interp','FaceColor','interp')
view(0,90)
colormap(jet);
xlim([4810 4850])
xticks(4810:10:4850);
ylim([10.78 10.92]);
% axis tight
xlabel(('Fiber length (m)'),'FontSize',Fsize);
ylabel(('Frequency (GHz)'),'FontSize',Fsize);
set(gca,'FontName','Arial','FontSize',Fsize);
set(gca,'looseInset',[0 0 0.01 0.01])
% title('BGS recon'); % 添加标题
filename = sprintf('./shiyan_%dns%d_%dp_BGS_recon', Pulse,m, point);
print(filename, '-dpng', '-r600');

figure
set(gcf,'Units','centimeter','Position',[5 5 10 6]);
% imagesc(BGS)
surf(X1,Y1,BGS_raw,'EdgeColor','interp','FaceColor','interp')
view(0,90)
colormap(jet);
xlim([4810 4850])
xticks(4810:10:4850);
ylim([10.78 10.92]);
xlabel(('Fiber length (m)'),'FontSize',Fsize);
ylabel(('Frequency (GHz)'),'FontSize',Fsize);
set(gca,'FontName','Arial','FontSize',Fsize);
set(gca,'looseInset',[0 0 0.01 0.01])
filename = sprintf('./shiyan_%dns%d_%dp_BGS_raw', Pulse,m, point);
print(filename, '-dpng', '-r600');

%% 自监督迭代切片
[X1,Y1] = meshgrid((500.1:0.1:550),10.78:0.002:10.92);
epoch = 2000;
BFS = squeeze(para_pre(epoch,1,:))';
SW = squeeze(para_pre(epoch,2,:))';
Intensity = squeeze(para_pre(epoch,3,:))';
BFS = (BFS * (10.89 - 10.81) + 10.81) * 1e9;
SW = (SW * (35 - 25) + 25) * 1e6;
Intensity = Intensity * (1 - 0.8) + 0.8;
% 构建 Freq_diff 矩阵
[SWEEP, BFS_MAT] = ndgrid(sweepFreq, BFS);
Freq_diff = SWEEP - BFS_MAT;
% 计算BGS
BGS_epoch = Intensity.* (1 ./ (1 + (Freq_diff ./ (SW/2)).^2));
BGS_epoch = BGS_epoch(:,301:800);

figure
set(gcf,'Units','centimeter','Position',[5 5 10 6]);
% imagesc(BGS)
surf(X1,Y1,BGS_epoch,'EdgeColor','interp','FaceColor','interp')
view(0,90)
colormap(jet);
xticks(500:10:550);
ylim([10.78 10.92]);
% axis tight
xlabel(('Fiber length (m)'),'FontSize',8,'FontName','Arial');
ylabel(('Frequency (GHz)'),'FontSize',8,'FontName','Arial');
set(gca,'FontName','Arial','FontSize',8,'FontName','Arial');
set(gca,'looseInset',[0 0 0.01 0.01])
filename = sprintf('./shiyan_%dns_%dp_BGS_recon_%depoch', Pulse, point,epoch);
print(filename, '-dpng', '-r600');

%% BFS自监督

Fsize = 14;
BFS_all = squeeze(para_pre(:,1,:));
BFS_all = (BFS_all * (10.89 - 10.81) + 10.81) * 1e9;
BFS_all = BFS_all(:,end-a+1:end);
x_start = 4810;
x = x_start+0.1:0.1:a*0.1+x_start;
% 创建图形窗口
figure;
set(gcf,'Units','centimeter','Position',[5 5 10 8]);
hold on; 

% ===== 自定义部分 =====
selected_rows = [1,5,10,100,1000,1926];
legend_labels = {'Ep 1','Ep 5','Ep 10','Ep 100','Ep 1000','Ep 2000'}; % 自定义标签

colors = [
    0.000, 0.447, 0.741;   % 蓝色 (蓝)
    0.850, 0.325, 0.098;   % 红橙色 (红)
    0.929, 0.694, 0.125;   % 黄色 (金黄)
    0.494, 0.184, 0.556;   % 紫色 (紫)
    0.466, 0.674, 0.188;   % 绿色 (绿)
    0.301, 0.745, 0.933    % 青色 (浅蓝)
];

for idx = 1:length(selected_rows)
    i = selected_rows(idx);
    if i > size(BFS_all,1)
        warning('跳过不存在行: %d', i);
        continue;
    end
    % 判断是否为最后一条曲线（Epoch 2000）
    if idx == length(selected_rows)
        % 最后一条曲线使用虚线
        plot(x, BFS_all(i,:)/1e9,...
            'Color', colors(idx,:),...
            'DisplayName', legend_labels{idx},...
            'LineWidth', 1.2,...
            'LineStyle', '-.');  % 添加虚线样式
    else
        % 其他曲线使用实线
        plot(x, BFS_all(i,:)/1e9,...
            'Color', colors(idx,:),...
            'DisplayName', legend_labels{idx},...
            'LineWidth', 1.2);
    end
end
% =====================
% 显式调用 box on 确保生效
box on;
% 坐标轴设置
ylim([10.80 10.9]);
xlim([x_start x_start+40]);
% 标签样式
xlabel('Fiber length (m)', 'FontName','Arial', 'FontSize',Fsize);
ylabel('BFS (GHz)', 'FontName','Arial', 'FontSize',Fsize);

% 专业级图例设置（关键修改部分）
hLegend = legend('show','Location', 'north','FontSize', Fsize,'Box', 'off','NumColumns', 3);
hLegend.ItemTokenSize = [10,20];
% 设置单位为归一化，便于调整位置
set(hLegend, 'Units', 'normalized');
% 获取当前位置并向上移动（调整pos(2)值）
pos = get(hLegend, 'Position');
deltaY = 0.03; % 向上移动距离（按图形高度比例，0.03≈3%）
deltaX = 0.04;
set(hLegend, 'Position', [pos(1)+deltaX pos(2)+deltaY pos(3) pos(4)]);

set(gca,'FontName','Arial','FontSize',Fsize);
set(gca,'looseInset',[0 0 0.01 0.25])
% 输出图像
print('./shiyan_40ns_BFS_self','-dpng','-r600');