clc;clear;close all;
Pulse = 40; % 这个可以是任何你想要的数字
point = 71;
l = 500;
step = (10.92e9 - 10.78e9) / (point - 1);
sweepFreq = 10.78*10^9:step:10.92*10^9;     
% 加载数据
fileName0 = sprintf('./PSRN_Simulation_%dns_%dp_BGS.h5', Pulse,point);
h5disp(fileName0);  % 显示HDF5文件内容
para_GT = h5read(fileName0, '/BGS_GT');
para_GT = para_GT';
loss_all = h5read(fileName0, '/TrainLoss');
loss_mse = loss_all(2,:);   
loss = loss_all(1,:);
BGS_phy = h5read(fileName0, '/BGS_recon');
BGS_phy = permute(BGS_phy,[3,2,1]);
BGS_raw = h5read(fileName0, '/BGS_raw');
BGS_raw = BGS_raw';
BGS_raw = BGS_raw(:,1:l);
para_pre = h5read(fileName0, '/BGS_pre');
para_pre = permute(squeeze(para_pre),[3,2,1]);
BGS_noNoise = h5read(fileName0, '/BGS_clean');
BGS_noNoise = BGS_noNoise';
BGS_noNoise =  BGS_noNoise/max(BGS_noNoise(:));

%获取高空间分辨率BGS真值
BFS_GT = (para_GT(1,:) * (10.89 - 10.81) + 10.81) * 1e9;
SW_GT = (para_GT(2,:) * (35 - 25) + 25) * 1e6;
Intensity_GT =  para_GT(3,:) * (1 - 0.8) + 0.8;
[SWEEP, BFS_MAT] = ndgrid(sweepFreq, BFS_GT);
Freq_diff = SWEEP - BFS_MAT;
% 计算BGS真值
BGS_GT = Intensity_GT.* (1 ./ (1 + (Freq_diff ./ (SW_GT/2)).^2));
BGS_GT = BGS_GT(:,1:l);
%获取最佳Loss的索引
minValue = min(loss);% 找到最小值
[~, minIndex] = min(loss);% 找到最小值的索引（线性索引）
% 显示结果
fprintf('lossAll最小值是: %d\n', minValue);
fprintf('lossAll最小值的索引是: %d\n', minIndex);

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
BGS_recon = BGS_recon(:,1:l);
BGS_phy = squeeze(BGS_phy(epoch,:,1:l));

diff_BFS = abs(BFS(1:l)-BFS_GT(1:l));
mean_BFS = mean(diff_BFS);
disp(['MAE值为: ', num2str(mean_BFS, '%.4e')])
% figure;
% plot(BFS(1:l))
% hold on
% plot(BFS_GT(1:l))

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
% SSIM曲线
ssim_value = ssim(BGS_GT, BGS_recon);
% 显示结果
disp(['SSIM值为: ', num2str(ssim_value)]);


%% BGS

Fsize = 12;
[X1,Y1] = meshgrid((500.1:0.1:550),10.78:0.002:10.92);
figure
set(gcf,'Units','centimeter','Position',[5 5 10 6]);
% imagesc(BGS)
surf(X1,Y1,BGS_recon,'EdgeColor','interp','FaceColor','interp')
view(0,90)
colormap(jet);
xticks(500:10:550);
ylim([10.78 10.92]);
% axis tight
xlabel(('Fiber length (m)'),'FontSize',Fsize);
ylabel(('Frequency (GHz)'),'FontSize',Fsize);
set(gca,'FontName','Arial','FontSize',Fsize);
set(gca,'looseInset',[0 0 0.01 0.01])
% title('BGS recon'); % 添加标题
% axis off; % 去掉坐标轴
filename = sprintf('./PhysenNet/fangzhen_%dns_%dp_BGS_recon', Pulse, point);
print(filename, '-dpng', '-r600');

figure
set(gcf,'Units','centimeter','Position',[5 5 10 6]);
% imagesc(BGS)
surf(X1,Y1,BGS_phy,'EdgeColor','interp','FaceColor','interp')
view(0,90)
colormap(jet);
xticks(500:10:550);
ylim([10.78 10.92]);
xlabel(('Fiber length (m)'),'FontSize',Fsize);
ylabel(('Frequency (GHz)'),'FontSize',Fsize);
set(gca,'FontName','Arial','FontSize',Fsize);
set(gca,'looseInset',[0 0 0.01 0.01])

filename = sprintf('./PhysenNet/fangzhen_%dns_%dp_BGS_phy', Pulse, point);
print(filename, '-dpng', '-r600');


figure
set(gcf,'Units','centimeter','Position',[5 5 10 6]);
% imagesc(BGS)
surf(X1,Y1,BGS_raw,'EdgeColor','interp','FaceColor','interp')
view(0,90)
colormap(jet);
xticks(500:10:550);
ylim([10.78 10.92]);
xlabel(('Fiber length (m)'),'FontSize',Fsize);
ylabel(('Frequency (GHz)'),'FontSize',Fsize);
set(gca,'FontName','Arial','FontSize',Fsize);
set(gca,'looseInset',[0 0 0.01 0.01])
% axis off; % 去掉坐标轴
filename = sprintf('./PhysenNet/fangzhen_%dns_%dp_BGS_raw', Pulse, point);
print(filename, '-dpng', '-r600');


figure
set(gcf,'Units','centimeter','Position',[5 5 10 6]);
% imagesc(BGS)
surf(X1,Y1,BGS_GT,'EdgeColor','interp','FaceColor','interp')
view(0,90)
colormap(jet);
xticks(500:10:550);
ylim([10.78 10.92]);
xlabel(('Fiber length (m)'),'FontSize',Fsize);
ylabel(('Frequency (GHz)'),'FontSize',Fsize);
set(gca,'FontName','Arial','FontSize',Fsize);
set(gca,'looseInset',[0 0 0.01 0.01])

filename = sprintf('./PhysenNet/fangzhen_%dns_%dp_BGS_GT', Pulse, point);
print(filename, '-dpng', '-r600');

%% BFS
colors = [
    0.00, 0.45, 0.74;   % 深蓝
    0.85, 0.33, 0.10;   % 橙红
    0.47, 0.67, 0.19;   % 橄榄绿
    0.49, 0.18, 0.56;   % 紫罗兰
    0.93, 0.69, 0.13;   % 金色
];
x = 500.1:0.1:550; % 自动适配列数

Fsize = 12;
% 创建图形窗口
figure;
set(gcf,'Units','centimeter','Position',[5 5 10 6]);
hold on; 
% grid on;
plot(x, BFS_GT(1,1:l)/1e9,'Color', 'k','LineStyle', '-','LineWidth', 1.5);
plot(x, BFS(1,1:l)/1e9,'Color', colors(2,:),'LineStyle', '-.','LineWidth', 1.5);
ylim([10.78 10.92]);
leg = legend('Ground truth BFS','PSRN');
leg.ItemTokenSize = [15,20];
set(legend,'FontName','Arial','FontSize',Fsize,'location','northwest','Orientation','horizontal','NumColumns', 2,'box','off');
xlabel('Fiber length (m)','FontSize',Fsize);
ylabel('BFS (GHz)','FontSize',Fsize);
set(gca,'FontName','Arial','FontSize',Fsize,'Box','on','looseInset',[0 0 0.01 0.01]);
filename = sprintf('./PhysenNet/fangzhen_%dns_%dp_BFS_duibi1', Pulse, point);
print(filename, '-dpng', '-r600');



%% 自监督迭代切片
[X1,Y1] = meshgrid((500.1:0.1:550),10.78:0.002:10.92);
epoch = 1000;%epoch=1,5,10,100,1000,2000
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
BGS_epoch = BGS_epoch(:,1:l);
Fsize = 10;
figure
set(gcf,'Units','centimeter','Position',[5 5 10 6]);
% imagesc(BGS)
surf(X1,Y1,BGS_epoch,'EdgeColor','interp','FaceColor','interp')
view(0,90)
colormap(jet);
xticks(510:20:550);
ylim([10.78 10.92]);
% axis tight
xlabel(('Fiber length (m)'),'FontSize',Fsize);
ylabel(('Frequency (GHz)'),'FontSize',Fsize);
set(gca,'FontName','Arial','FontSize',Fsize);
set(gca,'looseInset',[0 0 0.01 0.01])
axis off; % 去掉坐标轴
filename = sprintf('./PhysenNet/fangzhen_%dns_%dp_BGS_recon_%depoch', Pulse, point,epoch);
print(filename, '-dpng', '-r600');


%% BFS自监督
% 生成x轴数据
Fsize = 12;
BFS_all = squeeze(para_pre(:,1,:));
BFS_all = (BFS_all * (10.89 - 10.81) + 10.81) * 1e9;
x = 500.1:0.1:550;
% 创建图形窗口
figure;
set(gcf,'Units','centimeter','Position',[5 5 10 8]);
hold on; 

% ===== 自定义部分 =====
selected_rows = [1,5,10,100,1000,2000];
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
        plot(x, BFS_all(i,1:l)/1e9,...
            'Color', colors(idx,:),...
            'DisplayName', legend_labels{idx},...
            'LineWidth', 1.2,...
            'LineStyle', '-.');  % 添加虚线样式
    else
        % 其他曲线使用实线
        plot(x, BFS_all(i,1:l)/1e9,...
            'Color', colors(idx,:),...
            'DisplayName', legend_labels{idx},...
            'LineWidth', 1.2);
    end
end
% =====================
% 显式调用 box on 确保生效
box on;
% 坐标轴设置
xticks(500:10:550);
ylim([10.80 10.92]);
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
print('./PhysenNet/fangzhen_40ns_BFS_self','-dpng','-r600');

