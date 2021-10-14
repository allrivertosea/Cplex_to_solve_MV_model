function h1 = two_d_plotx_y(X1,Y1)
figure
h1 = scatter(subs(X1),subs(Y1),40,'*','r');
legend('最优边界PF')
xlabel('收益')
ylabel('风险')
xticks(0:0.002:0.01)
yticks(0:0.001:0.004)
%ax = gca;
%ax.PlotBoxAspectRatio = [1 0.5 0.5]
%set(gca,'XDir','reverse')
%xlim([0 0.06])
%ylim([0 0.013])
%zlim([0 0.15])
pbaspect([1 1 1])
%view(60,20)
saveas(h1,'F:\答辩\程序演示及答疑\论文程序演示\基于多目标进化算法的投资组合优化研究\Platypus-master\data_process_code\经典模型最优解\最优前沿','jpg')