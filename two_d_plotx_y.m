function h1 = two_d_plotx_y(X1,Y1)
figure
h1 = scatter(subs(X1),subs(Y1),40,'*','r');
legend('���ű߽�PF')
xlabel('����')
ylabel('����')
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
saveas(h1,'F:\���\������ʾ������\���ĳ�����ʾ\���ڶ�Ŀ������㷨��Ͷ������Ż��о�\Platypus-master\data_process_code\����ģ�����Ž�\����ǰ��','jpg')