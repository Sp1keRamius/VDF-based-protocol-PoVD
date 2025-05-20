from matplotlib import mathtext
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import re
from datetime import datetime

def print_fig_to_paper(output_type, plot_file_name, fig_font_size=16, fig_font_name='Times New Roman', fig_width=7, 
                       is_print=True, is_print_time=True, fig=None, fig_height_custom=None, is_hide_axis=False):
    '''
    Adjust the format of the figure to be suitable for the journal paper.
    output_type options in matplotlib: 'eps', 'pdf', 'png'
    fig_font_size: font size  ## NOTE: 50% insert into lyx, then the fontsize in the journal paper will be 8pt, close to the fontsize of the text in the journal paper
    fig_font_name: the name of the font
    fig_width: Width of the figure
    is_print: Whether to save the figure
    is_print_time: Whether to add the current time to the file name
    fig: The figure to be saved, if None, the current figure (plt.gcf()) will be saved
    figu_height_custom: Custom height of the figure
    is_hide_axis: Whether to hide the axis
    '''
    numdip = 300

    fig = fig or plt.gcf()
    
    # Set font properties
    # plt.rcParams.update({'font.size': fig_font_size, 'font.family': fig_font_name})
    
    # Use xelatex to render the text, including the mathemtical symbols
    # plt.rcParams['text.usetex'] = True
    # plt.rcParams['pgf.texsystem'] = 'xelatex'
    # plt.rcParams['pgf.rcfonts'] = False
    # plt.rcParams['pgf.preamble'] = '\n'.join([
    #     r'\usepackage{fontspec}',
    #     r'\setmainfont{Times New Roman}'])
    # mpl.use("pgf")
    
    # set the default font of mathtext to fig_font_name
    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['mathtext.it'] = fig_font_name + ':italic'
    
    # Get current figure dimensions
    current_fig_width, current_fig_height = fig.get_size_inches()
    
    # Calculate figure height maintaining the aspect ratio
    fig_height = fig_width / current_fig_width * current_fig_height
    
    # If custom height is specified
    fig_height = fig_height_custom or fig_height
    
    # Set figure size
    fig.set_size_inches(fig_width, fig_height)
    
    if is_hide_axis:
        ax = plt.gca()
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
    else:
        ax = plt.gca()
        ax.set_facecolor('none') # clear the background color of the axis
    
    for text in fig.findobj(match=plt.Text):
        # set the size and font of the text, specify custom math font
        text.set_fontsize(fig_font_size)
        text.set_fontname(fig_font_name)
        text.set_math_fontfamily('custom')
        # if pattern.search(text.get_text()): # if the text contains math symbols
        #     text.set_usetex(True)

    fig.tight_layout()
    
    # Ensure the 'pic' directory exists
    if not os.path.exists('pic'):
        os.makedirs('pic')

    # Save the figure
    if is_print:
        if is_print_time:
            plot_file_name = f"pic/{plot_file_name}{datetime.now().strftime('%Y%m%dT%H%M%S')}.{output_type}"
        else:
            plot_file_name = f"pic/{plot_file_name}.{output_type}"
        
        fig.savefig(plot_file_name, format=output_type, dpi=numdip, bbox_inches='tight')

# Example usage
if __name__ == "__main__":
    import numpy as np
    
    # Example plot
    plt.rc('font', family='Times New Roman')

# 数据
    categories = ['0000F', '0001F', '0002F', '0003F', '0004F']
    group1 = [(0.000236495, 0.003897738), (0.000472957, 0.007778499), (0.000709388, 0.011642353), (0.000945785, 0.015489369), (0.00118215, 0.019319616)]
    group2 = [(0.003998763, 3.33E-15), (0.007956838, 4.00E-15), (0.011877193, 1.22E-15), (0.015759095, 1.11E-15), (0.019594206, 6.66E-16)]

    # 将数据转换为NumPy数组便于操作
    group1 = np.array(group1)
    group2 = np.array(group2)

    # 计算比例
    group1_total = group1[:, 0] + group1[:, 1]
    group2_total = group2[:, 0] + group2[:, 1]

    # 类别标签的位置
    ind = np.arange(len(categories))  # 类别的x坐标
    width = 0.15  # 柱的宽度

    # 创建图表
    fig, ax = plt.subplots()

    # 学术风格颜色
    colors_pow = ['#A9D073', '#9BC6BC']  # 浅蓝和深蓝，用于PoW
    colors_vdf = [ '#F6A57A', '#F6A57A']  # 浅橙和深橙，用于VDF




    # 绘制group1 (PoW)的柱形图
    rects1 = ax.bar(ind - width, group1[:, 0], width, label='PoW concurrent fork', color=colors_pow[0])
    rects1b = ax.bar(ind - width, group1[:, 1], width, bottom=group1[:, 0], label='PoW propagation fork', color=colors_pow[1])

    # 绘制group2 (VDF)的柱形图
    rects2 = ax.bar(ind, group2[:, 0], width, label='PoVD concurrent fork', color=colors_vdf[0])
    # rects2b = ax.bar(ind, group2[:, 1], width, bottom=group2[:, 0], label='PoVD propagation fork', color=colors_vdf[1])

    # 添加标签、标题和自定义x轴标签
    ax.set_xlabel('Mining Target', fontsize=14)
    ax.set_ylabel('Fork rate', fontsize=14)
    ax.set_xticks(ind)
    ax.set_xticklabels(categories, fontsize=12)
    ax.legend(loc='upper left', fontsize=12)


    plt.plot(ind - width, group1_total, color='#328E6E', linestyle='-', linewidth=1, marker = 'o')

    # 添加VDF连线

    plt.plot(ind, group2_total, color='#FF6666', linestyle='--', linewidth=1, marker = 'x')
    # 添加网格线
    ax.grid(True, which='both', axis='y', linestyle='--', linewidth=0.7, alpha=0.7)

    # 调整布局
    fig.tight_layout()

    # 保存图像，符合IEEE风格
    print_fig_to_paper(output_type='eps', plot_file_name='bar', 
                    fig_font_size=16, fig_font_name='Times New Roman', 
                    fig_width=7, is_print=True, is_print_time=False)

    # 另存为png格式
    print_fig_to_paper(output_type='png', plot_file_name='bar', 
                    fig_font_size=16, fig_font_name='Times New Roman', 
                    fig_width=7, is_print=True, is_print_time=False)

