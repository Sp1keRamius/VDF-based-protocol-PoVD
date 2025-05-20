from matplotlib import mathtext
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import re
from datetime import datetime

def Fp(p, n, alpha = 5):
    part1 = (n / p) * (1 - (1 - p)**(1/n)) * (1 - p)**((n - 1) / n)
    return 1 - part1 * (1 - p)**alpha
def Fv(p, n, L):
    p_L = (L * p) / (1 + L * p - p)
    part2 = (n / p_L) * (1 - (1 - p_L)**(1/n)) * (1 - p_L)**((n - 1) / n)
    return 1 - part2


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
    from scipy.stats import norm


    
    plt.rc('font', family='Times New Roman')

    
    x = np.linspace(1, 32, 32)
    x = [x[i] for i in range(1, 33, 6)]

    y1 = [14.0885, 31.2433, 47.8855, 55.72, 76.406, 95.9796, 110.2941, 127.6305, 143.2993, 159.7026, 174.1704, 191.7489, 206.2996, 217.6805, 236.5706, 255.3597, 271.1029, 280.7538, 303.8533, 312.428, 335.2954, 340.646, 367.9693, 383.726, 399.1863, 415.7319, 431.8835, 445.9095, 463.9536, 479.8507, 494.2437, 507.6355]
    y1 = [y1[i] for i in range(1, 33, 6)]
    y1_n = [34.3701, 128.4209, 221.7035, 321.4454, 412.3526, 513.6308]
    
    y1_0 = [15.253, 15.9984, 15.871, 15.9448, 15.8709, 15.827, 15.9132, 15.1639, 15.9989, 15.9576, 15.9958, 15.9217, 15.9963, 15.9845, 15.9846, 15.9591, 14.9684, 15.8324, 15.9623, 15.8273, 14.7656, 15.9943, 15.8293, 15.8937, 15.9897, 15.9797, 15.9705, 15.858, 15.8685, 15.8087, 15.9003, 15.9797]
    y1_0 = [y1_0[i] for i in range(1, 33, 6)]

    y2 = [62.2667, 251.3057, 436.8875, 639.6823, 829.7663, 1023.8665]
    y2_0 = [31.6481, 31.4433, 31.9844, 30.7901, 31.5362, 27.4249]
    y2_n = [58.6187, 254.4046, 447.2405, 644.7127, 829.0778, 1020.2352]


    y3 = [124.3953, 499.9475, 879.1964, 1267.5953, 1657.2985, 2047.803]
    y3_0 = [62.4889, 63.7386, 63.8954, 63.7935, 63.8846, 66.8203]
    y3_n = [126.6492, 509.5943, 895.5245, 1280.3333, 1669.1346, 2046.6804]




    # 创建图表
    # fig, ax = plt.subplots()

    # 绘制每一组曲线
    # plt.plot([-5],[-5], linestyle='-', color='grey', label='PoW')
    # plt.plot([-5],[-5],  linestyle='-.', color='grey', label='VDF')
    plt.plot([-5],[31.9994], color='g', linestyle = '--',  label='PoW')
    plt.plot([-5],[63.7693], color='b', linestyle = 'dotted', label='VDF Baseline')
    plt.plot([-5],[14.0885], color='r', label='PoVD')

    plt.plot(x, y1, linestyle='--', marker = '^', color='g')
    plt.plot(x, y1_0,  linestyle='-', marker = '^', color='r')
    plt.plot(x,y1_n,  linestyle='dotted', marker = '^', color='blue')

    plt.plot(x, y2, linestyle='--', marker = 's', color='g')
    plt.plot(x, y2_0, linestyle='-', marker = 's', color='r')
    plt.plot(x,y2_n,  linestyle='dotted', marker = 's', color='b')

    plt.plot(x, y3, linestyle='--', marker = 'x', color='g')
    plt.plot(x, y3_0, linestyle='-', marker = 'x', color='r')
    plt.plot(x,y3_n,  linestyle='dotted', marker = 'x', color='b')


    plt.xlim(0,33)
    # plt.ylim(0,2000)
    # 设置对数纵坐标
    plt.yscale('log')

    # 设置标签和图例
    plt.xlabel('Number of processors')
    plt.ylabel('Computational complexity')
    plt.legend(loc='upper left')
    plt.grid()
    
    # Call the function with your parameters
    print_fig_to_paper(output_type='png', plot_file_name='Computation complexity', 
                       fig_font_size=16, fig_font_name='Times New Roman', 
                       fig_width=7, is_print=True, is_print_time=False,
                       )
    print_fig_to_paper(output_type='eps', plot_file_name='Computation complexity', 
                       fig_font_size=16, fig_font_name='Times New Roman', 
                       fig_width=7, is_print=True, is_print_time=False,
                       )
    print_fig_to_paper(output_type='svg', plot_file_name='Computation complexity', 
                       fig_font_size=16, fig_font_name='Times New Roman', 
                       fig_width=7, is_print=True, is_print_time=False,
                       )
    print_fig_to_paper(output_type='svg', plot_file_name='Computation complexity', 
                       fig_font_size=16, fig_font_name='Times New Roman', 
                       fig_width=7, is_print=True, is_print_time=False,
                       )

