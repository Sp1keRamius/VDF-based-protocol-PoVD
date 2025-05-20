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
    

    L = 10
    p_values = np.linspace(0.0001, 0.2, 500)
    n_values = [10, 50, 100, 1000000]

    colorlist = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    # def compute_function(p, n, L):
    #     p_L = (L * p) / (1 + L * p - p)
    #     term1 = n / p_L * (1 - (1 - p_L) ** (1 / n))
    #     term2 = (1 - p_L) ** ((n - 1) / n)
    #     result = np.log(term1 * term2) / np.log(1-p)
    #     return 1-result/L - 1/L + 1/L/n
    
    def compute_function(p, n, L):
        term1 = (1+L*p-p)**(1/n)
        term2 = (1-p)**(1/n)

        result = (np.log((term1 - term2)/(1 - term2)/L)) / np.log(1-p)
        return 1-result/L-1/L+(1/L/n if n <= 1000 else 0)


    for i,n in enumerate(n_values):
        y_values = compute_function(p_values, n, L)
        plt.semilogx([1/p for p in p_values], y_values, label=f'n = {n}', color = colorlist[i])
    plt.plot([1/p for p in p_values], [0.5-1/L/2]*len(p_values), label=f'n = {n}', linestyle='-.', color = 'grey')

    # theta = np.linspace(0, 2 * np.pi, 100)
    # a, b = 0.003, 0.03  # 椭圆的长轴和短轴
    # x = a * np.cos(theta) + 0.075
    # y = b * np.sin(theta) + 0.76
    # plt.plot(x, y, color = 'grey')

    # end = [-0.015, 0.015]
    # start = [0.07, 0.50]
    # plt.arrow(start[0], start[1], end[0], end[1], fc='grey', ec='grey', linestyle='--',linewidth =0.2)
    # plt.text(0.071, 0.50, r'$d$ =10', fontsize=12, color='grey')


    label = []
    label.append('10 miners')
    label.append('50 miners')
    label.append('100 miners')
    label.append('Infinite miners')

    L = 50
    for i,n in enumerate(n_values):
        y_values = compute_function(p_values, n, L)
        plt.semilogx([1/p for p in p_values], y_values, label=f'n = {n}', linestyle='--', color = colorlist[i])
    plt.plot([1/p for p in p_values], [0.5-1/L/2]*len(p_values), label=f'n = {n}', linestyle='-.', color = 'grey')

    # x = a * np.cos(theta) + 0.05
    # y = b * np.sin(theta) + 0.545
    # plt.plot(x, y, color = 'grey')

    # end = [-0.015, 0.015]
    # start = [0.095, 0.715]
    # plt.arrow(start[0], start[1], end[0], end[1], fc='grey', ec='grey', linestyle='--',linewidth =0.2)
    # plt.text(0.0951, 0.715, r'$d$ = 50', fontsize=12, color='grey')
    
    # L = 100
    # for i,n in enumerate(n_values):
    #     y_values = compute_function(p_values, n, L)
    #     plt.plot(p_values, y_values, label=f'n = {n}', linestyle='-.', color = colorlist[i])




    plt.legend(label)
    plt.xlim([1/p for p in p_values][-1],[1/p for p in p_values][0])
    plt.grid()
    # plt.xlabel(r'$1-\left(1-p\right)^{n}$')
    # plt.xlabel(r'$1-(1-p)^n$')
    plt.xlabel("Block time")
    plt.ylabel(r'Network capability indicator $c$')
    
    # Call the function with your parameters
    print_fig_to_paper(output_type='eps', plot_file_name='boundary', 
                       fig_font_size=16, fig_font_name='Times New Roman', 
                       fig_width=7, is_print=True, is_print_time=False,
                       )
    print_fig_to_paper(output_type='png', plot_file_name='boundary', 
                       fig_font_size=16, fig_font_name='Times New Roman', 
                       fig_width=7, is_print=True, is_print_time=False,
                       )
    print_fig_to_paper(output_type='svg', plot_file_name='boundary', 
                       fig_font_size=16, fig_font_name='Times New Roman', 
                       fig_width=7, is_print=True, is_print_time=False,
                       )

