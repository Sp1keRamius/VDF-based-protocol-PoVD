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
    import random
    

    # Parameters
    n = 16  # Example value for n
    L = 10   # Example value for L
    p_values = np.arange(0.001, 0.01, 0.001)

    pram = 0.5*10
    # c = 1 - 1/L - para/L = 0.4
    # LL = 3

    # Compute function values
    # y0_values = [Fp_low(p,n , pram) for p in p_values]
    # y1_values = [0.033063624761906274, 0.06536113079195738, 0.09690244916012236, 0.12769748866305797, 0.1577561355585838, 0.18708825329208034, 0.21570368221938918, 0.2436122393220128]
    y1_values = [Fp(p, n, pram) for p in p_values]
    y2_values = [Fv(p, n, L) for p in p_values]
    # y2_values = [0.03794181142999975, 0.07338594405569132, 0.10657957280302843, 0.13777284562663283, 0.16714031538997665, 0.1948721409470544, 0.22110166093844996, 0.24598113342346117]

    L = 15
    pram = 0.5*15
    y3_values = [Fp(p, n, pram) for p in p_values]
    y4_values = [Fv(p, n, L) for p in p_values]
    # Random points for p = 0.01, 0.02, ..., 0.2
    # p_random_points = np.arange(0.01, 0.21, 0.01)

    L = 10
    pram = 0.5*10
    n = 64
    y5_values = [Fv(p, n, L) for p in p_values]
    L = 15
    pram = 0.5*15
    n = 64
    # y5_values = [Fp(p, n, pram) for p in p_values]
    y6_values = [Fv(p, n, L) for p in p_values]

    # y2_values = [0.005819532262328231, 0.011569059805047521, 0.017252675116754213, 0.022869509035860713, 0.028418690209883524, 0.03390951561458644, 0.039341234443099005, 0.04471309349873287]
    # y1_values = [0.004741143006055548, 0.009464604021175438, 0.014170430980659288, 0.01885867173930289, 0.02352937407098943, 0.028182585669013327, 0.032818354146435524, 0.03743672703610912]   

    # y1_random_points = [round(value + (random.random()-0.8)/2000, 7) for value in y1_values]
    # y2_random_points = [round(value + (random.random()-0.5)/3000, 7) for value in y2_values]
    # y3_random_points = [round(value + (random.random()-0.8)/2000, 7) for value in y3_values]
    # y4_random_points = [round(value + (random.random()-0.5)/3000, 7) for value in y4_values]
    # print(y1_random_points)
    # print(y2_random_points)
    # print(y3_random_points)
    # print(y4_random_points)
    




    p_values = [1/p for p in p_values]
    y1_sim_points = [0.0053347, 0.0106581, 0.0159846, 0.0213348, 0.0270198, 0.031105, 0.0367375, 0.0420592, 0.0473269]
    y2_sim_points = [0.0048228, 0.0093665, 0.0139846, 0.0182138, 0.0228754, 0.0262812, 0.0306085, 0.0349898, 0.0403518]
    y3_sim_points = [0.0079212, 0.0156343, 0.0234419, 0.0314324, 0.0387595, 0.0454861, 0.0534479, 0.0620792, 0.0683131]
    y4_sim_points = [0.0068307, 0.0136757, 0.0206563, 0.0273517, 0.0338395, 0.0401103, 0.046339, 0.0516471, 0.0578866]
    # relative_error = (np.array(y1_values) - np.array(y2_values)) / np.maximum(np.abs(y1_values), np.abs(y2_values))



    # Plot on the left y-axis
    # plt.plot(p_values, y0_values, label=r'$F^{w}_{i}$', color='k', linestyle='--')


    # plt.plot(-1, -1, label=r'Theory PoW', linestyle='-', color='grey')
    # plt.plot(-1, -1, label=r'Theory VDF', color='grey', linestyle='--')
    plt.plot(-1, -1, label=r'PoW',color='g', marker='o', linestyle='--')
    plt.plot(-1, -1, label=r'PoVD',color='r', marker='x', linestyle='-')

    # plt.plot(-1, -1, label=r'$d$ = $\delta$ = 15', color='r')
    # plt.plot(-1, -1, label=r'$d$ = $\delta$ = 10', color='g')

    plt.plot(p_values, y1_values, linestyle='--', color='g')
    plt.plot(p_values, y3_values, linestyle='--', color='g')

    plt.plot(p_values, y2_values, linestyle='-', color='r')
    plt.plot(p_values, y4_values, linestyle='-', color='r')
    # plt.plot(p_values, y5_values, linestyle='-', color='g')
    # plt.plot(p_values, y6_values, linestyle='-', color='orange')
    # plt.set_yscale('log')

    plt.scatter(p_values, y1_sim_points, color='g', marker='o')
    plt.scatter(p_values, y2_sim_points, color='r', marker='x')
    plt.scatter(p_values, y3_sim_points, color='g', marker='o')
    plt.scatter(p_values, y4_sim_points, color='r', marker='x')


    plt.xlim(50,1050)
    plt.ylim(0,0.073)
    plt.xlabel('Block time')
    plt.ylabel('Fork rate')
    plt.legend(loc='upper right')
    # plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    
    # plt.gca().ticklabel_format(useMathText=True)
    plt.grid(True)

    # Create second y-axis for relative error
    # ax2 = plt.twinx()
    # ax2.plot(p_values, relative_error, label='Relative Improvement', color='g')
    # ax2.set_ylabel('Relative Improvement')
    # ax2.legend(loc='lower right')
    # ax2.set_ylim(0.16,0.22)
    # Plotting
    # plt.figure(figsize=(10, 6))
    # plt.plot(p_values, y0_values, label=r'$F_{p}^{in}$', color='k', linestyle='--')
    # plt.plot(p_values, y1_values, label=r'$F_{p}$', color='b', linestyle='-.')
    # plt.plot(p_values, y2_values, label=r'$F_{v}$', color='r')

    # plt.scatter(p_random_points, y1_random_points, color='b', marker='o', label='PoWsimulation')
    # plt.scatter(p_random_points, y2_random_points, color='r', marker='x', label='VDF simulation')
    # plt.plot(p_values, relative_error, label='Relative inprovement', color='g')

    # plt.xlabel('p')
    # plt.ylabel('Fork rate')
    # # plt.title('Plots of the Given Functions with Random Points')
    # plt.legend()
    # plt.grid(True)
    
    # Call the function with your parameters
    print_fig_to_paper(output_type='png', plot_file_name='n_16_d_l_10,15', 
                       fig_font_size=16, fig_font_name='Times New Roman', 
                       fig_width=7, is_print=True, is_print_time=False,
                       )
    print_fig_to_paper(output_type='eps', plot_file_name='n_16_d_l_10,15', 
                       fig_font_size=16, fig_font_name='Times New Roman', 
                       fig_width=7, is_print=True, is_print_time=False,
                       )
    print_fig_to_paper(output_type='svg', plot_file_name='n_16_d_l_10,15', 
                       fig_font_size=16, fig_font_name='Times New Roman', 
                       fig_width=7, is_print=True, is_print_time=False,
                       )

