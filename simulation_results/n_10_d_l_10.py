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
    n = 10  # Example value for n
    L = 10   # Example value for L
    p_values = np.arange(0.001, 0.01, 0.001)

    pram = 0.5*10
    # c = 1 - 1/L - para/L
    # LL = 3

    # Compute function values
    # y0_values = [Fp_low(p,n , pram) for p in p_values]
    # y1_values = [0.033063624761906274, 0.06536113079195738, 0.09690244916012236, 0.12769748866305797, 0.1577561355585838, 0.18708825329208034, 0.21570368221938918, 0.2436122393220128]
    blocktime = np.linspace(110, 1000, 1000)
    y1_values = [Fp(1/b, n, pram) for b in blocktime]
    y2_values = [Fv(1/b, n, L) for b in blocktime]
    # y2_values = [0.03794181142999975, 0.07338594405569132, 0.10657957280302843, 0.13777284562663283, 0.16714031538997665, 0.1948721409470544, 0.22110166093844996, 0.24598113342346117]

    pram = 0.45*10
    y3_values = [Fp(1/b, n, pram) for b in blocktime]
    y4_values = [Fv(1/b, n, L) for b in blocktime]

    pram = 0.6*10
    y4_values = [Fp(1/b, n, pram) for b in blocktime]

    # Random points for p = 0.01, 0.02, ..., 0.2
    # p_random_points = np.arange(0.01, 0.21, 0.01)

    # y2_values = [0.005819532262328231, 0.011569059805047521, 0.017252675116754213, 0.022869509035860713, 0.028418690209883524, 0.03390951561458644, 0.039341234443099005, 0.04471309349873287]
    # y1_values = [0.004741143006055548, 0.009464604021175438, 0.014170430980659288, 0.01885867173930289, 0.02352937407098943, 0.028182585669013327, 0.032818354146435524, 0.03743672703610912]   

    # y1_random_points = [round(value + (random.random()-0.8)/2000, 7) for value in y3_values]
    # # y2_random_points = [Fv(p, n, L) + (random.random()-1)/500 for p in p_random_points]
    # y2_random_points = [round(value + (random.random()-0.5)/3000, 7) for value in y4_values]
    # print(y1_random_points)
    # print(y2_random_points)





    y1_sim_points = [0.005418, 0.01095, 0.015979, 0.021241, 0.026554, 0.032332, 0.037002, 0.042392, 0.047522]
    y2_sim_points = [0.004449, 0.009032, 0.013359, 0.017619, 0.021616, 0.026079, 0.029876, 0.033875, 0.037814]
    # y2_sim_points = [0.004459, 0.009092, 0.012959, 0.017319, 0.022016, 0.026319, 0.030096, 0.033915, 0.037474]
    y3_sim_points = [0.0046515, 0.0096507, 0.0147841, 0.0196082, 0.0243955, 0.0293787, 0.0339459, 0.0383445, 0.0432045]
    y4_sim_points = [0.0062657, 0.0123608, 0.0191834, 0.0254823, 0.032102, 0.0373609, 0.0430855, 0.0498438, 0.0558054]
    # relative_error = (np.array(y1_values) - np.array(y2_values)) / np.maximum(np.abs(y1_values), np.abs(y2_values))

    # fig, plt = plt.subplots()

    # Plot on the left y-axis
    # plt.plot(p_values, y0_values, label=r'$F^{w}_{i}$', color='k', linestyle='--')

    # plt.scatter(-1, -1, marker='o', label = 'PoW', color='grey')
    # plt.scatter(-1, -1, marker='x', label = 'VDF', color='grey')
    # plt.plot(-1, -1, label=r'$F^{w}$', linestyle='-.', color='grey')
    # plt.plot(-1, -1, label=r'$F^{v}$', color='g', linestyle='--')

    # plt.plot(-1, -1, label=r'c=0.4', color='r')
    # plt.plot(-1, -1, label=r'c=0.45', color='b')
    p_values = [1/p for p in p_values]
    plt.plot(blocktime, y1_values, linestyle='--', color='g')
    plt.plot(blocktime, y3_values, linestyle='--', color='b')
    plt.plot(blocktime, y2_values, linestyle='-', color='r')
    plt.plot(blocktime, y4_values, linestyle='--', color='orange')

    plt.scatter(p_values, y1_sim_points, color='g', marker='o')
    # plt.plot(-1, -1, label=r'PoW, $c$ = 0.4', color='r', marker='o', linestyle='-')

    plt.scatter(p_values, y3_sim_points, color='b', marker='o')
    # plt.plot(-1, -1, label=r'PoW, $c$ = 0.45', color='b', marker='o', linestyle='-')

    plt.scatter(p_values, y4_sim_points, color='orange', marker='o')
    # plt.plot(-1, -1, label=r'PoW, $c$ = 0.45', color='b', marker='o', linestyle='-')

    plt.scatter(p_values, y2_sim_points, color='r', marker='x')
    # plt.plot(-1, -1, label=r'VDF', color='g', marker='x', linestyle='--')
    # plt.scatter(p_values, y4_sim_points, color='r', marker='d')

    plt.plot(-1, -1, label=r'PoW',color='grey', marker='o', linestyle='--')
    plt.plot(-1, -1, label=r'PoVD',color='grey', marker='x', linestyle='-')

    plt.xlim(50,1050)
    plt.ylim(0,0.06)
    # plt.yscale('log')
    plt.xlabel('Block time')
    plt.ylabel('Fork rate')
    
    # plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    # plt.gca().ticklabel_format(useMathText=True)
    
    # plt.legend(loc='upper right')
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
    print_fig_to_paper(output_type='png', plot_file_name='n_10_d_l_10', 
                       fig_font_size=16, fig_font_name='Times New Roman', 
                       fig_width=7, is_print=True, is_print_time=False,
                       )
    print_fig_to_paper(output_type='eps', plot_file_name='n_10_d_l_10', 
                       fig_font_size=16, fig_font_name='Times New Roman', 
                       fig_width=7, is_print=True, is_print_time=False,
                       )
    print_fig_to_paper(output_type='svg', plot_file_name='n_10_d_l_10', 
                       fig_font_size=16, fig_font_name='Times New Roman', 
                       fig_width=7, is_print=True, is_print_time=False,
                       )


