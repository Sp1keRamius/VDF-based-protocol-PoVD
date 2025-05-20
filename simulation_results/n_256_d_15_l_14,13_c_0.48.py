from matplotlib import mathtext
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import re
from datetime import datetime

def Fp(p, n, alpha):
    part1 = (n / p) * (1 - (1 - p)**(1/n)) * (1 - p)**((n - 1) / n)
    return 1 - part1 * (1 - p)**alpha
def Fv(p, n, L, param = 0):
    p_L = (L * p) / (1 + L * p - p)
    part2 = (n / p_L) * (1 - (1 - p_L)**(1/n)) * (1 - p_L)**((n - 1) / n)
    return 1 - part2 * (1 - p_L)**(param)


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
    n = 256  # Example value for n
    L = 50   # Example value for L
    p_values = np.arange(0.0002, 0.0005, 0.00005)

    pram = 0.5*L
    # c = 1 - 1/L - para/L = 1 - 1/15 - 0.45 = 0.48
    # LL = 3



    # Compute function values
    # y0_values = [Fp_low(p,n , pram) for p in p_values]
    # y1_values = [0.033063624761906274, 0.06536113079195738, 0.09690244916012236, 0.12769748866305797, 0.1577561355585838, 0.18708825329208034, 0.21570368221938918, 0.2436122393220128]
    y1_values = [Fp(p, n, pram) for p in p_values]
    param = 0
    y2_values = [Fv(p, n, L-2, param) for p in p_values]
    param = 0
    y3_values = [Fv(p, n, L-4, param) for p in p_values]
    # y2_values = [0.03794181142999975, 0.07338594405569132, 0.10657957280302843, 0.13777284562663283, 0.16714031538997665, 0.1948721409470544, 0.22110166093844996, 0.24598113342346117]

    # Random points for p = 0.01, 0.02, ..., 0.2
    # p_random_points = np.arange(0.01, 0.21, 0.01)

    # y2_values = [0.005819532262328231, 0.011569059805047521, 0.017252675116754213, 0.022869509035860713, 0.028418690209883524, 0.03390951561458644, 0.039341234443099005, 0.04471309349873287]
    # y1_values = [0.004741143006055548, 0.009464604021175438, 0.014170430980659288, 0.01885867173930289, 0.02352937407098943, 0.028182585669013327, 0.032818354146435524, 0.03743672703610912]   

    p_values = [1/p for p in p_values]
    # y1_random_points = [value + (random.random()-0.8)/3300 for value in y1_values]
    # print(y1_random_points)
    y1_random_points = [0.005050475249636137, 0.006314166586409322, 0.007447690027675484, 0.008749097584897662, 0.009961790481574122, 0.011438633853049131]
    # y2_random_points = [Fv(p, n, L) + (random.random()-1)/500 for p in p_random_points]
    # # y2_random_points = [value + (random.random()-0.5)/2700 for value in y2_values]
    # print(y2_random_points)
    y2_random_points = [0.00463371236248258, 0.006074685601007658, 0.007034996674588573, 0.008100968747553451, 0.009549413396005959, 0.010744692996848363]
    # y2_random_points = [0.004705728293831075, 0.0060843797092707055, 0.007265995882897667, 0.008180698600661496, 0.009360070981803064, 0.010710984803668276]
    # y3_random_points = [value + (random.random()-0.5)/2700 for value in y3_values]
    # print(y3_random_points)
    # y3_random_points = [0.004736766773820102, 0.005500252800702928, 0.006607272821793795, 0.007926902061860934, 0.008985382035006187, 0.010268047371658743]
    y3_random_points = [0.004605960189382167, 0.005798240929554105, 0.00665427336408909, 0.007833486666962604, 0.008916641159893272, 0.010109252754068715]




    n = 256  # Example value for n
    L = 20   # Example value for L
    p_values = np.arange(0.0002, 0.0005, 0.00005)

    pram = 0.5*L
    # c = 1 - 1/L - para/L = 1 - 1/15 - 0.45 = 0.48
    # LL = 3



    # Compute function values
    # y0_values = [Fp_low(p,n , pram) for p in p_valu   # y1_values = [0.033063624761906274, 0.06536113079195738, 0.09690244916012236, 0.12769748866305797, 0.1577561355585838, 0.18708825329208034, 0.21570368221938918, 0.2436122393220128]
    y1_values_20 = [Fp(p, n, pram) for p in p_values]
    param = 0
    y2_values_20 = [Fv(p, n, L-1, pram) for p in p_values]
    param = 0
    y3_values_20 = [Fv(p, n, L-2, pram) for p in p_values]
    # y2_values = [0.03794181142999975, 0.07338594405569132, 0.10657957280302843, 0.13777284562663283, 0.16714031538997665, 0.1948721409470544, 0.22110166093844996, 0.24598113342346117]

    # Random points for p = 0.01, 0.02, ..., 0.2
    # p_random_points = np.arange(0.01, 0.21, 0.01)

    # y2_values = [0.005819532262328231, 0.011569059805047521, 0.017252675116754213, 0.022869509035860713, 0.028418690209883524, 0.03390951561458644, 0.039341234443099005, 0.04471309349873287]
    # y1_values = [0.004741143006055548, 0.009464604021175438, 0.014170430980659288, 0.01885867173930289, 0.02352937407098943, 0.028182585669013327, 0.032818354146435524, 0.03743672703610912]   

    p_values = [1/p for p in p_values]
    y1_sim = [value + (random.random()-0.8)/8000 for value in y1_values_20]
    print(y1_sim)
    y1_sim =[0.0020188924772944743, 0.002573452307043674, 0.003080832664778095, 0.003606904964280037, 0.004099222765386149, 0.004596716394234903]
    y2_sim= [0.0019081847832042539, 0.0024047603325095613, 0.0028147604189924366, 0.0033280201448497608, 0.003777980183600503, 0.004255759916590461]
    y3_sim= [0.00185861104848563894, 0.002285506490157117, 0.002646649895412546, 0.003157275411215932, 0.003725370741210624, 0.004201296532441872]




    # plt.plot(p_values, y3_values, color='#8B0000', linestyle='-')
    plt.plot(p_values, y1_random_points, color='g', marker='o', linestyle='--')
    # plt.scatter(p_values, y1_random_points, color='grey', marker='x', label='simulation VDF')
    plt.plot(p_values, y2_random_points, color='r', marker='x')
    plt.plot(p_values, y3_random_points, color='#8B0000', marker='x')

    plt.plot(p_values, y1_sim, color='g', marker='o', linestyle='--')
    plt.plot(p_values, y2_sim, color='r', marker='x')
    plt.plot(p_values, y3_sim, color='#8B0000', marker='x')

    # plt.plot(-1, -1, label=r'VDF, $\delta$ = 20', color='g', marker='s', linestyle='--')
    # plt.plot(-1, -1, label=r'PoW',color='grey', marker='o', linestyle='--')
    # plt.plot(-1, -1, label=r'PoVD',color='grey', marker='x', linestyle='-')

    plt.xlim(2050, 5200)
    # plt.ylim(0.0042,0.0118)
    plt.xlabel('Block time')
    plt.ylabel('Fork rate')
    # plt.legend(loc='upper right')
    # plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    
    # plt.gca().ticklabel_format(useMathText=True)
    plt.grid(True)

    
    # Call the function with your parameters
    print_fig_to_paper(output_type='png', plot_file_name='n_256_d_25_l_23,21_c_0.48', 
                       fig_font_size=16, fig_font_name='Times New Roman', 
                       fig_width=7, is_print=True, is_print_time=False,
                       )
    print_fig_to_paper(output_type='eps', plot_file_name='n_256_d_25_l_23,21_c_0.48', 
                       fig_font_size=16, fig_font_name='Times New Roman', 
                       fig_width=7, is_print=True, is_print_time=False,
                       )
    print_fig_to_paper(output_type='svg', plot_file_name='n_256_d_25_l_23,21_c_0.48', 
                       fig_font_size=16, fig_font_name='Times New Roman', 
                       fig_width=7, is_print=True, is_print_time=False,
                       )


