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
    

    nodes = np.arange(2, 72, 8)

    # Theoretical curves for PBFT (O(n^2)) and Raft (O(n))
    pbft_theory = 2 * nodes**2 - nodes - 1 # PBFT has quadratic complexity O(n^2)
    pow_theory = nodes  # Raft has linear complexity O(n)
    vdf_theory = nodes - 1


    # Simulation data (arbitrary example, can be replaced with real data)
    # print(pbft_data)
    # print(pow_data)
    # print(vdf_data)
    pbft_data = [7.0660, 191.4098, 628.2468, 1323.8445, 2277.7795, 3484.6981, 4948.3358, 6666.5579, 8644.7029]
    pow_data = [2.8739, 11.3659, 20.2994, 26.6586, 34.9419, 45.8964, 51.0278, 58.6370, 67.9660]
    nk_data = [2.3612, 10.8937, 20.5081, 28.5143, 35.6845, 46.9796, 51.8516, 59.4882, 67.6231]
    vdf_data = [3.0059, 11.5136, 18.9655, 26.9385, 35.5024, 43.5317, 50.7539, 58.2585, 64.9722]
    # Plot the curves
    # plt.plot(nodes, vdf_theory, 'b-', label="VDF (Theory)")
    # plt.plot(nodes, pbft_theory, 'r-', label="PBFT (Theory)")
    # plt.plot(nodes, pow_theory, 'g-', label="PoW (Theory)")

    # Plot the simulation data
    plt.plot(nodes, vdf_data, 'rx-', label="PoVD")
    plt.plot(nodes, pbft_data, 'b^-.', label="PBFT")
    plt.plot(nodes, pow_data, 'go--', label="PoW ")
    plt.plot(nodes, nk_data, color = 'orange', linestyle = 'dotted', marker = 's', label = "VDF Baseline")

    # Labels and title
    plt.xlabel("Number of miners")
    plt.ylabel("Communication complexity")
    plt.yscale('log')
    # plt.title("Communication Complexity vs Number of Nodes")

    # Add a legend
    plt.legend()

    # Show grid
    plt.grid(True)
    
    # Call the function with your parameters
    print_fig_to_paper(output_type='png', plot_file_name='Communication complexity', 
                       fig_font_size=16, fig_font_name='Times New Roman', 
                       fig_width=7, is_print=True, is_print_time=False,
                       )
    print_fig_to_paper(output_type='eps', plot_file_name='Communication complexity', 
                       fig_font_size=16, fig_font_name='Times New Roman', 
                       fig_width=7, is_print=True, is_print_time=False,
                       )


