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
    

    # fig, plt = plt.subplots()

    # p_list = [0.000488166,0.000976101,0.001463805,0.001951278,0.002438521,0.002925534,0.003412316,
    #           0.003898868,0.00438519,0.004871282,0.005357144,0.005842776,0.006328178,0.006813351,0.007298294,0.007783008,
    #           0.008,0.009,0.010,0.011,0.012,0.013,0.014,0.015,0.016,0.017,0.018,0.019,0.02]
    # p_list = [0.000488166,0.000976101,0.001463805,0.001951278,0.002438521,0.002925534,0.003412316,
    #           0.003898868,0.00438519,0.004871282,0.005357144,0.005842776,0.006328178,0.006813351,0.007298294,0.007783008]
    p_list = [0.008,0.009,0.010,0.011,0.012,0.013,0.014,0.015,0.016,0.017,0.018,0.019,0.02]
    # opt_forkrate = [0.004460168362421246, 0.008802004177745681, 0.013037343285719705, 0.017173777212535213, 0.021219058555287607,
    #                  0.02516301038382851, 0.02902256158499328, 0.03279670748608443, 0.03648445408371559, 0.040103174453179125,
    #                  0.04364288246848358, 0.047102712940312874, 0.0505096318010414, 0.053835159343115224, 0.05710646336294112,
    #                  0.060313622725829075,0.06173818465998315, 0.06812692877486237, 0.07429114615551691, 0.08024558374851853, 
    #                  0.08601512837893566, 0.09158672554834502, 0.096995759783158, 0.10223942531583707, 0.10733460341823564, 
    #                  0.11228874527248878, 0.1170996083282474, 0.12178492352555126, 0.12635281346341587]
    # opt_forkrate = [0.003998762526662647, 0.007956838026670465, 0.011877193206547787, 0.015759094972944254, 0.019594206113612733, 
    #                 0.023396944566407618, 0.02715900754699141, 0.03088737149660037, 0.03458140910568708, 0.03823268389767598, 
    #                 0.04185613258461418, 0.04543546638006879, 0.048993706627932454, 0.05250655009630456, 0.055989247292978495, 0.05944127176818803,
    opt_forkrate = [0.060980159112031807, 0.06797643789277252, 0.07485515660142972, 0.08161220695177829, 0.08824344026038466, 
                    0.09477009068727948, 0.10118866366850099, 0.107495630138711, 0.11369614230589808, 0.11979564089551664, 
                    0.12579985448379816, 0.13171479879863246, 0.13752872593529142]
    # opt_forkrate = [0.0021908730284263545, 0.004372311168066911, 0.006533779436627363, 0.008688971217225117, 
    #                 0.010830808177584306, 0.012966140304408125, 0.015080801558427726, 0.01718867852821082, 
    #                 0.019282593091695888, 0.02136949039438263, 0.02344214271229661, 0.025500383845306396, 
    #                 0.027551230086437117, 0.029594586733864392, 0.031623127486577474, 0.03363668447978563]
    # pow_forkrate = [0.004134235024279342, 0.008251459365235747, 0.01235174180367904, 0.016435150828553402, 0.020501763003933915,
    #                  0.02455164619893624, 0.028584859745202307, 0.032601479309642745, 0.03660157196480529, 0.04058520451833558, 
    #                  0.04455244351512755, 0.04850335523880378, 0.052438005711963, 0.05635646875871714, 0.06025880176275866, 
    #                  0.06414507797358038,0.06588025494569305, 0.07384018256741243, 0.08174029274223893, 0.08958097457974157, 
    #                  0.09736261505294164, 0.1050855990077203, 0.1127503091718236, 0.1203571261655233, 0.12790642850983902, 
    #                  0.13539859263700904, 0.1428339928992346, 0.15021300157786643, 0.1575359888937622]
    # pow_forkrate = [0.002414102889074732, 0.004822433488213762, 0.007225004942112312, 0.009621830372297024, 0.012012927771811954, 
    #                 0.014398310177992202, 0.016777985714429455, 0.019151972264662476, 0.021520282782525935, 0.023882930193341534, 
    #                 0.026239927396299545, 0.028591287262648812, 0.030937022636150502, 0.033277151152592, 0.03561168076467469, 0.0379406290350206]
    pow_forkrate = [0.06588025494569305, 0.07384018256741243, 0.08174029274223893, 0.08958097457974157, 0.09736261505294164, 
                    0.1050855990077203, 0.1127503091718236, 0.1203571261655233, 0.12790642850983902, 0.13539859263700904, 
                    0.1428339928992346, 0.15021300157786643, 0.1575359888937622]
    opt_forkrate_sim = [0.06173, 0.06791, 0.07504, 0.08199, 0.08927, 0.09615, 0.10145, 0.107495630138711, 0.11403, 0.11979564089551664, 
                    0.1259, 0.1315, 0.1371]
    pow_forkrate_sim = [0.066, 0.073, 0.07970000000000001, 0.08885, 0.09645, 0.1046, 0.1111, 0.1205, 0.1283, 0.13505, 0.14205, 0.1502, 0.1573]
    
    # opt_forkrate_sim = [0.00222, 0.00416, 0.00672, 0.00858, 0.01008, 0.01332, 0.01468, 0.01718, 0.01946, 0.02082, 0.02324, 0.0257, 0.02718, 0.03006, 0.03108, 0.03444]
    # pow_forkrate_sim = [0.00236, 0.00476, 0.00738, 0.00946, 0.01204, 0.014799999999999999, 0.01636, 0.01956, 0.021859999999999997, 0.024880000000000003, 0.02588, 0.028259999999999997, 0.03148, 0.0323, 0.035500000000000004, 0.03816]
    # 计算误差
    error = [-(o - p) / p for o, p in zip(opt_forkrate, pow_forkrate)]
    error2 = [-(o - p) / p * 100 for o, p in zip(opt_forkrate_sim, pow_forkrate_sim)]
    
    # 找到误差为零的点
    zero_crossings = [p_list[i] for i in range(1, len(error)) if error[i-1] * error[i] < 0]
    zero_points = [(p_list[i], opt_forkrate[i]) for i in range(1, len(error)) if error[i-1] * error[i] < 0]

    # 画出pow和opt曲线
    p_list = [(1 - (1-p)**(1/32)) for p in p_list]
    plt.plot(p_list, pow_forkrate, label=r'$F^{w}$', color='b', linestyle='-.')
    plt.plot(p_list, opt_forkrate, label=r'$F^{v}$', color='r')

    plt.scatter(p_list,
                 pow_forkrate_sim, color='b', marker='o', label='simulation PoW')

    plt.scatter(p_list,
                 opt_forkrate_sim, color='r', marker='x', label='simulation VDF')

    plt.xlabel(r'$p$', fontsize=14)
    plt.ylabel('Fork Rate', fontsize=14)
    plt.legend(loc='upper left')
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    
    plt.gca().ticklabel_format(useMathText=True)
    plt.grid()

    # # 创建第二个纵坐标轴
    # ax2 = plt.twinx()
    # ax2.plot(p_list, error, label='Relative Improvement', color='g')
    # # ax2.plot(p_list, 
    # #          error2, linestyle='-.', color='orange', label="relative improvement sim (%)",alpha=0.6)
    # ax2.set_ylabel('Relative Improvement', fontsize=14)
    # ax2.legend(loc='lower right')
    # ax2.set_ylim(0.06,0.14)
    
    # # 设置第二纵坐标轴为百分比格式
    # ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.2f}'.format(y)))
    
    # 在误差为零的位置画竖线并标记交点
    for zero in zero_crossings:
        plt.axvline(x=zero, color='k', linestyle='--',lw=0.9)

    for point in zero_points:
        plt.plot(point[0], point[1], 'ko')  # 标记交点
        plt.text(point[0], point[1], f'({point[0]:.4f}, {point[1]:.4f})', fontsize=10, ha='left')

    
    # Call the function with your parameters
    print_fig_to_paper(output_type='png', plot_file_name='32_round', 
                       fig_font_size=16, fig_font_name='Times New Roman', 
                       fig_width=7, is_print=True, is_print_time=False,
                       )
    
    print_fig_to_paper(output_type='eps', plot_file_name='32_round', 
                       fig_font_size=16, fig_font_name='Times New Roman', 
                       fig_width=7, is_print=True, is_print_time=False,
                       )


