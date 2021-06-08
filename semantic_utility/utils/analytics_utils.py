import numpy as np
import matplotlib.pyplot as plt

def plot_learning_curve(function):
    """[summary]"""    
    def plot_wrapper(*args,**kwargs):
        figure_file = kwargs.pop('figure_file', None)
        x,y,figure_name= function(*args,**kwargs)
        plt.plot(x, y)
        plt.title(figure_name)
        plt.savefig(figure_file)
    return plot_wrapper

@plot_learning_curve
def avg_score_plot(x,y,limit=100):
    running_avg = np.zeros(len(y))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(y[max(0, i-limit):(i+1)])
    print(running_avg)
    figure_name='Running average of previous '+ str(limit) +' scores'
    return x , y ,figure_name

@plot_learning_curve	
def score_plot(x,y):
    figure_name= "This is plot gives the score of each episode"
    return x,y,figure_name

@plot_learning_curve	
def log_score_plot(x,y):
    figure_name= "This is plot gives the score of each episode"
    return x,y,figure_name
 
