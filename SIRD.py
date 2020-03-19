import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

def deriv(y, t, N, beta, gamma, delta):
    '''
    Function returning SIRD differential vectors.
    '''
    S, I, R, D = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I - delta * I
    dRdt = gamma * I
    dDdt = delta * I
    return dSdt, dIdt, dRdt, dDdt

def generate_vectors(beta, gamma, delta, I0, R0, D0, t, N):
    '''
    Function returning SIRD vectors.
    '''
    S0 = N - I0 - R0 - D0
    y0 = S0, I0, R0, D0
    ret = odeint(deriv, y0, t, args=(N, beta, gamma, delta))
    S, I, R, D = ret.T
    return S, I, R, D

def generate_quarantine_vectors(n_quarantine_groups, I0_total, N_samples):
    '''
    Function that randomly selects the number of initially infected individuals in each cluster (given by "n_quarantine_groups" total clusters), averages across N_samples trials.
    The sum of initially infected individuals must be I0_total.
    '''    
    SpopVec, IpopVec, RpopVec, DpopVec = [], [], [], []
    
    for _ in range(N_samples):
        
        I0_vector = np.random.multinomial(I0_total, np.ones(n_quarantine_groups)/n_quarantine_groups, size=1)[0]
        n_subjects = N // n_quarantine_groups
        
        Spop, Ipop, Rpop, Dpop = np.zeros(len(t)), np.zeros(len(t)), np.zeros(len(t)), np.zeros(len(t))
        
        for I0 in I0_vector:
            S, I, R, D = generate_vectors(beta, gamma, delta, I0, R0, D0, t, n_subjects)
            
            Spop += S
            Ipop += I
            Rpop += R
            Dpop += D
    
        SpopVec.append(Spop)
        IpopVec.append(Ipop)
        RpopVec.append(Rpop)
        DpopVec.append(Dpop)
        
        np.concatenate(SpopVec, axis=0)
        np.concatenate(IpopVec, axis=0)
        np.concatenate(RpopVec, axis=0)
        np.concatenate(DpopVec, axis=0)
        
        SpopAve = np.mean(SpopVec, axis=0)
        IpopAve = np.mean(IpopVec, axis=0)
        RpopAve = np.mean(RpopVec, axis=0)
        DpopAve = np.mean(DpopVec, axis=0)

    return SpopAve, IpopAve, RpopAve, DpopAve


def create_subplot(fig, nRow, nCol, S,I,R,D, ylim, n_group, additional_label=''):
    '''
    Function that generates SIRD subplots, formats them.
    '''
    global subplot_order, ch
    ax = fig.add_subplot(nRow*100+nCol*10+subplot_order, facecolor='#dddddd', axisbelow=True)   
    ax.plot(t, S, 'b', alpha=0.5, lw=2, label='Susceptible')
    ax.plot(t, I, 'r', alpha=0.5, lw=2, label='Infected')
    ax.plot(t, R, 'g', alpha=0.5, lw=2, label='Recovered/Immune')
    ax.plot(t, D, 'y', alpha=0.5, lw=2, label='Died')
    ax.set_ylabel('Population', fontsize=10)
    ax.set_title(('Scenario {}: Randomly quarantine population of {:,} into {:,} group(s) of size {:,}'+additional_label).format(ch, N, n_group, N//n_group), fontsize=10)
    ax.set_ylim(0,ylim)
    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=0)
    fmt = '{x:,.0f}'
    tick = mtick.StrMethodFormatter(fmt)
    ax.yaxis.set_major_formatter(tick) 
    ax.grid(b=True, which='major', c='w', lw=2, ls='-')
    ax.tick_params(
        axis='x',         
        which='both',      
        bottom=False,      
        top=False,         
        labelbottom=False)
    
    ymax = max(I)
    xpos = I.argmax()
    xmax = t[xpos]
    
    ax.plot(xmax,ymax,'ro--', linewidth=2, markersize=8)
    ax.annotate('Peak infection reached on day {0:.0f}, with {1:,.0f} people infected at peak'.format(xmax, ymax), xy=(125,2.5e6), fontsize=9)
    ax.text(151, 1.6e6, 'Total deaths: {0:,.0f} '.format(D[-1]), fontsize=9)
    
    for spine in ('top', 'right', 'bottom', 'left'):
        ax.spines[spine].set_visible(False)
    
    ch=chr(ord(ch) + 1)
    subplot_order+=1

    if subplot_order==nRow+1:
        
        ax.set_xlabel('days')
        ax.tick_params(
        axis='x',         
        which='both',      
        bottom=True,      
        top=False,         
        labelbottom=True)
    
    return ax


def run_analysis(beta, gamma, delta, N_samples, N, t):
    '''
    Function that generates figure with SIRD subplots - each portraying a quarantine scenario.
    '''
    fig = plt.figure(facecolor='w')
    fig.suptitle('Illustrating the Effect of Quarantining on Epidemic Progression using an SIRD Model with R0={:.1f}'.format(beta/(gamma+delta)), fontweight='bold')
    fig.text(0.33,0.93,'Simulation assumes {:,} infected individuals in a population of {:,} \n case infectivity ratio: {:.4f}, case recovery ratio: {:.4f}, case fatality ratio {:.4f}'.format(I0_total, N, beta, gamma , delta))
    
    n_groups = [1, 8, 26, 850, 60_000]
    labels = [' -- Roughly equivalent to closing down national borders', ' -- Roughly equivalent to Mohafaza-level quarantine',  ' -- Roughly equivalent to Caza-level quarantine', ' -- Roughly equivalent to town/city-level quarantine', ' -- Roughly equivalent to extended family-level quarantine']
    
    for n_group, label in zip(n_groups, labels):
        S, I, R, D = generate_quarantine_vectors(n_group, I0_total, N_samples)
        ax = create_subplot(fig, len(n_groups), 1, S,I,R,D, N, n_group, label)
       
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', fontsize=8)

if __name__ == '__main__':
    
    N = 6_000_000 #population
    N_samples = 100 #number of trials
    beta, gamma, delta = .227, .08, .0058 #infectivity, recovery, mortality rates
    I0_total, R0, D0 = 150, 0, 0 #total number of initially infected individuals
    t = np.linspace(0, 180, 180) #grid of days 
    subplot_order=1 #global variable that tracks order of subplots
    ch = 'A' #global variable that tracks scenario labels
    
    run_analysis(beta, gamma, delta, N_samples, N, t)