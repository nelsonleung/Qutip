#!/usr/bin/python

#multimode simulations
#author:peter g

import qutip as q
import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
import multiprocessing as mp
import itertools 
import timeit
import time
import copy

default_sim_parameters={
'show_approx_evolution':False,
'show_plots':True,
'show_progress_bar':True,
'only_show_ev':False,
'dissipators_in_dressed_basis':False,
'print_extra_info':True,
'odeoptions':q.Odeoptions(),
'bessel_indices':[1],
'show_approx_evolution':False,
'transform_before_measurement':False #should we apply a unitary basis transformation on the evolved density matrix before calculating expectation values
}
# 'odeoptions':q.Odeoptions(nsteps=1000, norm_tol=0.001, rtol=1e-6, atol=1e-8, max_step=0.001)
# 'odeoptions':q.Odeoptions(nsteps=10000, norm_tol=0.00001, rtol=1e-9, atol=1e-9, max_step=0.00001)

data_folder="./data/"
hbar=1.0545718e-34

np.set_printoptions(precision=5, suppress=True)

params = {
        # 'text.usetex':True,
        #'legend.fontsize': 16,
        # 'legend.linewidth': 3,
        # 'font.size': 18,
        'font.size': 26,
        # 'font.size': 20,
        #'font.size': 12,
        # 'lines.linewidth': 2,
        'lines.linewidth': 2,
        'lines.markersize': 5,
        'legend.fontsize': 18,
        'axes.grid': True,
        #'legend.linewidth': 2,
        #'figure.figsize':(5,3.105)}
        'figure.figsize':(12,12/1.61)}
        # 'figure.figsize':(10,3.105)}
        #'figure.figsize':(12,3.425)}
plt.rcParams.update(params)

def ts():
    """Start timer"""
    global start_time
    start_time=timeit.default_timer() 

def te():
    """End timer and display time elapsed"""
    global start_time
    end_time=timeit.default_timer()
    print("\nrun time: %fs\n" % (end_time-start_time))

class DictExtended(dict):
    """A dictionary that also provides access to key/value pairs via the 'dot' notation. 
    NOTE: Might break things if keys are the same as dictionary internal vars...  (just don't start keys with underscores to avoid this, or use 'self')
    """
    def __init__(self, initfrom=(), **kw):
        dict.__init__(self, initfrom)
        self.update(kw)
        self.__dict__ = self


def parameters_to_string(p):
    """we could make this a method of DictExtended or inherit... but let's not pollute... or overdo it.
    """
    parameters=[
            "epsilon=%0.4g GHz" % float(p.epsilon/(2.0*np.pi))
            ,"omega_drive=%0.4g GHz" % float(p.omega_drive/(2.0*np.pi))
            ,"non_rwa_terms=%d" % p.non_rwa_terms
            ,"mode_count=%d" % p.mode_count
            ,"\nN_m=%d" % p.N_m
            ,"mode_freqs=%s GHz" % ",".join(["%0.4g" % float(val/(2.0*np.pi)) for val in p.mode_freqs])
            ,"N_q=%d" % p.N_q
            ,"q_level_freqs=%s GHz" % ",".join(["%0.4g" % float(val/(2.0*np.pi)) for val in p.q_level_freqs])
            ,"\nq_level_decay_rate=%s GHz" %  ",".join(["%0.4g" % float(val) for val in p.q_level_decay_rate])
            ,"q_level_dephasing_rate=%s GHz" %  ",".join(["%0.4g" % float(val) for val in p.q_level_dephasing_rate])
            ,"mode_q_coupling_freqs=%s GHz" % ",".join(["%0.4g" % float(val/(2.0*np.pi)) for val in p.mode_q_coupling_freqs])
            ,"\nmode_decay_rates=%s GHz" % ",".join(["%0.4g" % float(val/(2.0*np.pi)) for val in p.mode_decay_rates])
            ,"dissipators_in_dressed_basis=%r" % p.dissipators_in_dressed_basis
            ]
    return r', '.join(parameters) 


def fock_state(p, qudit_n, mode_n_entries=[], density_matrix_form=True):
    """
    Creates specific fock states
    p: DictExtended object with sizes of qudit and mode Hilbert spaces
    qubit_n: fock level of the qudit
    mode_n_entries: list of lists, each with an index and a corresponding (nonground) fock state
    For example:
        mode_n_entries=[[0, 2], [4, 1]]
    would mean we want to have the mode with zero index be in the 2nd excited state, and mode with index 4
    be in the 1st excited state. All other modes should be in their ground states.

    """
    mode_states = [q.basis(p.N_m, 0)] * p.mode_count 

    for entry in mode_n_entries:
        mode_states[entry[0]]=q.basis(p.N_m, entry[1])

    state=q.tensor(q.basis(p.N_q, qudit_n), *mode_states)

    if density_matrix_form:
        return q.ket2dm(state)

    return state

def generate_single_excitation_states(p):
    """Generate a collection of single excitation fock states for our given system.
    """
    states=DictExtended()
    states["state_1"]=fock_state(p, 1, [])
    for m in xrange(p.mode_count):
        states["state_0" + ("_0" * m ) + "_1"]=fock_state(p, 0, [[m, 1]])
    return states

def parallel_map(f, a_list, *args, **kw):
    """
    NOTE: jupyter seems a bit sketchy with multiprocessing - in particular keeping track of spawned processes. 
    when things break, one may need to restart the jupyter server
    """

    pool=mp.Pool(*args, **kw)
    time.sleep(0.8) #maybe/probably not needed as Pool() blocks until procs are spawned i think.

    result=pool.map(f, a_list)

    try:
        #try to close everything up
        pool.close()
        pool.terminate()
    except:
        pass

    return result

def state_name_to_latex(name):
    if name.startswith("state"):
        name=name.replace("state","").replace("_","")
        return r'$\langle %s | \rho(t) | %s \rangle$' % (name, name)
    return name

def plot_results(p, results):

    #TEST: playing with some for-presentation-plots
    # linestyles_iterator=iter(['-', '-', '--', '-.'])
    #linestyles_iterator=itertools.cycle(['-', '-', '--', '-.'])
    # linestyles_iterator=itertools.cycle(['-', '-', '-', '-', '-'])
    linestyles_iterator=itertools.cycle(['-'])

    fig=plt.figure(figsize=(12,6))
    ax = fig.add_subplot(111) 
    color_iterator=itertools.cycle(plt.cm.brg(np.linspace(0,1,len(p.states_to_evolve))))
    color_iterator2=itertools.cycle(plt.cm.brg(np.linspace(0,1,len(p.states_to_evolve))))
    # color_iterator2=itertools.cycle(plt.cm.jet(np.linspace(0,1,len(p.states_to_evolve))))

    # color_iterator=itertools.cycle(plt.cm.brg([0.0, 0.5, 0.75, 0.25]))  #special case - 3 curves
    # color_iterator=itertools.cycle(plt.cm.brg([0.0, 0.5, 0.75]))  #special case - 3 curves
    # color_iterator=itertools.cycle(plt.cm.jet(np.linspace(0,1,len(p.states_to_evolve))))
    # color_iterator=itertools.cycle(plt.cm.hsv(np.linspace(0,1,len(p.states_to_evolve))))
    for i, state_name in enumerate(p.states_to_evolve):
        legend_label=state_name_to_latex(state_name)
        a_color=color_iterator.next()
        ax.plot(p.tlist, np.real(results.output.expect[i]), linestyle=linestyles_iterator.next(), color=a_color, label=legend_label)
        # ax.plot(p.tlist, np.real(results.output.expect[i]), '-', color=a_color, label=legend_label)
        if p.show_approx_evolution:
            a_color=color_iterator2.next()
            ax.plot(p.tlist, np.real(results.output_rf.expect[i]), '--', color=a_color, label="rf:" + legend_label)
    ax.set_ylim(0.0,1.0)
    ax.set_xlabel("Time (ns)")
    ax.set_title("Lab and Rotating Frames Evolution")
    ax.legend(loc='center left', fancybox=False, shadow=False, bbox_to_anchor=(1.0, 0.5 ))

    fig=plt.figure(figsize=(12,6))
    ax = fig.add_subplot(111) 
    color_iterator=itertools.cycle(plt.cm.brg(np.linspace(0,1,len(p.states_to_evolve))))
    color_iterator2=itertools.cycle(plt.cm.brg(np.linspace(0,1,len(p.states_to_evolve))))
    for i, state_name in enumerate(p.states_to_evolve):
        legend_label=state_name_to_latex(state_name)
        a_color=color_iterator.next()
        ax.plot(p.tlist, np.real(results.output.expect[i]), linestyle=linestyles_iterator.next(), color=a_color, label=legend_label)
        # ax.plot(p.tlist, np.real(results.output.expect[i]), '-', color=a_color, label=legend_label)
    ax.set_ylim(0.0,1.0)
    ax.set_xlabel("Time (ns)")
    ax.set_title("Lab Frame Evolution")
    ax.legend(loc='center left', fancybox=False, shadow=False, bbox_to_anchor=(1.0, 0.5 ))

    fig=plt.figure(figsize=(12,6))
    ax = fig.add_subplot(111) 
    color_iterator=itertools.cycle(plt.cm.brg(np.linspace(0,1,len(p.states_to_evolve))))
    color_iterator2=itertools.cycle(plt.cm.brg(np.linspace(0,1,len(p.states_to_evolve))))
    for i, state_name in enumerate(p.states_to_evolve):
        legend_label=state_name_to_latex(state_name)
        a_color=color_iterator2.next()
        ax.plot(p.tlist, np.real(results.output_rf.expect[i]), '--', color=a_color, label="rf:" + legend_label)
    ax.set_ylim(0.0,1.0)
    ax.set_xlabel("Time (ns)")
    ax.set_title("Rotating Frame Evolution")
    ax.legend(loc='center left', fancybox=False, shadow=False, bbox_to_anchor=(1.0, 0.5 ))

def run_sim_full_and_approx(p):
    """
    TODO:
    -Should get separate the approximate (rotating wave) evolution into its own function. Then the plotting routines should take a list 
        of results, using different  
    -Double check that coupling of higher qudit levels scales as standard lowering/rasing ops to a good approximation.
    -Add co-rotating terms evolution to approximate RF results.

    IMPORTANT:
    the passed in options 'p' should not be changed inside this function if one wants to run things in parallel via multiprocessing.Pool.map()
    Otherwise, in particular if p gets entries with new objects, can lead to things break down in non-trivial way.
    
    """

    # if p.show_approx_evolution:
        # print("The codebase for the approx evolution needs a couple of minor updates with the recent changes...\n\n\n")
        # return None

    if p.N_q>2 and p.show_approx_evolution:
        #Need to generalize the approximate treatment to more levels
        print("ERROR: The approximate evolution (in terms of J_m functions) for now only handles a 2 level qudit...\n\
        ... set p.show_approx_evolution=False in order to look at the full evolution instead.\n") 
        return None

    #make sure the config is sane
    assert(len(p.mode_freqs)==len(p.mode_q_coupling_freqs)==len(p.mode_decay_rates))
    assert(p.N_q==len(p.q_level_freqs)==len(p.q_level_freqs_time_dep)==len(p.q_level_dephasing_rate)==(len(p.q_level_decay_rate)+1))
    assert(p.non_rwa_terms==0 or p.non_rwa_terms==1)

    results=DictExtended()
    
    #keep a reference to the specific options/parameters that were used in the simulation along with the results 
    results.p=DictExtended(p)

    #qubit lowering operator
    c=q.tensor(q.destroy(p.N_q), *[q.qeye(p.N_m) for i in xrange(p.mode_count)])


    #mode lowering operators; a[j], gives the lowering operator for the jth mode (first is the 0th indexed)
    a=[]
    for j in xrange(p.mode_count):
        #prettier way to do this?
        temp_op_list=[q.qeye(p.N_q)]
        for i in xrange(p.mode_count):
            if i==j:
                temp_op_list.append(q.destroy(p.N_m))
            else:
                temp_op_list.append(q.qeye(p.N_m))

        a.append(q.tensor(*temp_op_list))

    #keep a reference to the key operators of our system
    results.op=DictExtended(a=a, c=c)

    #Build the total Hamiltonian of our system
    H_rest_list=[]
    for i in xrange(p.mode_count):
        H_rest_list.append(p.mode_freqs[i] * a[i].dag() * a[i])
        H_rest_list.append(p.mode_q_coupling_freqs[i] * (a[i].dag()*c + a[i]*c.dag()) + 
                        p.non_rwa_terms * p.mode_q_coupling_freqs[i] * (a[i].dag()*c.dag() + a[i]*c) )
    H_rest_static=np.sum(H_rest_list)

    H_q_static=q.tensor(q.Qobj(np.diag(p.q_level_freqs)), *[q.qeye(p.N_m) for i in xrange(p.mode_count)])
    H_q_t_list=[]
    #then time dependence... 
    for lvl in xrange(p.N_q):
        h_matrix=q.tensor(q.fock_dm(p.N_q,lvl), *[q.qeye(p.N_m) for i in xrange(p.mode_count)])
        H_q_t_list.append([h_matrix, p.q_level_freqs_time_dep[lvl](p.tlist, p.epsilon, p.omega_drive)])

    H_static=H_q_static + H_rest_static
    results.H=[H_static] + H_q_t_list
    # print(H)

    #Get the eigenvalues/vectors of static portion of the Hamiltonian
    eigen_system=H_static.eigenstates()
    results.eigen_vals=eigen_system[0]
    results.eigen_vecs=eigen_system[1]

    if p.print_extra_info:
        print("Eigen frequencies:")
        print(map(lambda item: "%0.9g" % item, (results.eigen_vals-results.eigen_vals[0])/(2.0*np.pi)))
        print("Consecutive differences in eigen frequencies:")
        print(map(lambda item: "%0.9g" % item, (results.eigen_vals[1:] - results.eigen_vals[:-1])/(2.0*np.pi)))
        print("Differences between mode and 1st excited qubit freqs")
        print((np.array(p.mode_freqs) - p.q_level_freqs[1])/(2.0*np.pi))

    if p.only_show_ev:
        #in case we only want to see the eigenvalues/extra_info
        return None

    #Here we generate a whole bunch of states that might be of interest... but later we'll only consider some subset of those that we want to plot
    #...this could be streamlined... in reality no need to create all these states if they are not to be used... but this portion of the code is fast anyway
    #WE ASSUME that the initial state is one of those states defined here
    #Generate the single excitation states in bare basis
    results.states_collection=generate_single_excitation_states(p)
    # print(states_collection)
    #add the ground state
    results.states_collection['state_0']=fock_state(p, 0,[])
    #add the systems's eigenstates
    eigenvectors_count=p.N_q*p.N_m*p.mode_count
    for m in xrange(eigenvectors_count):
        results.states_collection["state_v%d" % m]=results.eigen_vecs[m]*results.eigen_vecs[m].dag()
        # results.states_collection["state_pv%d" % m]=   results.eigen_vecs[m]*results.eigen_vecs[m].dag()
    #add some other fun states
    results.states_collection["g"]=q.tensor(q.fock_dm(p.N_q, 0), *[q.qeye(p.N_m) for i in xrange(p.mode_count)])
    results.states_collection["e"]=q.tensor(q.fock_dm(p.N_q, 1), *[q.qeye(p.N_m) for i in xrange(p.mode_count)])
    results.states_collection["sup_v1-v2"]=0.5 * (results.states_collection["state_v1"] + results.states_collection["state_v2"])

    #get the right states whose expectation values we want to plot
    results.e_ops=[results.states_collection[state_name] for state_name in p.states_to_evolve]

    #initial state
    results.rho0 = results.states_collection[p.rho0_name]

    results.c_op_list = []
    #qubit dephasing
    if np.count_nonzero(p.q_level_dephasing_rate)>0:
        dis_op=q.tensor(q.Qobj(np.diag( np.sqrt(2.0) * np.sqrt(p.q_level_dephasing_rate) ) ), *[q.qeye(p.N_m) for i in xrange(p.mode_count)])
        if p.dissipators_in_dressed_basis:
            dis_op=dis_op.transform(results.eigen_vecs, inverse=False)
            # dis_op=dis_op.transform(results.eigen_vecs, inverse=True)
        results.c_op_list.append(dis_op)
        #TESTING
        #reverse which state "feels" the phase shift
        # rev_dephasing=np.array(p.q_level_dephasing_rate)[::-1]
       # results.c_op_list.append(q.tensor(q.Qobj(np.diag( np.sqrt(2.0) * np.sqrt(rev_dephasing) ) ), *[q.ket2dm(q.basis(p.N_m,0)) for i in xrange(p.mode_count)]))
        # results.c_op_list.append(q.tensor(q.Qobj(np.diag( np.sqrt(2.0) * np.sqrt(rev_dephasing) ) ), *[q.qeye(p.N_m) for i in xrange(p.mode_count)]))
        # results.c_op_list.append(q.tensor(q.Qobj(np.diag( np.sqrt(2.0) * np.sqrt(p.q_level_dephasing_rate) ) ), *[q.ket2dm(q.basis(p.N_m,0)) for i in xrange(p.mode_count)]))
    #qubit decay
    if np.count_nonzero(p.q_level_decay_rate)>0:
        dis_op=q.tensor(q.Qobj(np.diag( np.sqrt(p.q_level_decay_rate), 1 ) ), *[q.qeye(p.N_m) for i in xrange(p.mode_count)])
        if p.dissipators_in_dressed_basis:
            dis_op=dis_op.transform(results.eigen_vecs, inverse=False)
            # dis_op=dis_op.transform(results.eigen_vecs, inverse=True)
        results.c_op_list.append(dis_op)
        #TESTING
        # results.c_op_list.append(q.tensor(q.Qobj(np.diag( np.sqrt(p.q_level_decay_rate), 1 ) ), *[q.ket2dm(q.basis(p.N_m,0)) for i in xrange(p.mode_count)]))
    #mode decay
    for i, entry in enumerate(p.mode_decay_rates):
        if entry!=0:
            dis_op=np.sqrt(entry) * a[i]
            if p.dissipators_in_dressed_basis:
                dis_op=dis_op.transform(results.eigen_vecs, inverse=False)
                # dis_op=dis_op.transform(results.eigen_vecs, inverse=True)
            results.c_op_list.append(dis_op)
    # print(results.c_op_list)

    results.output = q.mesolve(results.H, results.rho0, p.tlist, results.c_op_list, results.e_ops, options=p.odeoptions, progress_bar={False:None, True:True}[p.show_progress_bar])

    if p.show_approx_evolution:

        if p.N_q>2:
            #Need to generalize the approximate (i.e. rotating frame) treatment to more levels
            print("ERROR: The approximate (in terms of J_m functions) for now only handles a 2 level qudit...\n\n\n") 
            return None

        if p.omega_drive==0:
            #the rotated frame we've chosen assumes p.omega_drive is not zero, because we end up with a 1/p.omega_drive in the arugment
            #of the Bessel functions.
            print("ERROR: The approximate expression for the Hamiltonian (in terms of Bessel functions) assumes that omega_drive is not zero. \
                If the drive is not needed, simply set its amplitude to zero. Full numerics if of course possible with omega_drive==0.")
            return results

        #TODO: double check minus signs, etc and generalize to many qudit levels
        def Utrans(t): 
            return ( (+1.0j) *  (  (p.q_level_freqs[1] - p.q_level_freqs[0]) * t  - p.epsilon/(1.0 * p.omega_drive ) * np.cos(p.omega_drive * t) ) * c.dag()*c   +
                    np.sum([  (+1.0j) * t * p.mode_freqs[i] * a[i].dag()*a[i]  for i in xrange(p.mode_count)]) ).expm()

        results.Utrans=Utrans

        zero_ops=[q.Qobj(np.zeros((p.N_q, p.N_q)))] + [q.Qobj(np.zeros((p.N_m, p.N_m))) for i in xrange(p.mode_count)]
        zero_H=q.tensor(*zero_ops) 

        results.H_rf=[]
        results.H_rf.append(zero_H)  #dirrtyyy... seems like qutip wants at least one time independent Hamiltonian portion... just give it all zeros.
        for i in xrange(p.mode_count):
            ham_coeff_time_array=np.zeros(np.array(p.tlist).shape)
            for m in p.bessel_indices:
                ham_coeff_time_array=ham_coeff_time_array + ( (1.0j)**m * p.mode_q_coupling_freqs[i] * sc.special.jv(m, p.epsilon / ( 1.0 * p.omega_drive )) * np.exp( (1.0j) *  (p.mode_freqs[i] - (p.q_level_freqs[1] - p.q_level_freqs[0]) - m * p.omega_drive )  * p.tlist))

            matrix_components=[c*a[i].dag(), ham_coeff_time_array]
            results.H_rf.append(matrix_components)
            results.H_rf.append([matrix_components[0].dag(), np.conjugate(matrix_components[1])])
            
            if p.non_rwa_terms!=0:  #do the same but for the non-rwa terms
                ham_coeff_time_array=np.zeros(np.array(p.tlist).shape)
                for m in p.bessel_indices:
                    ham_coeff_time_array=ham_coeff_time_array + ( (1.0j)**m * p.mode_q_coupling_freqs[i] * sc.special.jv(m, p.epsilon / ( 1.0 * p.omega_drive )) * np.exp( (1.0j) *  (p.mode_freqs[i] + (p.q_level_freqs[1] - p.q_level_freqs[0]) - m * p.omega_drive )  * p.tlist))

                matrix_components=[c.dag()*a[i].dag(), ham_coeff_time_array]
                results.H_rf.append(matrix_components)
                results.H_rf.append([matrix_components[0].dag(), np.conjugate(matrix_components[1])])

        results.rho0_rf=Utrans(0)*results.rho0*Utrans(0).dag()

        #In general we have to be careful about how the dissipators transform in the rotated frame... 
        #take them as the same as in lab frame, which should be true in our case
        # results.c_op_list = []

        #If the "measurement operators" (i.e. ones we calculate the expectation values of) do not commute with the unitary transformation 
        #that defines the rotating frame, we need to transform the resulting density matrix before calculating expectation values if we 
        #want to compare the results to the full lab frame evolution

        if p.transform_before_measurement:

            def e_ops_func(t, rho, transformation=Utrans, e_ops=results.e_ops):
                """Transform the density matrix into the lab frame, and calculate the expectation values. 
                TODO: this could probably be streamlined... need to look into qutip's code
                """
                rho_lab_frame=Utrans(t).dag()*q.Qobj(rho)*Utrans(t)
                for i, e_operator in enumerate(e_ops):
                    expectation_values[i][e_ops_func.idx]=q.expect(e_operator, rho_lab_frame).real
                e_ops_func.idx+=1

            e_ops_func.idx=0
            expectation_values=[np.zeros(len(p.tlist)) for i in xrange(len(results.e_ops))]

            results.output_rf=q.mesolve(results.H_rf, results.rho0_rf, p.tlist, results.c_op_list, e_ops_func, options=p.odeoptions, progress_bar={False:None, True:True}[p.show_progress_bar])
            results.output_rf.expect=expectation_values

        else:
            results.output_rf=q.mesolve(results.H_rf, results.rho0_rf, p.tlist, results.c_op_list, results.e_ops, options=p.odeoptions, progress_bar={False:None, True:True}[p.show_progress_bar])

        print("p.epsilon / ( p.omega_drive )=%f, " %  (p.epsilon / ( 1.0 * p.omega_drive ), ) + 
                ",".join([ "J_%d()=%f" % (m, sc.special.jv(m,(p.epsilon / ( 1.0 * p.omega_drive )))) for m in p.bessel_indices]) ) 

    if p.show_plots: 
        plot_results(p, results)

    return results



# def main:
    # pass

if __name__ == '__main__':
    pass
    # main()

