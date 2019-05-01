#!/usr/bin/env python
#Sarah Denny and Winston Becker

# Import modules
import scipy
import numpy as np
import pandas as pd
import sys
import os
import ipdb
import itertools

kb = 0.0019872041 # kcal/(mol deg K)
kelvin = 273.15 # Kelvin at 0 deg C

def load_params(model_param_basename='annotations/RNAmap/qMotif_20180302_'):
    """Load the model parameters based on the basename"""
    base_params     = pd.read_csv(model_param_basename + 'term1.csv', index_col=0) #.stack().values
    flip_params     = pd.read_csv(model_param_basename + 'term2_single.csv', index_col=0) #.stack().values
    dflip_params    = pd.read_csv(model_param_basename + 'term2_double.csv', index_col=0, squeeze=True) #.stack().values, 10])
    coupling_params = pd.read_csv(model_param_basename + 'term3.csv', index_col=0, squeeze=True) #.stack().values
    
    return flip_params, base_params, coupling_params, dflip_params
        

def get_ddG_conversion(temperature):
    return -(temperature+kelvin)*kb

def perfect_match_ddG_coupling_terms(sequence, base_penalties, coupling_terms, first_coupling, second_coupling):
    # Inputs:
    # first_coupling--set to True if first (7G) coupling should be included if the conditions are met (this is included so that this can be set to false
    # when flips occur in the coupling region, which will likely prevent coupling)
    # second_coupling--set to True if second (7C) coupling should be included if the necessary conditions are met
    # base_penalties--base penalties in partition function space (exp(-ddG_base/kT))
    # coupling_terms--coupling penalties in partition function space (exp(-ddG_base/kT))
    # sequence register that the mutational penalty contribution is being computed for
    # Output: ddG transformed into partition function space for passed sequence

    # Initialize to a penalty of 0 kcal: 
    ddG = 0
    # Iterate through base penalties
    for i, base in enumerate(sequence):
        if i==8 and sequence[7]!='A':
            # exception at position 8--nothing happens
            continue
        ddG = ddG + base_penalties.loc[i, base]

    # Apply coupling corrections
    if first_coupling:
        if ((sequence[4] == 'U' or sequence[4] == 'C') and
            sequence[5] == 'A' and
            sequence[6] == 'G' and
            sequence[7] != 'A'):
            ddG = ddG + coupling_terms.loc['c1']
    if second_coupling:
        if sequence[5] != 'A' and sequence[6] == 'C' and sequence[7] != 'A':
            ddG = ddG + coupling_terms.loc['c2']

    return ddG


def compute_ensemble_ddG_set(single_dG_values, temperature):
    """Same as below but better starting with an array. Also assumes inputs are in dG,
    not in 'partition function space'"""
    ddG_conversion_factor = get_ddG_conversion(temperature)
    return ddG_conversion_factor*np.log(np.exp(single_dG_values/ddG_conversion_factor).sum(axis=1))
    


def compute_ensemble_ddG(single_dG_values, temperature, needs_exponentiating=False):
    # sums the individual contributions ot the partition function to get the compute partition 
    # function and then converts that into a ddG for the ensemble
    # Inputs:
    # single_dG_values--a list of the single contributions to the partition function from all possible registers
    # Outputs:
    # final_ddG--final ddG of the ensemble
    ddG_conversion_factor = get_ddG_conversion(temperature)

    if needs_exponentiating:
        single_dG_values = np.exp(single_dG_values/ddG_conversion_factor).copy()

    # Sum the logged ddG values to compute the partition function        
    partition_function = np.sum(single_dG_values)

    # Convert the partition function to the ensemble free energy
    
    final_ddG = ddG_conversion_factor*np.log(partition_function)

    return final_ddG

def get_coupling_bool_term1(flip_pos):
    # oonly apply the first coupling term if there is no flip at position 4, 5
    if flip_pos==4 or flip_pos==5 or flip_pos==6:
        return False
    else:
        return True
    
def get_coupling_bool_term2(flip_pos):
    # oonly apply the second coupling term if there is no flip at position 5
    if flip_pos==5 or flip_pos==6:
        return False
    else:
        return True
    
def get_register_key(i, seq_length, flip_pos=None, n_flip=1):
    """Return a tuple key, with first entry giving the sequence register and
    second the flipping configuration"""
    # return sequence register
    key1 = '%d:%d'%(i, i+seq_length)
    
    if flip_pos is None:
        key2 = '-'
    else:
        # make sure flip pos is a list if it is not None
        if not isinstance(flip_pos, list):
            raise TypeError('flip_pos must be a list')
      
        # make sure n_flip is also a list
        if isinstance(n_flip, int) or isinstance(n_flip, float):
            # n_flip is a scalar
            n_flip = [n_flip]*len(flip_pos)
        
        # return position and num flips for each flip
        key2 = ';'.join(['pos%d_%dnt'%(pos, n) for n, pos in zip(n_flip, flip_pos)])
        
    return key1, key2


def get_noflip_registers(sequence, base_penalties, coupling_params):
    """for a sequence, find the ddGs for each 1 nt register of the no-flip binding configuration."""
    seq_length = 9
    registers = {}
    for i in range(len(sequence)-seq_length+1):
        ddG = perfect_match_ddG_coupling_terms(sequence[i:i+seq_length], base_penalties, coupling_params, True, True)
        registers[get_register_key(i, seq_length)] = ddG
    registers = pd.Series(registers)
    return registers

def get_1flip_registers(sequence, base_penalties, coupling_params, flip_params):
    """for a sequence, find the ddGs for each 1 nt register of the 1nt-flip binding configuration."""
    possible_flip_positions = flip_params.index.tolist()
    seq_length = 10
    registers = {}
    for i in range(len(sequence)-seq_length+1):
        current_sequence = sequence[i:i+seq_length]
        for flip_pos in possible_flip_positions:
            seq_not_flipped = current_sequence[:flip_pos]+current_sequence[flip_pos+1:]
            flip_base = current_sequence[flip_pos]

            dG = (flip_params.loc[flip_pos, flip_base] +  # this is the penalty of flipping the residue
                  perfect_match_ddG_coupling_terms(seq_not_flipped, base_penalties, coupling_params,
                                                   get_coupling_bool_term1(flip_pos),
                                                   get_coupling_bool_term2(flip_pos)))
            registers[get_register_key(i, seq_length, [flip_pos], n_flip=1)] = dG
    registers = pd.Series(registers)
    return registers

def get_2flip_registers(sequence, base_penalties, coupling_params, flip_params, double_flip_params):
    """for a sequence, find the ddGs for each 1 nt register of the 2nt-flip binding configuration."""
    # double flips
    possible_flip_positions = flip_params.index.tolist()
    possible_double_flip_pos = double_flip_params.index.tolist()
    seq_length = 11
    registers = {}
    for i in range(len(sequence)-seq_length+1):
        current_sequence = sequence[i:i+seq_length]
        # 2x1nt flips
        for flip_pos1, flip_pos2 in itertools.combinations(possible_flip_positions, 2):
            

            # find the sequence without the two bases fliped at flip_pos1 and flip_pos2
            # watch for off by one errors
            seq_not_flipped = (current_sequence[:flip_pos1] +
                               current_sequence[flip_pos1+1:flip_pos2+1] +
                               current_sequence[flip_pos2+2:])
            flip_base1 = current_sequence[flip_pos1]
            flip_base2 = current_sequence[flip_pos2+1]
            
            dG = (flip_params.loc[flip_pos1, flip_base1] + # this is the penalty of flipping the residue 1
                  flip_params.loc[flip_pos2, flip_base2] + # this is the penalty of flipping the residue 2
                  perfect_match_ddG_coupling_terms(seq_not_flipped, base_penalties, coupling_params,
                                                   get_coupling_bool_term1(flip_pos1) and get_coupling_bool_term1(flip_pos2),
                                                   get_coupling_bool_term2(flip_pos1) and get_coupling_bool_term2(flip_pos2)))
            
            
            registers[get_register_key(i, seq_length, [flip_pos1, flip_pos2], n_flip=1)] = dG

        
        # 1x2nt flips
        for flip_pos in possible_double_flip_pos:

            seq_not_flipped = current_sequence[:flip_pos+1]+current_sequence[flip_pos+3:] 
            dG = (double_flip_params.loc[flip_pos] +
                  perfect_match_ddG_coupling_terms(seq_not_flipped, base_penalties, coupling_params,
                                                   get_coupling_bool_term1(flip_pos),
                                                   get_coupling_bool_term2(flip_pos)))

            registers[get_register_key(i, seq_length, [flip_pos], n_flip=2)] = dG

    registers = pd.Series(registers)
    return registers


def get_start_and_stop(seq_length, interval_length, i):
    """Return the stop and start around nt i."""
    start = max(i-interval_length+1, 0)
    stop = min(i+1, seq_length - interval_length+1)
    return start, stop
        
def find_energy_for_1nt_sequence_registers(sequence, base_penalties, coupling_params, flip_params, double_flip_params, temperature):
    """Find the ensemble energy for each 1 nt register"""
    linear_binding_ddGs = get_noflip_registers(sequence, base_penalties, coupling_params)
    oneflip_binding_ddGs = get_1flip_registers(sequence, base_penalties, coupling_params, flip_params)
    twoflip_binding_ddGs = get_2flip_registers(sequence, base_penalties, coupling_params, flip_params, double_flip_params)

    seq_length_linear = 9
    seq_length_oneflip = 10
    seq_length_twoflip = 11
    
    ddGs_final = {}
    for i in range(len(sequence)):
        ddGs = {}
        
        # linear binding
        #if seq_length_linear > seq_length_interval:
        #    raise ValueError('sequence is too short')

        start, stop = get_start_and_stop(len(sequence), seq_length_linear, i)
        for j in range(start, stop):
            key = '%d:%d'%(j, j+seq_length_linear)
            ddGs[key] = linear_binding_ddGs.loc[key]
        
          
        start, stop = get_start_and_stop(len(sequence), seq_length_oneflip, i)
        for j in range(start, stop):
            key = '%d:%d'%(j, j+seq_length_oneflip)
            ddGs[key] = oneflip_binding_ddGs.loc[key]
            
        start, stop = get_start_and_stop(len(sequence), seq_length_twoflip, i)
        for j in range(start, stop):
            key = '%d:%d'%(j, j+seq_length_twoflip)
            ddGs[key] = twoflip_binding_ddGs.loc[key]            

        try:
            ddGs = pd.concat(ddGs)
        except ValueError:
            ipdb.set_trace()
        # combine into a single ensemble ddG


        ddG = compute_ensemble_ddG(ddGs, temperature, needs_exponentiating=True)
        ddGs_final[i] = ddG
        
    return pd.Series(ddGs_final)    
    



