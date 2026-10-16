"""
Model 2: Jackson Queueing Network & Staffing Optimization (M/M/c)
=================================================================
Models the IT incident support tiers (Level 1, Level 2, Level 3)
as an open Jackson Queueing Network.

Provides:
1. Analytical M/M/c formulas (Erlang-C, Wq, W, Lq, L, P(Wq > SLA))
2. Multi-tier staffing optimization (minimizing staffing cost subject to SLA & stability constraints)
3. SimPy Discrete-Event Simulation to validate queue dynamics under stochastic arrivals
"""

import math
import os
import json
import numpy as np
import pandas as pd
import simpy

DEFAULT_QUEUE_CONFIG = {
    'arrival_rate_lambda': 3.60, # tickets / hour arriving to L1
    'p12': 0.5847,               # escalation probability L1 -> L2
    'p23': 0.1476,               # escalation probability L2 -> L3
    'direct_l2_arrival': 0.39,   # tickets / hour arriving directly to L2
    'mu': {
        'L1': 0.70,              # effective tickets / hour / resolver
        'L2': 0.60,
        'L3': 0.50
    },
    'current_staffing': {
        'L1': 2, # David, Sam
        'L2': 3, # Emma, Michael, Sarah
        'L3': 4  # Ava, James, Olivia, William
    },
    'sla_target_wq_h': {
        'L1': 1.0,
        'L2': 1.5,
        'L3': 0.5
    },
    'hourly_resolver_cost': {
        'L1': 45.0, # USD / hour or cost weight
        'L2': 65.0,
        'L3': 90.0
    }
}

def erlang_c(c, a):
    """
    Computes Erlang-C formula: P(Wait > 0)
    c: number of servers (resolvers)
    a: offered load (lambda / mu)
    """
    if c <= a:
        return 1.0
    rho = a / c
    # Sum_{n=0}^{c-1} a^n / n!
    sum_terms = sum((a**n) / math.factorial(n) for n in range(c))
    last_term = (a**c) / (math.factorial(c) * (1.0 - rho))
    p0 = 1.0 / (sum_terms + last_term)
    pw = last_term * p0
    return float(min(1.0, max(0.0, pw)))

def compute_mmc_metrics(c, lam, mu, sla_target_h=1.0):
    """
    Calculates exact steady-state M/M/c queueing metrics.
    """
    if c <= 0 or mu <= 0:
        return None
    a = lam / mu
    rho = a / c
    
    if rho >= 1.0:
        return {
            'c': c, 'lam': lam, 'mu': mu, 'a': a, 'rho': rho,
            'is_stable': False,
            'P_wait': 1.0,
            'Lq': np.nan, 'Wq_hours': np.nan, 'W_hours': np.nan,
            'L': np.nan, 'sla_compliance': 0.0
        }
        
    pw = erlang_c(c, a)
    lq = (pw * rho) / (1.0 - rho)
    wq = lq / lam
    w = wq + (1.0 / mu)
    l = lam * w
    
    # Probability that wait time Wq exceeds SLA target:
    # P(Wq > t) = C(c, a) * e^{-(c*mu - lam)*t}
    excess_capacity = (c * mu) - lam
    p_exceed_sla = pw * math.exp(-excess_capacity * sla_target_h) if excess_capacity > 0 else 1.0
    sla_compliance = max(0.0, 1.0 - p_exceed_sla)
    
    return {
        'c': c,
        'lam': round(lam, 4),
        'mu': round(mu, 4),
        'a': round(a, 3),
        'rho': round(rho, 3),
        'is_stable': True,
        'P_wait': round(pw, 3),
        'Lq': round(lq, 3),
        'Wq_hours': round(wq, 3),
        'W_hours': round(w, 3),
        'L': round(l, 3),
        'sla_compliance': round(sla_compliance * 100.0, 1)
    }

class JacksonQueueNetwork:
    """Jackson Queueing Network Model for IT Incident Support Tiers."""
    
    def __init__(self, config=None, data_dir='.'):
        if config is None:
            # Check if empirical params exist
            params_file = os.path.join(data_dir, 'data_processed', 'queue_params.json')
            if os.path.exists(params_file):
                with open(params_file, 'r') as f:
                    p = json.load(f)
                config = DEFAULT_QUEUE_CONFIG.copy()
                config['arrival_rate_lambda'] = p.get('lambda_overall', DEFAULT_QUEUE_CONFIG['arrival_rate_lambda'])
                config['p12'] = p['routing'].get('P_12', DEFAULT_QUEUE_CONFIG['p12'])
                config['p23'] = p['routing'].get('P_23', DEFAULT_QUEUE_CONFIG['p23'])
            else:
                config = DEFAULT_QUEUE_CONFIG.copy()
                
        self.config = config
        
    def get_traffic_rates(self, lambda_mult=1.0):
        """
        Solves Jackson Network traffic equations for effective arrival rates Lambda_j.
        Lambda_1 = lambda
        Lambda_2 = direct_l2 + P12 * Lambda_1
        Lambda_3 = P23 * Lambda_2
        """
        lam1 = self.config['arrival_rate_lambda'] * lambda_mult
        lam2 = (self.config['direct_l2_arrival'] * lambda_mult) + (self.config['p12'] * lam1)
        lam3 = self.config['p23'] * lam2
        return {'L1': lam1, 'L2': lam2, 'L3': lam3}
        
    def evaluate_staffing(self, staffing_dict, lambda_mult=1.0):
        """Evaluates steady-state metrics for a given staffing vector (c1, c2, c3)."""
        traffic = self.get_traffic_rates(lambda_mult)
        results = {}
        total_cost = 0.0
        
        for tier in ['L1', 'L2', 'L3']:
            c = staffing_dict.get(tier, self.config['current_staffing'][tier])
            lam = traffic[tier]
            mu = self.config['mu'][tier]
            sla = self.config['sla_target_wq_h'][tier]
            cost_rate = self.config['hourly_resolver_cost'][tier]
            
            metrics = compute_mmc_metrics(c, lam, mu, sla_target_h=sla)
            results[tier] = metrics
            total_cost += c * cost_rate
            
        # Network-level summary
        all_stable = all(results[t]['is_stable'] for t in ['L1', 'L2', 'L3'])
        if all_stable:
            total_arrival = traffic['L1']
            weighted_wq = sum(traffic[t] * results[t]['Wq_hours'] for t in ['L1', 'L2', 'L3']) / total_arrival
            weighted_w = sum(traffic[t] * results[t]['W_hours'] for t in ['L1', 'L2', 'L3']) / total_arrival
            weighted_sla = sum(traffic[t] * results[t]['sla_compliance'] for t in ['L1', 'L2', 'L3']) / sum(traffic.values())
        else:
            weighted_wq = np.nan
            weighted_w = np.nan
            weighted_sla = 0.0
            
        return {
            'tier_metrics': results,
            'traffic_rates': traffic,
            'total_headcount': sum(staffing_dict[t] for t in ['L1', 'L2', 'L3']),
            'total_hourly_cost': total_cost,
            'network_mean_Wq_h': round(weighted_wq, 3) if not np.isnan(weighted_wq) else np.nan,
            'network_mean_W_h': round(weighted_w, 3) if not np.isnan(weighted_w) else np.nan,
            'network_sla_compliance': round(weighted_sla, 1),
            'all_stable': all_stable
        }
        
    def optimize_staffing(self, max_rho=0.82, min_sla_pct=85.0, lambda_mult=1.0):
        """
        Finds the cost-minimal staffing vector (c1*, c2*, c3*)
        satisfying stability and SLA targets.
        """
        traffic = self.get_traffic_rates(lambda_mult)
        optimal_c = {}
        
        for tier in ['L1', 'L2', 'L3']:
            lam = traffic[tier]
            mu = self.config['mu'][tier]
            sla_h = self.config['sla_target_wq_h'][tier]
            
            # Minimum servers for theoretical stability
            c_min = math.ceil(lam / mu)
            best_c = None
            
            for c in range(c_min, c_min + 25):
                met = compute_mmc_metrics(c, lam, mu, sla_target_h=sla_h)
                if (met['is_stable'] and 
                    met['rho'] <= max_rho and 
                    met['Wq_hours'] <= sla_h and 
                    met['sla_compliance'] >= min_sla_pct):
                    best_c = c
                    break
                    
            if best_c is None:
                best_c = c_min + 1
            optimal_c[tier] = best_c
            
        opt_eval = self.evaluate_staffing(optimal_c, lambda_mult)
        return optimal_c, opt_eval
        
    def generate_tradeoff_curves(self, lambda_mult=1.0):
        """Generates Wq vs c trade-off data for all tiers."""
        traffic = self.get_traffic_rates(lambda_mult)
        curves = {}
        
        for tier in ['L1', 'L2', 'L3']:
            lam = traffic[tier]
            mu = self.config['mu'][tier]
            sla_h = self.config['sla_target_wq_h'][tier]
            c_min = math.ceil(lam / mu)
            
            records = []
            for c in range(c_min, c_min + 8):
                met = compute_mmc_metrics(c, lam, mu, sla_target_h=sla_h)
                if met['is_stable']:
                    records.append({
                        'c': c,
                        'rho': met['rho'],
                        'Wq_hours': met['Wq_hours'],
                        'Lq': met['Lq'],
                        'sla_compliance': met['sla_compliance'],
                        'cost': c * self.config['hourly_resolver_cost'][tier]
                    })
            curves[tier] = pd.DataFrame(records)
        return curves

    def run_discrete_event_simulation(self, staffing_dict, sim_hours=1500, lambda_mult=1.0, seed=42):
        """
        Runs SimPy discrete-event simulation of the 3-tier queueing network.
        Validates analytical formulas against simulated queuing realizations.
        """
        np.random.seed(seed)
        env = simpy.Environment()
        
        traffic = self.get_traffic_rates(lambda_mult)
        lam1 = traffic['L1']
        lam2_direct = self.config['direct_l2_arrival'] * lambda_mult
        
        mu1 = self.config['mu']['L1']
        mu2 = self.config['mu']['L2']
        mu3 = self.config['mu']['L3']
        p12 = self.config['p12']
        p23 = self.config['p23']
        
        r1 = simpy.Resource(env, capacity=staffing_dict['L1'])
        r2 = simpy.Resource(env, capacity=staffing_dict['L2'])
        r3 = simpy.Resource(env, capacity=staffing_dict['L3'])
        
        wait_times = {'L1': [], 'L2': [], 'L3': []}
        service_times = {'L1': [], 'L2': [], 'L3': []}
        
        def handle_ticket(t_id, start_tier='L1'):
            # Tier 1
            if start_tier == 'L1':
                t_arr = env.now
                with r1.request() as req:
                    yield req
                    wq = env.now - t_arr
                    wait_times['L1'].append(wq)
                    s_dur = np.random.exponential(1.0 / mu1)
                    service_times['L1'].append(s_dur)
                    yield env.timeout(s_dur)
                    
                if np.random.rand() >= p12:
                    return # Solved at L1
                    
            # Tier 2
            t_arr2 = env.now
            with r2.request() as req:
                yield req
                wq2 = env.now - t_arr2
                wait_times['L2'].append(wq2)
                s_dur2 = np.random.exponential(1.0 / mu2)
                service_times['L2'].append(s_dur2)
                yield env.timeout(s_dur2)
                
            if np.random.rand() >= p23:
                return # Solved at L2
                
            # Tier 3
            t_arr3 = env.now
            with r3.request() as req:
                yield req
                wq3 = env.now - t_arr3
                wait_times['L3'].append(wq3)
                s_dur3 = np.random.exponential(1.0 / mu3)
                service_times['L3'].append(s_dur3)
                yield env.timeout(s_dur3)
                
        def l1_arrivals():
            t_id = 0
            while True:
                inter = np.random.exponential(1.0 / lam1)
                yield env.timeout(inter)
                t_id += 1
                env.process(handle_ticket(f"T-{t_id}", start_tier='L1'))
                
        def l2_direct_arrivals():
            t_id = 100000
            while True:
                inter = np.random.exponential(1.0 / max(0.01, lam2_direct))
                yield env.timeout(inter)
                t_id += 1
                env.process(handle_ticket(f"T-D2-{t_id}", start_tier='L2'))
                
        env.process(l1_arrivals())
        if lam2_direct > 0:
            env.process(l2_direct_arrivals())
            
        env.run(until=sim_hours)
        
        sim_metrics = {}
        for t in ['L1', 'L2', 'L3']:
            waits = wait_times[t]
            sla_target = self.config['sla_target_wq_h'][t]
            if len(waits) > 0:
                sim_metrics[t] = {
                    'count': len(waits),
                    'sim_mean_Wq_h': round(float(np.mean(waits)), 3),
                    'sim_median_Wq_h': round(float(np.median(waits)), 3),
                    'sim_p90_Wq_h': round(float(np.percentile(waits, 90)), 3),
                    'sim_sla_compliance': round(float(np.mean(np.array(waits) <= sla_target) * 100.0), 1)
                }
            else:
                sim_metrics[t] = {'count': 0, 'sim_mean_Wq_h': 0.0, 'sim_sla_compliance': 100.0}
                
        return sim_metrics

if __name__ == '__main__':
    jq = JacksonQueueNetwork()
    print("Default Traffic Rates:", jq.get_traffic_rates())
    print("\nCurrent Staffing Evaluation (L1=2, L2=3, L3=4):")
    eval_curr = jq.evaluate_staffing({'L1': 2, 'L2': 3, 'L3': 4})
    print(eval_curr)
    
    print("\nOptimizing Staffing:")
    opt_c, opt_eval = jq.optimize_staffing()
    print("Optimal Staffing:", opt_c)
    print("Optimal Evaluation:", opt_eval)
