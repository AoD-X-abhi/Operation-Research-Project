"""
Model 1: Integer Linear Programming (ILP) Prescriptive Ticket Assignment
========================================================================
Formulates and solves the optimal ticket-to-resolver matching problem:
- Minimizes priority-weighted service time
- Maximizes expected customer satisfaction
- Enforces workload fairness (linearized variance/deviation penalty)
- Satisfies single assignment, resolver capacity, and tier routing constraints.
"""

import sys
import os
import numpy as np
import pandas as pd
from scipy.optimize import milp, LinearConstraint, Bounds

try:
    from src.ml_models import Predictor
except ImportError:
    from ml_models import Predictor

# Default active resolvers and initial capacities
DEFAULT_RESOLVER_CONFIG = {
    'David': {'tier': 'L1', 'capacity': 25, 'current_workload': 6},
    'Sam': {'tier': 'L1', 'capacity': 25, 'current_workload': 18},
    'Emma': {'tier': 'L2', 'capacity': 30, 'current_workload': 22},
    'Michael': {'tier': 'L2', 'capacity': 30, 'current_workload': 12},
    'Sarah': {'tier': 'L2', 'capacity': 30, 'current_workload': 5},
    'Ava': {'tier': 'L3', 'capacity': 15, 'current_workload': 2},
    'James': {'tier': 'L3', 'capacity': 15, 'current_workload': 8},
    'Olivia': {'tier': 'L3', 'capacity': 15, 'current_workload': 4},
    'William': {'tier': 'L3', 'capacity': 15, 'current_workload': 7},
}

PRIORITY_WEIGHTS = {
    'High': 3.0,
    'Medium': 2.0,
    'Low': 1.0
}

def calculate_gini(array):
    """
    Computes Gini Coefficient for workload distribution inequality.
    0 = Perfect equality, 1 = Complete inequality.
    """
    arr = np.array(array, dtype=np.float64)
    if np.all(arr == 0):
        return 0.0
    arr = np.sort(arr)
    n = len(arr)
    index = np.arange(1, n + 1)
    return float((np.sum((2 * index - n - 1) * arr)) / (n * np.sum(arr)))

class ILPTicketAssigner:
    """Integer Linear Programming Solver for incident dispatching."""
    
    def __init__(self, data_dir='.'):
        self.predictor = Predictor(data_dir=data_dir)
        
    def solve_assignment(self, tickets, resolver_config=None, alpha=1.0, beta=1.5, gamma=2.0):
        """
        Solves the ILP ticket assignment problem.
        
        Args:
            tickets: list of dicts with keys: ['id', 'tier', 'priority', 'issue_type', 'channel']
            resolver_config: dict of resolver properties (tier, capacity, current_workload)
            alpha: weight for priority-weighted service time
            beta: weight for expected CSAT score
            gamma: weight for workload variance/balance penalty
            
        Returns:
            dict containing:
                - solver_status: str
                - objective_value: float
                - assignments: list of dicts (ticket, resolver, pred_s, pred_c)
                - initial_workloads: dict
                - final_workloads: dict
                - gini_before: float
                - gini_after: float
                - total_pred_service_time: float
                - avg_pred_csat: float
        """
        if resolver_config is None:
            resolver_config = DEFAULT_RESOLVER_CONFIG
            
        resolvers = list(resolver_config.keys())
        resolver_tier = {r: resolver_config[r]['tier'] for r in resolvers}
        capacities = {r: resolver_config[r]['capacity'] for r in resolvers}
        init_workload = {r: resolver_config[r]['current_workload'] for r in resolvers}
        
        # Build candidate pairs (resolver in eligible tier for ticket)
        pairs = []
        for t in tickets:
            t_id = t['id']
            t_tier = t['tier']
            for r in resolvers:
                if resolver_tier[r] == t_tier:
                    pairs.append((r, t_id))
                    
        if not pairs:
            return {'solver_status': 'No eligible pairs', 'assignments': []}
            
        pair_idx = {pair: idx for idx, pair in enumerate(pairs)}
        n_x = len(pairs)
        n_r = len(resolvers)
        n_vars = n_x + n_r
        
        # Predict S_{i,k} and CSAT_{i,k} for each candidate pair
        pred_input = []
        for (r, t_id) in pairs:
            t = next(item for item in tickets if item['id'] == t_id)
            pred_input.append({
                'tier': resolver_tier[r],
                'resolver': r,
                'priority': t['priority'],
                'issue_type': t.get('issue_type', 'Incident'),
                'channel': t.get('channel', 'Website'),
                'hour_of_day': 12,
                'day_of_week': 2
            })
            
        pred_df = pd.DataFrame(pred_input)
        preds_s, preds_c = self.predictor.batch_predict(pred_df)
        
        S = {pair: preds_s[i] for i, pair in enumerate(pairs)}
        CSAT = {pair: preds_c[i] for i, pair in enumerate(pairs)}
        
        # Objective Vector c
        # min sum c_j * x_j
        c = np.zeros(n_vars)
        for pair, idx in pair_idx.items():
            r, t_id = pair
            t = next(item for item in tickets if item['id'] == t_id)
            pk = PRIORITY_WEIGHTS.get(t['priority'], 1.0)
            c[idx] = alpha * (pk * S[pair]) - beta * CSAT[pair]
            
        for i in range(n_r):
            c[n_x + i] = gamma
            
        # Integrality: binary for x, continuous for dev
        integrality = np.zeros(n_vars)
        integrality[:n_x] = 1 # binary x_{i,k}
        
        # Variable Bounds: x in [0, 1], dev in [0, inf)
        lb = np.zeros(n_vars)
        ub = np.full(n_vars, np.inf)
        ub[:n_x] = 1.0
        bounds = Bounds(lb, ub)
        
        # Constraints:
        A_rows = []
        lhs = []
        rhs = []
        
        # 1. Single assignment: sum_{r} x_{r,k} = 1 for each ticket k
        for t in tickets:
            t_id = t['id']
            row = np.zeros(n_vars)
            for r in resolvers:
                if (r, t_id) in pair_idx:
                    row[pair_idx[(r, t_id)]] = 1.0
            A_rows.append(row)
            lhs.append(1.0)
            rhs.append(1.0)
            
        # 2. Capacity: sum_{k} x_{r,k} <= Capacity[r] - Current[r]
        for r_idx, r in enumerate(resolvers):
            row = np.zeros(n_vars)
            for t in tickets:
                t_id = t['id']
                if (r, t_id) in pair_idx:
                    row[pair_idx[(r, t_id)]] = 1.0
            A_rows.append(row)
            rem_cap = max(0, capacities[r] - init_workload[r])
            lhs.append(0.0)
            rhs.append(rem_cap)
            
        # 3. Workload deviation (linearized balance penalty within tier)
        tier_res = {}
        for r in resolvers:
            tier_res.setdefault(resolver_tier[r], []).append(r)
            
        for tier, r_list in tier_res.items():
            t_list = [t['id'] for t in tickets if t['tier'] == tier]
            if len(r_list) > 1 and len(t_list) > 0:
                total_tier_load = sum(init_workload[r] for r in r_list) + len(t_list)
                avg_tier_load = total_tier_load / len(r_list)
                for r in r_list:
                    r_idx = resolvers.index(r)
                    
                    # dev[r] - sum_k x_{r,k} >= init[r] - avg
                    row1 = np.zeros(n_vars)
                    row1[n_x + r_idx] = 1.0
                    for t_id in t_list:
                        if (r, t_id) in pair_idx:
                            row1[pair_idx[(r, t_id)]] = -1.0
                    A_rows.append(row1)
                    lhs.append(init_workload[r] - avg_tier_load)
                    rhs.append(np.inf)
                    
                    # dev[r] + sum_k x_{r,k} >= avg - init[r]
                    row2 = np.zeros(n_vars)
                    row2[n_x + r_idx] = 1.0
                    for t_id in t_list:
                        if (r, t_id) in pair_idx:
                            row2[pair_idx[(r, t_id)]] = 1.0
                    A_rows.append(row2)
                    lhs.append(avg_tier_load - init_workload[r])
                    rhs.append(np.inf)
                    
        A = np.array(A_rows)
        constraints = LinearConstraint(A, lhs, rhs)
        
        # Solve with HiGHS MILP
        res = milp(c=c, integrality=integrality, bounds=bounds, constraints=constraints)
        
        if not res.success:
            return {
                'solver_status': f"Failed: {res.message}",
                'success': False,
                'assignments': []
            }
            
        # Extract results
        assigned_tickets = []
        final_workloads = init_workload.copy()
        
        for pair, idx in pair_idx.items():
            if res.x[idx] > 0.5:
                r, t_id = pair
                t = next(item for item in tickets if item['id'] == t_id)
                assigned_tickets.append({
                    'ticket_id': t_id,
                    'tier': t['tier'],
                    'priority': t['priority'],
                    'issue_type': t.get('issue_type', 'Incident'),
                    'assigned_resolver': r,
                    'pred_service_time_h': round(float(S[pair]), 2),
                    'pred_csat': round(float(CSAT[pair]), 2),
                    'priority_weight': PRIORITY_WEIGHTS.get(t['priority'], 1.0)
                })
                final_workloads[r] += 1
                
        assigned_df = pd.DataFrame(assigned_tickets).sort_values('ticket_id')
        
        gini_before = calculate_gini(list(init_workload.values()))
        gini_after = calculate_gini(list(final_workloads.values()))
        
        return {
            'solver_status': 'Optimal (HiGHS)',
            'success': True,
            'objective_value': float(res.fun),
            'assignments': assigned_df.to_dict(orient='records'),
            'initial_workloads': init_workload,
            'final_workloads': final_workloads,
            'capacities': capacities,
            'gini_before': gini_before,
            'gini_after': gini_after,
            'total_pred_service_h': float(assigned_df['pred_service_time_h'].sum()),
            'avg_pred_csat': float(assigned_df['pred_csat'].mean())
        }

def generate_sample_incoming_batch(n=25, seed=42):
    """Generates realistic sample tickets for batch assignment demonstration."""
    np.random.seed(seed)
    priorities = ['High', 'Medium', 'Low']
    prio_probs = [0.25, 0.50, 0.25]
    issue_types = ['Bug', 'Performance Issue', 'Feature Request', 'Incident', 'Service Request']
    channels = ['Website', 'Email', 'App', 'Phone']
    tiers = ['L1', 'L2', 'L3']
    tier_probs = [0.55, 0.35, 0.10]
    
    tickets = []
    for i in range(n):
        tickets.append({
            'id': f"INC-{1000 + i + 1}",
            'tier': np.random.choice(tiers, p=tier_probs),
            'priority': np.random.choice(priorities, p=prio_probs),
            'issue_type': np.random.choice(issue_types),
            'channel': np.random.choice(channels)
        })
    return tickets

if __name__ == '__main__':
    assigner = ILPTicketAssigner()
    sample_batch = generate_sample_incoming_batch(25)
    result = assigner.solve_assignment(sample_batch)
    print("Optimization Result Status:", result['solver_status'])
    print(f"Gini Coefficient: Before={result['gini_before']:.3f} -> After={result['gini_after']:.3f}")
    print("\nSample Assignments:")
    print(pd.DataFrame(result['assignments']).head(5))
