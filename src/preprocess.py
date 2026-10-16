"""
Data Preprocessing & Process Mining Trace Extraction
====================================================
Transforms the raw incident management event log into:
1. Case-level performance dataset (Cycle time, queue time, service time, SLA flags, CSAT)
2. Task-level dataset for predictive ML modeling (Service time & CSAT prediction)
3. Empirical queueing parameters (Arrival rates, service rates, escalation routing probabilities)
"""

import os
import json
import pandas as pd
import numpy as np

# SLA Targets (in hours) based on incident priority
SLA_TARGETS = {
    'High': 6.0,
    'Medium': 12.0,
    'Low': 24.0
}

# Resolver Tier Mapping
RESOLVER_TIERS = {
    'David': 'L1',
    'Sam': 'L1',
    'Emma': 'L2',
    'Michael': 'L2',
    'Sarah': 'L2',
    'Ava': 'L3',
    'James': 'L3',
    'Olivia': 'L3',
    'William': 'L3'
}

def load_and_clean_raw_log(csv_path='Incident_Management_CSV.csv'):
    """Load and perform initial cleaning and sorting on the event log."""
    print(f"Loading raw event log from: {csv_path}...")
    df = pd.read_csv(csv_path, sep=';')
    
    # Strip whitespace from string columns
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).str.strip()
        
    # Filter empty event rows
    df = df[df['Event'] != ''].copy()
    
    # Parse chronological timestamps
    print("Parsing timestamps and sorting events chronologically...")
    df['dt'] = pd.to_datetime(df['Timestamp'], format='%d/%m/%Y %H:%M')
    df = df.sort_values(['Case ID', 'dt']).reset_index(drop=True)
    
    return df

def extract_traces_and_features(df):
    """
    Reconstruct process traces by Case ID.
    Extract temporal features (Wq, S, T) and ML task records.
    """
    print("Reconstructing process traces across all cases...")
    
    cases_list = []
    task_records = []
    
    total_hours = (df['dt'].max() - df['dt'].min()).total_seconds() / 3600.0
    
    grouped = df.groupby('Case ID', sort=False)
    for case_id, group in grouped:
        events = group['Event'].tolist()
        dts = group['dt'].tolist()
        resolvers = group['Resolver'].tolist()
        
        priority = group['Priority'].iloc[0]
        issue_type = group['Issue Type'].iloc[0]
        channel = group['Report Channel'].iloc[0]
        csat = float(group['Customer Satisfaction'].iloc[0])
        variant = group['Variant'].iloc[0]
        
        start_time = dts[0]
        end_time = dts[-1]
        cycle_time = (end_time - start_time).total_seconds() / 3600.0
        sla_limit = SLA_TARGETS.get(priority, 12.0)
        sla_violated = int(cycle_time > sla_limit)
        
        total_queue_time = 0.0
        total_service_time = 0.0
        reassignments = 0
        resolvers_involved = set()
        
        for i in range(len(events) - 1):
            ev = events[i].lower()
            next_ev = events[i+1].lower()
            dt1 = dts[i]
            dt2 = dts[i+1]
            res = resolvers[i] if resolvers[i] != 'nan' and pd.notnull(resolvers[i]) else resolvers[i+1]
            dur = max(0.01, (dt2 - dt1).total_seconds() / 3600.0)
            
            # Queue time: Assignment to WIP
            if 'assigned' in ev and 'wip' in next_ev:
                total_queue_time += dur
                
            # Service time: WIP to next state (solved, escalation, etc.)
            elif 'wip' in ev:
                total_service_time += dur
                if res != 'nan' and pd.notnull(res):
                    resolvers_involved.add(res)
                    tier = RESOLVER_TIERS.get(res, 'L1')
                    task_records.append({
                        'case_id': case_id,
                        'tier': tier,
                        'resolver': res,
                        'priority': priority,
                        'issue_type': issue_type,
                        'channel': channel,
                        'hour_of_day': dt1.hour,
                        'day_of_week': dt1.dayofweek,
                        'service_time_h': dur,
                        'next_event': events[i+1],
                        'csat': csat
                    })
                    
            if 'escalat' in ev:
                reassignments += 1
                
        cases_list.append({
            'case_id': case_id,
            'variant': variant,
            'priority': priority,
            'issue_type': issue_type,
            'channel': channel,
            'csat': csat,
            'start_time': start_time,
            'end_time': end_time,
            'cycle_time_h': cycle_time,
            'queue_time_h': total_queue_time,
            'service_time_h': total_service_time,
            'sla_limit_h': sla_limit,
            'sla_violated': sla_violated,
            'reassignments': reassignments,
            'num_resolvers': len(resolvers_involved)
        })
        
    cases_df = pd.DataFrame(cases_list)
    tasks_df = pd.DataFrame(task_records)
    
    # Calculate empirical queueing parameters
    total_cases = len(cases_df)
    lambda_overall = total_cases / total_hours
    
    # Tier completion statistics
    l1_tasks = tasks_df[tasks_df['tier'] == 'L1']
    l2_tasks = tasks_df[tasks_df['tier'] == 'L2']
    l3_tasks = tasks_df[tasks_df['tier'] == 'L3']
    
    p12 = float(l1_tasks['next_event'].str.contains('escalat', case=False).mean())
    p23 = float(l2_tasks['next_event'].str.contains('level 3', case=False).mean())
    
    # Effective completion rates (tickets/h per resolver active)
    # Using reciprocal of mean service duration
    mean_s1 = float(l1_tasks['service_time_h'].mean())
    mean_s2 = float(l2_tasks['service_time_h'].mean())
    mean_s3 = float(l3_tasks['service_time_h'].mean())
    
    queue_params = {
        'total_cases': total_cases,
        'total_hours': float(total_hours),
        'lambda_overall': float(lambda_overall),
        'routing': {
            'P_12': p12,
            'P_23': p23
        },
        'tier_service_times_mean_h': {
            'L1': mean_s1,
            'L2': mean_s2,
            'L3': mean_s3
        },
        'tier_service_rates_mu': {
            'L1': 1.0 / mean_s1,
            'L2': 1.0 / mean_s2,
            'L3': 1.0 / mean_s3
        },
        'sla_violation_rate': float(cases_df['sla_violated'].mean()),
        'mean_cycle_time_h': float(cases_df['cycle_time_h'].mean()),
        'mean_csat': float(cases_df['csat'].mean())
    }
    
    return cases_df, tasks_df, queue_params

def run_preprocessing(data_dir='.'):
    """Full execution function to preprocess and persist data."""
    raw_path = os.path.join(data_dir, 'Incident_Management_CSV.csv')
    out_dir = os.path.join(data_dir, 'data_processed')
    os.makedirs(out_dir, exist_ok=True)
    
    df = load_and_clean_raw_log(raw_path)
    cases_df, tasks_df, queue_params = extract_traces_and_features(df)
    
    print(f"Processed {len(cases_df):,} cases and {len(tasks_df):,} task records.")
    
    cases_csv = os.path.join(out_dir, 'cases_summary.csv')
    tasks_csv = os.path.join(out_dir, 'tasks_summary.csv')
    params_json = os.path.join(out_dir, 'queue_params.json')
    
    cases_df.to_csv(cases_csv, index=False)
    tasks_df.to_csv(tasks_csv, index=False)
    with open(params_json, 'w') as f:
        json.dump(queue_params, f, indent=2)
        
    print(f"Saved artifacts to {out_dir}:")
    print(f"  - {cases_csv}")
    print(f"  - {tasks_csv}")
    print(f"  - {params_json}")
    
    return cases_df, tasks_df, queue_params

if __name__ == '__main__':
    run_preprocessing()
