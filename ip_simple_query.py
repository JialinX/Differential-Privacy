import os
import pandas as pd
import sqlite3
import numpy as np
import cplex
import matplotlib.pyplot as plt

sql_query_init = """
SELECT 
    c_custkey,
    COUNT(*) as count_lineitem
FROM 
    customer,
    orders,
    lineitem
WHERE 
    c_custkey = o_custkey
    AND l_orderkey = o_orderkey
    AND c_custkey <= 1500
group by c_custkey
order by c_custkey;
"""

def read_DB():
    current_dir = os.getcwd()
    folder_path = os.path.join(current_dir, 'TPC-H_DB', 'tbl')

    table_names = ["customer", "lineitem", "nation", "orders", "part", "partsupp", "region", "supplier"]
    column_names = {
        "customer": ["c_custkey", "c_name", "c_address", "c_nationkey", "c_phone", "c_acctbal", "c_mktsegment", "c_comment"],
        "lineitem": ["l_orderkey", "l_partkey", "l_suppkey", "l_linenumber", "l_quantity", "l_extendedprice", "l_discount", "l_tax", "l_returnflag", "l_linestatus", "l_shipdate", "l_commitdate", "l_receiptdate", "l_shipinstruct", "l_shipmode", "l_comment"],
        "nation": ["n_nationkey", "n_name", "n_regionkey", "n_comment"],
        "orders": ["o_orderkey", "o_custkey", "o_orderstatus", "o_totalprice", "o_orderdate", "o_orderpriority", "o_clerk", "o_shippriority", "o_comment"],
        "part": ["p_partkey", "p_name", "p_mfgr", "p_brand", "p_type", "p_size", "p_container", "p_retailprice", "p_comment"],
        "partsupp": ["ps_partkey", "ps_suppkey", "ps_availqty", "ps_supplycost", "ps_comment"],
        "region": ["r_regionkey", "r_name", "r_comment"],
        "supplier": ["s_suppkey", "s_name", "s_address", "s_nationkey", "s_phone", "s_acctbal", "s_comment"]
    }

    conn = sqlite3.connect(":memory:")
    for table_name in table_names:
        file_path = os.path.join(folder_path, table_name + ".tbl")
        df = pd.read_csv(file_path, sep='|', header=None, names=column_names[table_name] + ["extra"], engine='python')
        
        if "extra" in df.columns:
            df = df.drop(columns=["extra"])
        
        df.to_sql(table_name, conn, index=False, if_exists='replace')
        print(f"Loaded {table_name} table into SQLite with shape: {df.shape}")
    
    return conn

def linear_problem(tau, result, count_lineitem, num_rows):
    problem = cplex.Cplex()
    problem.objective.set_sense(problem.objective.sense.maximize)

    variable_names = [f"U_{i}" for i in range(num_rows)]
    lower_bounds = [0] * num_rows 
    upper_bounds = count_lineitem 
    objective_coefficients = [1] * num_rows  # Maximize sum(Ui)

    problem.variables.add(obj=objective_coefficients,
                        lb=lower_bounds,
                        ub=upper_bounds,
                        names=variable_names)

    unique_custkeys = result["c_custkey"].unique()
    for custkey in unique_custkeys:
        indices = result[result["c_custkey"] == custkey].index.tolist()
        coefficients = [1] * len(indices)
        constraint_name = f"sum_U_{custkey}_leq_tau"
        # Add constraint: sum(Ui for custkey) <= tau
        problem.linear_constraints.add(
            lin_expr=[cplex.SparsePair(ind=indices, val=coefficients)],
            senses=["L"], 
            rhs=[tau], 
            names=[constraint_name],
        )

    problem.solve()
    return problem

def dp(problem, result, tau, gsq, epsilon, beta):
    solution_values = problem.solution.get_values()
    result["Ui"] = solution_values
    max_Ui_sum = sum(result["Ui"])
    log2_GSQ = np.log2(gsq)
    b = log2_GSQ * tau / epsilon
    noise = np.random.laplace(loc=0.0, scale=b)
    bias = log2_GSQ * np.log(log2_GSQ / beta) * (tau / epsilon)
    dp_result = max_Ui_sum + noise - bias
    return (int(max_Ui_sum), int(dp_result))

def draw_graph(true_result, ui_sum_result, best_dp_result):
    x_values = list(best_dp_result.keys())
    y1_values = list(best_dp_result.values())
    y2_values = list(ui_sum_result.values())

    tau_max = max(best_dp_result, key=best_dp_result.get)
    value_max = best_dp_result[tau_max] 

    plt.figure(figsize=(10, 6))
    plt.axhline(y=true_result, color='black', linestyle='dotted', linewidth=1, label=r"$Q(I)$")
    plt.scatter(x_values, y1_values, color='red', label=r"$\tilde{Q}(I, \tau)$") 
    plt.scatter(x_values, y2_values, color='blue', label=r"$Q(I, \tau)$")  

    plt.annotate(r"$\tilde{Q}(I)$", 
                xy=(tau_max, value_max-500), 
                xytext=(0, -40), 
                textcoords="offset points", 
                ha='center', 
                arrowprops=dict(arrowstyle='->', color='black'))

    plt.xscale('log', base=2)
    plt.xticks(x_values, labels=[str(i) for i in x_values])
    plt.xlabel(r"$\tau$", fontsize=12)
    plt.ylabel("sum of line item", fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.grid(visible=True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

def main(gsq = 10**6, epsilon = 1, beta = 0.1, num_round = 100, show_graph = False):
    conn = read_DB()
    result = pd.read_sql_query(sql_query_init, conn)
    print(result)
    conn.close()

    count_lineitem = result["count_lineitem"].tolist()
    true_result = sum(count_lineitem)
    num_rows = len(result)

    tau_list = [2**i for i in range(1, int(np.log2(gsq)))]
    dp_sum = 0
    linear_problems = {}
    for tau in tau_list:
        linear_problems[tau] = linear_problem(tau, result, count_lineitem, num_rows)

    for r in range(1, num_round + 1):
        best_dp_result = {}
        ui_sum_result = {}
        for tau in tau_list:
            problem = linear_problems[tau]
            max_Ui_sum, dp_result = dp(problem, result, tau, gsq, epsilon, beta)
            dp_result = max(dp_result, 0)
            ui_sum_result[tau] = max_Ui_sum
            best_dp_result[tau] = dp_result

            print("Updated DataFrame with Ui:\n", result)
            print("tau: ", tau)
            print("true result:    ", int(true_result))
            print("max Ui sum:     ", int(max_Ui_sum))
            print("result under DP:", int(dp_result))
            print("\n\n")

            if dp_result == 0:
                break

        highest_dp = max(best_dp_result.values())
        error = true_result - highest_dp
        dp_sum += highest_dp
        print(f"-------------Result (round {r})----------------------")
        print("true result: ", int(true_result))
        print("max Ui sum:\n", ui_sum_result)
        print("result (dict) under DP:\n", best_dp_result)
        print("result under DP: ", highest_dp)
        print("error: ", error)
        print("-----------------------------------------------------")
    
    avg_dp_result = round(dp_sum/num_round)
    avg_error = int(true_result) - avg_dp_result
    print(f"final result (average between {num_round} rounds):")
    print("true result: ", int(true_result))
    print("error: ", avg_error)
    print("DP output: ", avg_dp_result)
    if show_graph:
        draw_graph(int(true_result), ui_sum_result, best_dp_result)
    return int(true_result), avg_error, avg_dp_result

if __name__ == "__main__":
    main(gsq = 10**6, epsilon = 1, beta = 0.1, num_round = 100, show_graph = True)

