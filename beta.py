import ip_simple_query
import ip_full_query
import contextlib
import io

f = io.StringIO()
beta_list = [0.05, 0.10, 0.15, 0.20]

def show_result(true_result, errors, dp_results):
    def format_number(num):
        if isinstance(num, int):
            return f'{num:,}'
        return num
    
    def format_row(beta, error, dp):
        return f"│ {beta:^7} │ {format_number(error):>11} │ {format_number(dp):>11} │"
   
    table_width = len(format_row("Beta", "Error", "DP Result")) - 2
    print(f"True Result: {format_number(true_result)}")
    print("┌" + "─" * (table_width) + "┐")
    print(format_row("Beta", "Error", "DP Result"))
    print("├" + "─" * 9 + "┼" + "─" * 13 + "┼" + "─" * 13 + "┤")
    for beta in sorted(errors.keys()):
        print(format_row(beta, errors[beta], dp_results[beta]))
    print("└" + "─" * (table_width) + "┘")

errors = {}
dp_results = {}
for beta in beta_list:
    with contextlib.redirect_stdout(f):
        true_result, error, dp_result = ip_simple_query.main(beta=beta, show_graph = False)
    print(f"beta: {beta}, true_result: {true_result}, error: {error}, dp_result: {dp_result}")
    errors[beta] = error
    dp_results[beta] = dp_result
print("\nSimple Query Result:")
show_result(true_result, errors, dp_results)

errors = {}
dp_results = {}
for beta in beta_list:
    with contextlib.redirect_stdout(f):
        true_result, error, dp_result = ip_full_query.main(beta=beta, show_graph = False)
    print(f"beta: {beta}, true_result: {true_result}, error: {error}, dp_result: {dp_result}")
    errors[beta] = error
    dp_results[beta] = dp_result
print("\nFull Query Result:")
show_result(true_result, errors, dp_results)
