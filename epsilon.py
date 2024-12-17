import ip_simple_query
import ip_full_query
import contextlib
import io

f = io.StringIO()
epsilon_list = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

def show_result(true_result, errors, dp_results):
    def format_number(num):
        if isinstance(num, int):
            return f'{num:,}'
        return num
    
    def format_row(epsilon, error, dp):
        return f"│ {epsilon:^7} │ {format_number(error):>11} │ {format_number(dp):>11} │"
   
    table_width = len(format_row("Epsilon", "Error", "DP Result")) - 2
    print(f"True Result: {format_number(true_result)}")
    print("┌" + "─" * (table_width) + "┐")
    print(format_row("Epsilon", "Error", "DP Result"))
    print("├" + "─" * 9 + "┼" + "─" * 13 + "┼" + "─" * 13 + "┤")
    for epsilon in sorted(errors.keys()):
        print(format_row(epsilon, errors[epsilon], dp_results[epsilon]))
    print("└" + "─" * (table_width) + "┘")

errors = {}
dp_results = {}
for epsilon in epsilon_list:
    with contextlib.redirect_stdout(f):
        true_result, error, dp_result = ip_simple_query.main(epsilon=epsilon, show_graph = False)
    print(f"epsilon: {epsilon}, true_result: {true_result}, error: {error}, dp_result: {dp_result}")
    errors[epsilon] = error
    dp_results[epsilon] = dp_result
print("\nSimple Query Result:")
show_result(true_result, errors, dp_results)

errors = {}
dp_results = {}
for epsilon in epsilon_list:
    with contextlib.redirect_stdout(f):
        true_result, error, dp_result = ip_full_query.main(epsilon=epsilon, show_graph = False)
    print(f"epsilon: {epsilon}, true_result: {true_result}, error: {error}, dp_result: {dp_result}")
    errors[epsilon] = error
    dp_results[epsilon] = dp_result
print("\nFull Query Result:")
show_result(true_result, errors, dp_results)
