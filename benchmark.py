import time
import random
import re
from collections import Counter
import generators
from main import evaluate_expression, round_half_up

def analyze_expression_functions(expr: str) -> list[str]:
    return re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*(?=\()', expr)

def run_benchmark(generator_mode: str, num_runs: int, target_numbers: list, complexities: list):
    print(f"\n--- Running Benchmark for Generator Mode: '{generator_mode}' ---")
    print(f"Configuration: {num_runs} runs, targets in {target_numbers}, complexities in {complexities}")
    
    successes, total_failures = 0, 0
    fallback_attempts, fallback_successes = 0, 0
    total_time = 0.0
    function_counter = Counter()

    for i in range(num_runs):
        target = random.choice(target_numbers)
        complexity = random.choice(complexities) if complexities else None
        
        start_time = time.perf_counter()
        expr, metadata = generators.generate_expression(
            final_number=target, mode=generator_mode,
            eval_func=evaluate_expression, round_func=round_half_up,
            complexity=complexity
        )
        end_time = time.perf_counter()
        total_time += (end_time - start_time)

        is_successful = (expr != str(target))
        if is_successful:
            successes += 1
            function_counter.update(analyze_expression_functions(expr))
        else:
            total_failures += 1
            
        if metadata.get('used_fallback'):
            fallback_attempts += 1
            if metadata.get('fallback_succeeded'):
                fallback_successes += 1

    print("\n--- BENCHMARK RESULTS ---")
    success_rate = (successes / num_runs) * 100
    print(f"Overall Success Rate: {success_rate:.2f}% ({successes}/{num_runs})")

    if generator_mode == 'AST':
        fallback_rate = (fallback_attempts / num_runs) * 100
        print(f"Fallback Trigger Rate: {fallback_rate:.2f}% ({fallback_attempts}/{num_runs})")
        if fallback_attempts > 0:
            fallback_success_rate = (fallback_successes / fallback_attempts) * 100
            print(f"  - Fallback Success Rate: {fallback_success_rate:.2f}% ({fallback_successes}/{fallback_attempts})")
    
    avg_time = (total_time / num_runs) * 1000
    print(f"\nPerformance:")
    print(f"  - Average Generation Time: {avg_time:.2f} ms")
    print(f"  - Total Time for {num_runs} runs: {total_time:.2f} seconds")

    print("\nTop 15 Most Used Functions:")
    if not function_counter:
        print("  - No functions were used in successful generations.")
    else:
        for i, (func, count) in enumerate(function_counter.most_common(15)):
            print(f"  {i+1:2d}. {func:<10s}: {count} times")
    print("-" * 25)

if __name__ == "__main__":
    TARGET_NUMBERS = list(range(2, 100)) + list(range(1000, 1100)) + list(range(100000, 100100))
    COMPLEXITIES = list(range(5, 16))
    NUM_RUNS = 500

    run_benchmark(
        generator_mode='AST',
        num_runs=NUM_RUNS,
        target_numbers=TARGET_NUMBERS,
        complexities=COMPLEXITIES
    )
    run_benchmark(
        generator_mode='Legacy',
        num_runs=NUM_RUNS,
        target_numbers=TARGET_NUMBERS,
        complexities=COMPLEXITIES
    )