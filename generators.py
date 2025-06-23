import random
import math
from decimal import InvalidOperation 

# Pre generated outputs for faster generation speed
FACTORIAL_MAP = {math.factorial(i): i for i in range(2, 25)}
FIBONACCI_MAP = {0: 0, 1: 1, 2: 3, 3: 4, 5: 5, 8: 6, 13: 7, 21: 8, 34: 9, 55: 10, 89: 11, 144: 12, 233: 13, 377: 14, 610: 15, 987: 16, 1597: 17, 2584: 18, 4181: 19, 6765: 20}

CONSTANTS = {"pi": math.pi, "e": math.e, "tau": math.tau, "phi": (1+math.sqrt(5))/2, "γ": 0.5772156649}

# ==============================================================================
#  Abstract Syntax Tree (AST) 
# ==============================================================================

class Node:
    def to_string(self): raise NotImplementedError

class NumberNode(Node):
    def __init__(self, value): self.value = float(value)
    def to_string(self):
        if self.value.is_integer():
            val_int = int(self.value)
            if random.random() < 0.2 and val_int >= 0: return random.choice([hex(val_int), oct(val_int), bin(val_int)])
            if random.random() < 0.1: return f"{self.value:.1e}"
            return str(val_int)
        return f"{self.value:.3f}".rstrip('0').rstrip('.')

class ConstantNode(Node):
    def __init__(self, name): self.name = name
    def to_string(self): return self.name

class UnaryOpNode(Node):
    def __init__(self, op_str, child): self.op_str, self.child = op_str, child
    def to_string(self):
        if self.op_str == '~': return f"~{self.child.to_string()}"
        if self.op_str == 'sqrt': return f"√({self.child.to_string()})"
        return f"{self.op_str}({self.child.to_string()})"

class BinaryOpNode(Node):
    def __init__(self, op_str, left, right): self.op_str, self.left, self.right = op_str, left, right
    def to_string(self):
        op_map = {'*': '×', '/': '÷', '^': '⊻'}
        display_op = op_map.get(self.op_str, self.op_str)
        if self.op_str == '**' and isinstance(self.right, NumberNode):
            if self.right.value == 2: return f"({self.left.to_string()})²"
            if self.right.value == 3: return f"({self.left.to_string()})³"
        return f"({self.left.to_string()}{display_op}{self.right.to_string()})"

class Func2Node(Node):
    def __init__(self, op_str, left, right): self.op_str, self.left, self.right = op_str, left, right
    def to_string(self):
        return f"{self.op_str}({self.left.to_string()},{self.right.to_string()})"


def _get_divisors(n: int) -> list[int]:
    """Returns a list of divisors for an integer n, used for lcm back-solving."""
    n = abs(n)
    if n < 2: return [1]
    divs = {1, n}
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            divs.add(i)
            divs.add(n//i)
    return list(divs)

# These functions are prioritized to create more interesting expressions.
PRIORITY_PATTERNS = [
    {'type': 'binary_func', 'op': 'gcd', 'node': Func2Node, 'domain': lambda t: t.is_integer() and t > 1, 'back_solve': lambda t: (lambda c: (int(t) * c[0], int(t) * c[1]))(random.choice([(2,3), (3,5)]))},
    {'type': 'binary_func', 'op': 'lcm', 'node': Func2Node, 'domain': lambda t: t.is_integer() and abs(t) > 1, 'back_solve': lambda t: (int(t), random.choice(_get_divisors(int(t))))},
    {'type': 'binary_func', 'op': 'log', 'node': Func2Node, 'domain': lambda t: 1 < t < 40, 'back_solve': lambda t: (lambda b: (b**t, b))(random.uniform(1.5, 8))}, # log(value, base)
    {'type': 'binary_func', 'op': 'atan2', 'node': Func2Node, 'domain': lambda t: -math.pi < t < math.pi and abs(t) > 0.1, 'back_solve': lambda t: (lambda r: (r * math.sin(t), r * math.cos(t)))(random.uniform(1, 100))}, # atan2(y,x)
    {'type': 'binary', 'op': 'hypot', 'node': BinaryOpNode, 'domain': lambda t: 10 < t < 1e10, 'back_solve': lambda t: (a := random.uniform(1, t-1), math.sqrt(t**2 - a**2))},
    {'type': 'binary', 'op': '&', 'node': BinaryOpNode, 'domain': lambda t: t.is_integer() and t > 0, 'back_solve': lambda t: ( (t_int := int(t)) | random.randint(1, t_int + 100), t_int )},
    {'type': 'binary', 'op': '|', 'node': BinaryOpNode, 'domain': lambda t: t.is_integer() and t > 0, 'back_solve': lambda t: (sub := int(t) & random.randint(0, int(t) * 2), int(t) ^ sub)},
]

# Standard operations for building expressions.
REGULAR_PATTERNS = [
    {'type': 'binary', 'op': '+', 'node': BinaryOpNode, 'domain': lambda t: True, 'back_solve': lambda t: (t - (a := random.uniform(1, 50)), a)},
    {'type': 'binary', 'op': '-', 'node': BinaryOpNode, 'domain': lambda t: True, 'back_solve': lambda t: (t + (a := random.uniform(1, 50)), a)},
    {'type': 'binary', 'op': '*', 'node': BinaryOpNode, 'domain': lambda t: abs(t) > 1e-6, 'back_solve': lambda t: (t / (a := random.uniform(2, 10)), a)},
    {'type': 'binary', 'op': '/', 'node': BinaryOpNode, 'domain': lambda t: abs(t) > 1e-6 and abs(t) < 1e50, 'back_solve': lambda t: (t * (a := random.uniform(2, 10)), a)},
    {'type': 'binary', 'op': '**', 'node': BinaryOpNode, 'domain': lambda t: t > 0 and t < 1e50, 'back_solve': lambda t: (t**(1/(p:=random.choice([2,3,4,5]))), p)},
    {'type': 'unary', 'op': 'sqrt', 'node': UnaryOpNode, 'domain': lambda t: t > 0 and t < 1e100, 'back_solve': lambda t: t**2},
    {'type': 'unary', 'op': 'cbrt', 'node': UnaryOpNode, 'domain': lambda t: abs(t) < 1e100, 'back_solve': lambda t: t**3},
    {'type': 'unary', 'op': 'abs', 'node': UnaryOpNode, 'domain': lambda t: t > 0, 'back_solve': lambda t: t * random.choice([-1, 1])},
    {'type': 'unary', 'op': 'floor', 'node': UnaryOpNode, 'domain': lambda t: t.is_integer(), 'back_solve': lambda t: t + random.random()},
    {'type': 'unary', 'op': 'ceil', 'node': UnaryOpNode, 'domain': lambda t: t.is_integer(), 'back_solve': lambda t: t - random.random()},
    {'type': 'unary', 'op': 'round', 'node': UnaryOpNode, 'domain': lambda t: t.is_integer(), 'back_solve': lambda t: t + random.uniform(-0.49, 0.49)},
    {'type': 'unary', 'op': 'trunc', 'node': UnaryOpNode, 'domain': lambda t: t.is_integer(), 'back_solve': lambda t: t + random.uniform(-0.99, 0.99) if t != 0 else random.random()},
    {'type': 'unary', 'op': 'rad', 'node': UnaryOpNode, 'domain': lambda t: abs(t) < 1e10, 'back_solve': math.degrees},
    {'type': 'unary', 'op': 'deg', 'node': UnaryOpNode, 'domain': lambda t: abs(t) < 1e10, 'back_solve': math.radians},
    {'type': 'unary', 'op': 'log', 'node': UnaryOpNode, 'domain': lambda t: 0 < t < 709, 'back_solve': math.exp},
    {'type': 'unary', 'op': 'log10', 'node': UnaryOpNode, 'domain': lambda t: 0 < t < 300, 'back_solve': lambda t: 10**t},
    {'type': 'unary', 'op': 'log2', 'node': UnaryOpNode, 'domain': lambda t: 0 < t < 1023, 'back_solve': lambda t: 2**t},
    {'type': 'unary', 'op': 'exp', 'node': UnaryOpNode, 'domain': lambda t: t > 1, 'back_solve': math.log},
    {'type': 'unary', 'op': 'exp2', 'node': UnaryOpNode, 'domain': lambda t: t > 1, 'back_solve': math.log2},
    {'type': 'unary', 'op': 'log1p', 'node': UnaryOpNode, 'domain': lambda t: 0 < t < 709, 'back_solve': math.expm1},
    {'type': 'unary', 'op': 'expm1', 'node': UnaryOpNode, 'domain': lambda t: t > -1 and t != 0, 'back_solve': math.log1p},
    {'type': 'unary', 'op': 'sin', 'node': UnaryOpNode, 'domain': lambda t: -1 <= t <= 1, 'back_solve': math.asin},
    {'type': 'unary', 'op': 'cos', 'node': UnaryOpNode, 'domain': lambda t: -1 <= t <= 1, 'back_solve': math.acos},
    {'type': 'unary', 'op': 'tan', 'node': UnaryOpNode, 'domain': lambda t: True, 'back_solve': math.atan},
    {'type': 'binary', 'op': '^', 'node': BinaryOpNode, 'domain': lambda t: t.is_integer(), 'back_solve': lambda t: (int(t) ^ (a := random.randint(1, 100)), a)},
    {'type': 'binary', 'op': '<<', 'node': BinaryOpNode, 'domain': lambda t: t.is_integer() and t > 0, 'back_solve': lambda t: (int(t) >> (s:=random.randint(1,4)), s)},
    {'type': 'binary', 'op': '>>', 'node': BinaryOpNode, 'domain': lambda t: t.is_integer() and t > 0 and t < 1e9, 'back_solve': lambda t: (int(t) << (s:=random.randint(1,4)), s)},
    {'type': 'binary', 'op': '//', 'node': BinaryOpNode, 'domain': lambda t: t.is_integer(), 'back_solve': lambda t: (random.randint(int(t*(b:=random.randint(2,10))), int(t*b+b-1)), b)},
    {'type': 'binary', 'op': '%', 'node': BinaryOpNode, 'domain': lambda t: t.is_integer() and t>=0, 'back_solve': lambda t: ((m:=random.randint(1,5))*(b:=random.randint(int(t)+1,int(t)+10)) + int(t), b)},
    {'type': 'unary', 'op': '~', 'node': UnaryOpNode, 'domain': lambda t: t <= -1 and t.is_integer(), 'back_solve': lambda t: -t - 1},
]

# These patterns increase complexity without changing the value.
PASSTHROUGH_PATTERNS = [
    {'op': 'asin(sin({}))'}, {'op': 'acos(cos({}))'}, {'op': 'atan(tan({}))'},
    {'op': 'asinh(sinh({}))'}, {'op': 'acosh(cosh({}))'}, {'op': 'atanh(tanh({}))'}
]

def _build_ast_tree(target: float, depth: int) -> Node:
    """Recursively builds an expression tree that evaluates to the target value."""
    target = float(target)
    if depth <= 0:
        return NumberNode(round(target, 10))

    # Handle simple cases first, but not always, to allow for more complex expressions.
    if target.is_integer():
        target_int = int(target)
        if target_int in FACTORIAL_MAP and random.random() < 0.5:
            return UnaryOpNode("fact", NumberNode(FACTORIAL_MAP[target_int]))
        if target_int in FIBONACCI_MAP and random.random() < 0.5:
            return UnaryOpNode("fibonacci", NumberNode(FIBONACCI_MAP[target_int]))

    # Choose a generation pattern. Prioritize advanced functions for deeper trees.
    candidates = []
    priority_chance = min(0.75, 0.15 + 0.05 * depth) # Chance increases with depth
    if random.random() < priority_chance:
        candidates = [p for p in PRIORITY_PATTERNS if p['domain'](target)]

    if not candidates:
        candidates = [p for p in REGULAR_PATTERNS if p['domain'](target)]

    for p in PASSTHROUGH_PATTERNS:
        op_name = p['op'].split('(')[0]
        candidates.append({'type': 'unary_passthrough', 'op': op_name, 'node': UnaryOpNode, 'back_solve': lambda t: t})
        
    if not candidates:
        return NumberNode(round(target, 10))
    
    chosen = random.choice(candidates)
    
    # --- Branching Logic ---
    if chosen['type'] in ('unary', 'unary_passthrough'):
        child_node = _build_ast_tree(chosen['back_solve'](target), depth - 1)
        return chosen['node'](chosen['op'], child_node)

    if chosen['type'] in ('binary', 'binary_func'):
        # Give a small chance to use a mathematical constant.
        if chosen['type'] == 'binary' and random.random() < 0.15:
            const_name, const_val = random.choice(list(CONSTANTS.items()))
            op = chosen['op']
            try:
                if op == '+': target_a = target - const_val
                elif op == '-': target_a = target + const_val
                elif op == '*': target_a = target / const_val
                elif op == '/': target_a = target * const_val
                elif op == '**': target_a = target**(1/const_val)
                else: # Fallback for ops that don't mix well with constants
                    target_a, target_b = chosen['back_solve'](target)
                    return chosen['node'](op, _build_ast_tree(target_a, depth - 1), _build_ast_tree(target_b, 0))
                
                left = _build_ast_tree(target_a, depth - 1); right = ConstantNode(const_name)
                # Randomly swap for non-commutative operations to increase variety.
                if op in ('-', '/', '**') and random.random() < 0.5:
                    if op == '-': target_a = const_val - target
                    elif op == '/': target_a = const_val / target
                    elif op == '**': target_a = math.log(target, const_val)
                    left = ConstantNode(const_name); right = _build_ast_tree(target_a, depth - 1)

                return chosen['node'](op, left, right)
            except (ValueError, OverflowError, ZeroDivisionError):
                pass
        
        target_a, target_b = chosen['back_solve'](target)
        depth_a = random.randint(0, depth - 1)
        depth_b = (depth - 1) - depth_a
        return chosen['node'](chosen['op'], _build_ast_tree(target_a, depth_a), _build_ast_tree(target_b, depth_b))

    return NumberNode(round(target, 10))

def _generate_ast_expression(final_number: int, complexity: int = None) -> str:
    if complexity is None: complexity = random.randint(6, 15)
    try:
        tree = _build_ast_tree(float(final_number), complexity)
        return tree.to_string()
    except (ValueError, OverflowError, ZeroDivisionError, TypeError):
        return str(final_number)

def _generate_fallback_expression(target: int) -> str:
    target = float(target)
    fallback_templates = [
        lambda t: f"(gamma({(a:=random.randint(4, 8))})+({t - math.gamma(a)}))",
        lambda t: f"(exp(lgamma({(a:=random.randint(4, 8))}))+({t - math.gamma(a)}))",
        lambda t: f"({t - (1/math.tan(a:=random.uniform(1,2)))}+cot({a}))"
    ]
    try:
        return random.choice(fallback_templates)(target)
    except (ValueError, OverflowError):
        return str(target)

# ==============================================================================
#  Legacy Expression Generator
# ==============================================================================

def _generate_legacy_expression(target: float, depth: int) -> str:
    """recursive string based generator initially used, now simply kept for comparison"""
    MAGNITUDE_LIMIT = 1e9
    target_is_integer = isinstance(target, int) or (abs(target - round(target)) < 1e-9)

    if depth <= 0:
        if target_is_integer:
            target_int = int(round(target))
            if 6 <= target_int <= 3_628_800 and target_int in FACTORIAL_MAP: return f"fact({FACTORIAL_MAP[target_int]})"
            if random.random() < 0.3: return random.choice([hex(target_int), oct(target_int), bin(target_int)])
            return str(target_int)
        return f"{target:.4f}".rstrip('0').rstrip('.')

    operations = []
    if abs(target) < MAGNITUDE_LIMIT**(1/3): operations.append({'type': 'unary', 'template': '(cbrt({}))', 'new_target': target**3})
    if target > 0 and target < MAGNITUDE_LIMIT**(1/2): operations.append({'type': 'unary', 'template': '(sqrt({}))', 'new_target': target**2})
    if 0 < target < 15:
        try: operations.append({'type': 'unary', 'template': '(log({}))', 'new_target': math.exp(target)})
        except OverflowError: pass
    
    trig_funcs = [{'template': '(asin(sin({})))'}, {'template': '(acos(cos({})))'}]
    if abs(math.cos(target)) > 1e-4: trig_funcs.append({'template': '(atan(tan({})))'})
    for op in trig_funcs: operations.append({'type': 'unary_passthrough', 'template': op['template'], 'new_target': target})

    if target_is_integer: operations.append({'type': 'unary', 'template': '(~({}))', 'new_target': -int(round(target)) - 1})

    a = random.uniform(-10, 20) if not target_is_integer else random.randint(1, int(target) + 20 if target > 0 else 20)
    operations.append({'type': 'binary', 'template': '({}+{})', 'parts': (target - a, a)})
    operations.append({'type': 'binary', 'template': '({}-{})', 'parts': (target + a, a)})

    if abs(target) > 1e-6:
        a = random.uniform(1, 20)
        if a != 0: operations.append({'type': 'binary', 'template': '({}*{})', 'parts': (target / a, a)})
        b = random.uniform(1, 10)
        if b != 0 and abs(target * b) < MAGNITUDE_LIMIT: operations.append({'type': 'binary', 'template': '({}/{})', 'parts': (target * b, b)})

    if not operations: return _generate_legacy_expression(target, 0)

    chosen_op = random.choice(operations)
    if chosen_op['type'] in ('unary', 'unary_passthrough'):
        return chosen_op['template'].format(_generate_legacy_expression(chosen_op['new_target'], depth - 1))
    if chosen_op['type'] == 'binary':
        p1, p2 = chosen_op['parts']
        depth_a = random.randint(1, depth - 1) if depth > 1 else 0
        expr_a = _generate_legacy_expression(p1, depth_a)
        expr_b = _generate_legacy_expression(p2, depth - 1 - depth_a)
        return chosen_op['template'].format(expr_a, expr_b)
    
    return str(target)


def generate_expression(final_number: int, mode: str, eval_func, round_func, complexity: int = None) -> tuple[str, dict]:
    metadata = {'mode': mode, 'used_fallback': False, 'fallback_succeeded': False}
    
    if mode == "AST":
        for attempt in range(200):
            try:
                expr = _generate_ast_expression(final_number, complexity)
                if expr == str(final_number): continue
                if round_func(eval_func(expr)) == final_number:
                    return expr, metadata
            except (ValueError, OverflowError, ZeroDivisionError, TypeError, SyntaxError, InvalidOperation):
                continue
        
        metadata['used_fallback'] = True
        for attempt in range(20):
            try:
                expr = _generate_fallback_expression(final_number)
                if round_func(eval_func(expr)) == final_number:
                    metadata['fallback_succeeded'] = True
                    return expr, metadata
            except (ValueError, OverflowError, ZeroDivisionError, TypeError, SyntaxError, InvalidOperation):
                continue
        
        return str(final_number), metadata
    
    elif mode == "Legacy":
        for attempt in range(50):
            try:
                expr = _generate_legacy_expression(float(final_number), complexity or random.randint(4, 8))
                if round_func(eval_func(expr)) == final_number:
                    return expr, metadata
            except (ValueError, OverflowError, ZeroDivisionError, TypeError, InvalidOperation):
                continue
        return str(final_number), metadata

    return str(final_number), metadata