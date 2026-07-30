import re
import ast
import math
import operator

from numbers import Real

#-=-=-=-#

def _parse_number(value: str | Real) -> int | float | str:
	"""
	Converts numeric-ish strings and expressions into numbers.

	Examples
	--------
		"2e2"         -> 200
		"22.01k"      -> 22010
		"5"           -> 5
		"2.5"         -> 2.5
		"1.25k"       -> 1250
		"2**2"        -> 4
		"2+2"         -> 4
		"(2+3)*4"     -> 20
		"pi"          -> 3.14159...
		"sin(pi/2)"   -> 1
		"sqrt(44100)" -> 210
		"text"        -> text
	"""

	unitmap = {
		"k": 3,
		"m": 6,
		"g": 9,
	}

	def optimize(number: float) -> int | float:
		return int(number) if number.is_integer() else number

	def evaluate(expr: str) -> float:
		operators = {
			ast.Add: operator.add,
			ast.Sub: operator.sub,
			ast.Mult: operator.mul,
			ast.Div: operator.truediv,
			ast.Pow: operator.pow,
			ast.Mod: operator.mod,
			ast.USub: operator.neg,
			ast.UAdd: operator.pos,
		}

		constants = {
			"pi": math.pi,
			"e": math.e,
			"tau": math.tau,
		}

		functions = {
			# trigonometry (radians)
			"sin": math.sin,
			"cos": math.cos,
			"tan": math.tan,

			# inverse trigonometry
			"asin": math.asin,
			"acos": math.acos,
			"atan": math.atan,
			"atan2": math.atan2,

			# logarithmic/exponential
			"log": math.log,
			"log10": math.log10,
			"exp": math.exp,

			# misc
			"sqrt": math.sqrt,
			"abs": abs,
			"floor": math.floor,
			"ceil": math.ceil,

			# degree helpers
			"deg": math.radians,
			"rad": math.radians,
		}

		def walk(node):
			if isinstance(node, ast.Constant):
				if isinstance(node.value, (int, float)):
					return node.value

			if isinstance(node, ast.Name):
				if node.id in constants:
					return constants[node.id]

			if isinstance(node, ast.BinOp):
				if type(node.op) in operators:
					return operators[type(node.op)](
						walk(node.left),
						walk(node.right),
					)

			if isinstance(node, ast.UnaryOp):
				if type(node.op) in operators:
					return operators[type(node.op)](
						walk(node.operand)
					)

			if isinstance(node, ast.Call):
				if (
					isinstance(node.func, ast.Name)
					and node.func.id in functions
				):
					args = [
						walk(arg)
						for arg in node.args
					]

					return functions[node.func.id](*args)

			raise ValueError("Unsupported expression")

		tree = ast.parse(expr, mode = "eval")

		return walk(tree.body)

	# already numeric
	if isinstance(value, Real):
		return optimize(float(value))

	value = str(value).strip().lower()

	# allow calculator-style grouping
	value = value.replace("[", "(").replace("]", ")")

	# suffix units first
	match = re.fullmatch(
		rf"([0-9]*\.?[0-9]+)([{''.join(unitmap)}])",
		value,
	)

	if match:
		number, suffix = match.groups()
		return optimize(
			float(number) * (10 ** unitmap[suffix])
		)

	# plain number
	try:
		return optimize(float(value))
	except ValueError:
		pass

	# expression
	try:
		return optimize(float(evaluate(value)))
	except (ValueError, SyntaxError, TypeError, ZeroDivisionError):
		return value