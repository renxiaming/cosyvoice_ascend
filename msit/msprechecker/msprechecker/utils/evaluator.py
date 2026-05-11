# Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import operator
from typing import Any

from .version import Version
from .log import global_logger


def and_(a, b):
    "Same as a and b."
    return a and b


def or_(a, b):
    "Same as a or b."
    return a or b


class Evaluator:
    OPS = {
        # 一元运算符
        'not': (7, operator.not_),  # 最高优先级
        
        # 算术运算符
        '**': (6, operator.pow),    # 右结合
        '*': (5, operator.mul), '/': (5, operator.truediv),
        '//': (5, operator.floordiv), '%': (5, operator.mod),
        '+': (4, operator.add), '-': (4, operator.sub),
        
        # 比较运算符 (所有同级)
        '==': (3, operator.eq), '!=': (3, operator.ne),
        '>': (3, operator.gt), '<': (3, operator.lt),
        '>=': (3, operator.ge), '<=': (3, operator.le),
        
        # 逻辑/位运算符
        '&': (2, operator.and_), 'and': (2, and_),
        '|': (1, operator.or_), 'or': (1, or_)
    }

    # only cares about int, str and float
    _FUNC_REGEX = re.compile(
        r'(?P<FUNC>float|int|str)\((?P<FUNC_ARG>[}{)(-./\$\w\']+)\)'
    )

    _TOKEN_REGEX = re.compile(
        r'Version\((?P<VERSION_ARG>[\w.]+)\)'
        r'|(?P<STR>\'[^\']+\')'
        r'|(?P<NUMBER>-?\b\d+(\.\d*)?\b)' # add boundary to avoid match numbers in path
        r'| (?P<OP>==|!=|>=|<=|\*\*|//|>|<|and|or|not|[+\-*/%]) ' # add space to avoid match path delimiter /
        r'|(?P<PARENTHESE>[)(])'
        r'|(?P<NONE>\bNone\b)'
        r'|(?P<SKIP>\s+)'
    )
    
    _FUNC_MAP = {
        'float': float, 'int': int,
        'str': str, 'Version': Version
    }
    
    # max nesting depth
    MAX_DEPTH = 5

    @classmethod
    def evaluate(cls, expr: str):
        if isinstance(expr, list):
            return [cls.evaluate(ex) for ex in expr]
        elif isinstance(expr, dict):
            return {k: cls.evaluate(ex) for k, ex in expr.items()}
        elif isinstance(expr, str):
            return cls._evaluate(expr)
        else:
            return expr

    @classmethod
    def _evaluate(cls, expr: Any):
        try:
            expr = Evaluator._evaluate_nesting_func(expr)
        except Exception:
            global_logger.warning("Invalid expression: %s. Skipped", expr)
            return True

        tokens = cls._split_tokens(expr)
        if not tokens:
            global_logger.warning("Invalid expression: %s. Did you mean: %r", expr, expr)
            return False

        rpn = cls._convert_tokens_to_rpn(tokens)
        try:
            result = cls._evaluate_rpn(rpn)
        except (ValueError, TypeError, ZeroDivisionError):
            return expr
        return result

    @classmethod
    def _evaluate_nesting_func(cls, expr: str, depth: int = 0) -> str:
        """
        Evaluate fuctions nesting with max depth limiting to 5

        Only allows int, str and float nesting to each other.
        string type should to denote as single quote, and should
        strip single quote before evaluating it.
        """
        if depth >= cls.MAX_DEPTH:
            raise RuntimeError(f"Maximum recursion depth {cls.MAX_DEPTH} exceeded")

        replacements = dict()
        for mo in cls._FUNC_REGEX.finditer(expr):
            func_name = mo.group('FUNC')
            if func_name not in cls._FUNC_MAP:
                raise ValueError(f"Unknown function: {func_name}")

            func_arg = mo.group('FUNC_ARG')            
            func_arg = cls._evaluate_nesting_func(func_arg, depth + 1)
            if isinstance(func_arg, str) and \
                func_arg.startswith("'") and \
                func_arg.endswith("'"):
                func_arg = func_arg.strip("'")
            result = cls._FUNC_MAP[func_name](func_arg)

            span = mo.span()
            if isinstance(result, str):
                replacements[span] = repr(result)
            elif isinstance(result, Version):
                replacements[span] = result
            else:
                replacements[span] = f"{result}"

        if replacements:
            parts = []
            last_pos = 0
            for (start, end), repl in replacements.items():
                parts.append(f"{expr[last_pos:start]}{repl}")
                last_pos = end
            parts.append(expr[last_pos:])
            expr = ''.join(parts)

        return expr
    
    @classmethod
    def _split_tokens(cls, expr: str):
        tokens = []
        for mo in cls._TOKEN_REGEX.finditer(expr):
            kind = mo.lastgroup
            value = mo.group(kind)

            if kind == 'SKIP':
                continue
            elif kind == 'VERSION_ARG':
                func_arg = mo.group('VERSION_ARG')
                tokens.append(("VERSION", Version(func_arg)))
            elif kind == 'STR':
                tokens.append(('STR', value.strip("'")))
            else:
                tokens.append((kind, value))

        return tokens
    
    @classmethod
    def _convert_tokens_to_rpn(cls, tokens):
        output = []
        stack = []

        for token in tokens:
            typ, val = token
            if typ == "NUMBER":
                output.append(float(val) if '.' in val else int(val))
            elif typ == "PARENTHESE":
                cls._process_parenthese(val, stack, output)
            elif typ == "OP":
                cls._process_op(val, stack, output)
            elif typ == "NONE":
                output.append(None)
            else:
                output.append(val)
    
        while stack:
            if stack[-1] in ('(', ')'):
                raise ValueError("Mismatched parentheses in expression")
            output.append(stack.pop())

        return output
    
    @classmethod
    def _process_parenthese(cls, val, stack, output):
        """ 
        If val is '(', push to stack; otherwise pop all ops in stack between the previous ')'
        """
        if val == '(':
            stack.append(val)
            return

        while stack and stack[-1] != '(':
            output.append(stack.pop())
        if not stack:
            raise ValueError("Mismatched parentheses in expression")
        stack.pop()

    @classmethod
    def _process_op(cls, op, stack, output):
        """
        Each op has its precedence and function, pop all the op in the stack where its prec
        is higher than the current op (that means they calculate first), and push current op
        to stack
        """
        prec = cls.OPS[op][0]
        while stack and stack[-1] in cls.OPS and cls.OPS[stack[-1]][0] >= prec:
            output.append(stack.pop()) # pop last op to output and push itself to stack
        stack.append(op)

    @classmethod
    def _evaluate_rpn(cls, rpn):
        stack = []
        for token in rpn:
            if isinstance(token, Version):
                stack.append(token)
            elif token == "not": # unary
                a = stack.pop()
                stack.append(cls.OPS[token][1](a))
            elif token in cls.OPS:
                b = stack.pop()
                a = stack.pop()
                stack.append(cls.OPS[token][1](a, b))
            else:
                stack.append(token)
        if len(stack) != 1:
            raise ValueError(f"Invalid RPN evaluation: {rpn}")
        return stack[0]
