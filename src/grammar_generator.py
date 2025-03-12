from collections import defaultdict
import os
from typing import Dict, List, Set

from lark import Lark, Token, Tree
from paths import GRAMMARS_DIR


class GrammarGenerator:
    def __init__(self, path: str, variant: str):
        grammar_file = os.path.join(GRAMMARS_DIR, path)
        self.parser = Lark.open(grammar_file, start="call", parser="earley")
        fn_map = {
            "minimal_grammar": self.generate_minimal_grammar
        }
        self.generate_fn = fn_map[variant]

    @staticmethod
    def merge_quoted_strings(lst: List[str]) -> List[str]:
        merged_list = []
        temp = ""

        for item in lst:
            if item.startswith('"') and item.endswith('"'):
                temp += item[1:-1]
            else:
                if temp:
                    merged_list.append(f'"{temp}"')
                    temp = ""
                merged_list.append(item)

        if temp:
            merged_list.append(f'"{temp}"')

        return merged_list
    
    @staticmethod
    def convert_to_bnf(grammar_dict: Dict[str, Set[str]]) -> str:
        bnf_lines = []
        
        for non_terminal, productions in grammar_dict.items():
            bnf_line = f"{non_terminal} ::= " + " | ".join(productions)
            bnf_lines.append(bnf_line)
        
        return "\n".join(bnf_lines)

    def generate_minimal_grammar(self, program: str) -> str:
        tree = self.parser.parse(program)
        
        rule_dict = defaultdict(set)

        def traverse(node):
            if isinstance(node, Tree):  # If it's a rule (non-terminal)
                rule_name = node.data
                child_values = []
                for child in node.children:
                    if isinstance(child, Tree):  
                        child_values.append(child.data)
                    elif isinstance(child, Token):
                        child_values.append(f'"{child.value}"')
                    else:
                        raise ValueError(f"Unexpected type: {type(child)}")

                child_values = GrammarGenerator.merge_quoted_strings(child_values)
                concatenated_value = " ".join(child_values)
                rule_dict[rule_name].add(concatenated_value)

                for child in node.children:
                    traverse(child)

        traverse(tree)
        return GrammarGenerator.convert_to_bnf(rule_dict)
    
    def generate(self, program: str) -> str:
        return self.generate_fn(program)