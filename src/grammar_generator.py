from abc import ABC, abstractmethod
from collections import defaultdict
import os
from typing import Dict, List, Set

from lark import Lark, ParseTree, Token, Tree

from paths import get_dataset_name, get_grammar_path


class GrammarGenerator(ABC):
    def __init__(self):
        grammar_path = get_grammar_path()
        start = {
            "blocks": "list_value",
            "smcalflow": "call",
            "geoquery": "query"
        }
        dataset_name = get_dataset_name()
        self.parser = Lark.open(grammar_path, start=start[dataset_name], parser="earley")

    @staticmethod
    def create(variant: str) -> "GrammarGenerator":
        class_map = {
            "minimal": Minimal,
            "semi-minimal": SemiMinimal,
            "minimal-abstract": MinimalAbstract,
        }
        return class_map[variant]()

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
    
    @abstractmethod
    def _generate(self, tree: ParseTree) -> Dict[str, Set[str]]:
        raise NotImplementedError("Override me!")
    
    def generate(self, program: str) -> str:
        tree = self.parser.parse(program)
        rule_dict = self._generate(tree)
        return GrammarGenerator.convert_to_bnf(rule_dict)
    

class Minimal(GrammarGenerator):
    def token_to_string(self, token: Token) -> str:
        return f'"{token.value}"'
    
    def _generate(self, tree: ParseTree) -> Dict[str, Set[str]]:
        rule_dict = defaultdict(set)

        def traverse(node):
            if isinstance(node, Tree):  # If it's a rule (non-terminal)
                rule_name = node.data
                child_values = []
                for child in node.children:
                    if isinstance(child, Tree):
                        child_values.append(child.data)
                    elif isinstance(child, Token):
                        child_values.append(self.token_to_string(child))
                    else:
                        raise ValueError(f"Unexpected type: {type(child)}")

                child_values = GrammarGenerator.merge_quoted_strings(child_values)
                concatenated_value = " ".join(child_values)
                rule_dict[rule_name].add(concatenated_value)

                for child in node.children:
                    traverse(child)

        traverse(tree)
        return rule_dict
    

class MinimalAbstract(Minimal):
    def token_to_string(self, token: Token) -> str:
        if token.type in ["ESCAPED_STRING", "NUMBER"]:
            return f'{token.type}'
        return f'"{token.value}"'


class SemiMinimal(GrammarGenerator):
    def expansion_to_string(self, expansion: Tree) -> str:
        productions = []
        for value in expansion.children:
            value = value.children[0]
            if isinstance(value, Tree):
                value = value.children[0]

            if isinstance(value, Token):
                productions.append(f"{value.value}")
            else:
                productions.append(value.name)
        return " ".join(productions)

    def _generate(self, tree: ParseTree) -> Dict[str, Set[str]]:
        used_rules = []
        def traverse(node):
            if isinstance(node, Tree):
                rule_name = node.data.value
                used_rules.append(rule_name)

                for child in node.children:
                    traverse(child)
        
        traverse(tree)

        used_rules = list(dict.fromkeys(used_rules))
        rule_dict = {}

        for rule in self.parser.grammar.rule_defs:
            rule_name = rule[0].value
            expansions = rule[2]

            rule_variants = []
            for expansion in expansions.children:
                rule_variants.append(self.expansion_to_string(expansion))

            rule_dict[rule_name] = rule_variants

        return {rule: set(rule_dict[rule]) for rule in used_rules}