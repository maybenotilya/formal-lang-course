from antlr4 import CommonTokenStream, ParserRuleContext, InputStream, ParseTreeWalker

from project.graphlang.parser.GraphLangLexer import GraphLangLexer
from project.graphlang.parser.GraphLangParser import GraphLangParser
from project.graphlang.parser.listeners import CountListener, TokensListener


def program_to_tree(program: str) -> tuple[ParserRuleContext, bool]:
    lexer = GraphLangLexer(InputStream(program))
    stream = CommonTokenStream(lexer)
    parser = GraphLangParser(stream)

    tree = parser.prog()

    return (tree, parser.getNumberOfSyntaxErrors() == 0)


def nodes_count(tree: ParserRuleContext) -> int:
    listener = CountListener()
    ParseTreeWalker().walk(listener, tree)

    return listener.count


def tree_to_program(tree: ParserRuleContext) -> str:
    listener = TokensListener()
    ParseTreeWalker().walk(listener, tree)

    return " ".join(listener.tokens)
