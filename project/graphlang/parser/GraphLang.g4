grammar GraphLang;

prog: stmt* EOF;

stmt: bind | add | remove | declare;

declare: 'let' VAR 'is' 'graph';

bind: 'let' VAR '=' expr;

remove:
	'remove' ('vertex' | 'edge' | 'vertices') expr 'from' VAR;

add: 'add' ('vertex' | 'edge') expr 'to' VAR;

expr: NUM | CHAR | VAR | edge_expr | set_expr | regexp | select;

set_expr: '[' expr (',' expr)* ']';

edge_expr: '(' expr ',' expr ',' expr ')';

// Regexp without left recursion

regexp: term ('|' term)*;
term: factor (('.' | '&') factor)*;
factor: base ('^' range)*;
base: CHAR | VAR | '(' regexp ')';

range: '[' NUM '..' NUM? ']';

select:
	v_filter? v_filter? 'return' VAR (',' VAR)? 'where' VAR 'reachable' 'from' VAR 'in' VAR 'by'
		expr;

v_filter: 'for' VAR 'in' expr;

VAR: [a-z] [a-z0-9]*;
NUM: [0] | [1-9] [0-9]*;
CHAR: '"' [a-z] '"' | '\'' [a-z] '\'';


WS: [ \t\r\n]+ -> skip;
EOF_TOKEN: '<EOF>' -> skip;