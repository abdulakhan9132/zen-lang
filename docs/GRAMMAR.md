# ZEN Formal Grammar (EBNF)

A **grammar** is a set of rules describing every valid sentence in a programming language. Below is the Extended Backus–Naur Form (EBNF) grammar for ZEN.

> Notation:
> - `UPPERCASE` = token produced by the lexer (e.g. `IDENT`, `NUMBER`, `STRING`)
> - `"..."` = exact keyword or symbol
> - `|` = choice, `[ ... ]` = optional (0 or 1), `{ ... }` = repeat (0 or more)

```ebnf
program        = { statement } EOF ;

statement      = let_stmt
               | const_stmt
               | say_stmt
               | assert_stmt
               | assign_stmt
               | fn_decl
               | class_decl
               | import_stmt
               | if_stmt
               | while_stmt
               | for_stmt
               | return_stmt
               | break_stmt
               | continue_stmt
               | expr_stmt ;

let_stmt       = "let" IDENT [ ":" IDENT ] "=" expression [ ";" ] ;
const_stmt     = "const" IDENT [ ":" IDENT ] "=" expression [ ";" ] ;
say_stmt       = "say" expression { "," expression } [ ";" ] ;
assert_stmt    = "assert" expression [ "," expression ] [ ";" ] ;
assign_stmt    = postfix "=" expression [ ";" ] ;

fn_decl        = "fn" IDENT "(" [ param_list ] ")" [ "->" IDENT ] block ;
param_list     = param { "," param } ;
param          = IDENT [ ":" IDENT ] ;

class_decl     = "class" IDENT "{" { fn_decl } "}" ;

import_stmt    = "import" STRING [ "as" IDENT ] [ ";" ] ;

if_stmt        = "if" expression block
                 { "elif" expression block }
                 [ "else" block ] ;

while_stmt     = "while" expression block ;
for_stmt       = "for" IDENT "in" expression block ;

return_stmt    = "return" [ expression ] [ ";" ] ;
break_stmt     = "break" [ ";" ] ;
continue_stmt  = "continue" [ ";" ] ;
expr_stmt      = expression [ ";" ] ;

block          = "{" { statement } "}" ;

(* Expressions in increasing order of precedence *)
expression     = logic_or ;
logic_or       = logic_and { "or" logic_and } ;
logic_and      = equality { "and" equality } ;
equality       = comparison { ( "==" | "!=" ) comparison } ;
comparison     = term { ( "<" | "<=" | ">" | ">=" ) term } ;
term           = factor { ( "+" | "-" ) factor } ;
factor         = power { ( "*" | "/" | "%" | "@" ) power } ;
power          = unary [ "**" power ] ;
unary          = ( "-" | "not" ) unary | postfix ;

postfix        = primary { call_suffix | index_suffix | member_suffix } ;
call_suffix    = "(" [ arg_list ] ")" ;
index_suffix   = "[" expression "]" ;
member_suffix  = "." IDENT ;
arg_list       = expression { "," expression } ;

primary        = NUMBER
               | STRING
               | "true"
               | "false"
               | "none"
               | IDENT
               | fn_expr
               | "(" expression ")"
               | list_literal
               | dict_literal ;

fn_expr        = "fn" "(" [ param_list ] ")" block ;

list_literal   = "[" [ expression { "," expression } [ "," ] ] "]" ;
dict_literal   = "{" [ dict_entry { "," dict_entry } [ "," ] ] "}" ;
dict_entry     = expression ":" expression ;
```

## Lexical Tokens
- `IDENT`: `[a-zA-Z_][a-zA-Z0-9_]*` (excluding reserved keywords)
- `NUMBER`: `[0-9]+("."[0-9]+)?`
- `STRING`: `"` followed by characters or escape sequences (`\"`, `\\`, `\n`, `\t`), ending with `"`
- `COMMENT`: `//` until end of line (discarded by lexer)
- Whitespace (` `, `\t`, `\r`, `\n`): separates tokens and is discarded
