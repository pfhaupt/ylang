use std::fs;
use std::env;
use std::collections::{HashMap, VecDeque};

const KEYWORDS: &[&str] = &["let", "f", "while"];

#[derive(Debug, PartialEq, Clone, Copy)]
enum Assoc {
    Left,
    Right
}

#[allow(unused)]
#[derive(Debug, PartialEq)]
enum Token<'s> {
    Ident(&'s str),
    Number(&'s str),
    Keyword(&'s str),
    StrLit(&'s str), // Stored without quotes
    OpenParen,
    ClosingParen,
    OpenCurly,
    ClosingCurly,
    OpenSharp,
    ClosingSharp,
    Equal,
    Plus,
    Minus,
    Slash,
    Asterisk,
    Comma
}

impl<'s> std::fmt::Display for Token<'s> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        match self {
            Self::Number(n) => write!(f, "{n}"),
            Self::Ident(i) => write!(f, "{i}"),
            Self::Keyword(k) => write!(f, "{k}"),
            Self::StrLit(s) => write!(f, "\"{s}\""),
            Self::OpenParen => write!(f, "("),
            Self::ClosingParen => write!(f, ")"),
            Self::OpenCurly => write!(f, "{{"),
            Self::ClosingCurly => write!(f, "}}"),
            Self::OpenSharp => write!(f, "<"),
            Self::ClosingSharp => write!(f, ">"),
            Self::Equal => write!(f, "="),
            Self::Plus => write!(f, "+"),
            Self::Minus => write!(f, "-"),
            Self::Slash => write!(f, "/"),
            Self::Asterisk => write!(f, "*"),
            Self::Comma => write!(f, ","),
        }
    }
}

#[allow(unused)]
#[derive(Debug, PartialEq)]
enum AST<'s> {
    Block(Vec<AST<'s>>),
    Number(Token<'s>),
    StrLit(Token<'s>),
    Ident(Token<'s>),
    FuncCall(Box<AST<'s>>, Vec<AST<'s>>), // func, args
    FuncDecl(Vec<Token<'s>>, Box<AST<'s>>), // params, body
    VarDecl(Token<'s>, Option<Box<AST<'s>>>), // name, val
    BinExpr(Token<'s>, Box<AST<'s>>, Box<AST<'s>>), // op, lhs, rhs
    While(Box<AST<'s>>, Box<AST<'s>>), // condition, body
}

impl<'s> std::fmt::Display for AST<'s> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Block(block) => {
                let mut res = String::new();
                for b in block {
                    res.push_str(&format!("{} ", b));
                }
                write!(f, "{{ {res}}}")
            }
            Self::Number(n) => write!(f, "{n}"),
            Self::StrLit(s) => write!(f, "{s}"),
            Self::Ident(i) => write!(f, "{i}"),
            Self::FuncCall(func, args) => write!(f, "{func}({0})", args.iter().map(|a|format!("{a}")).collect::<Vec<_>>().join(", ")),
            Self::FuncDecl(param, body) => write!(f, "f({0}) {{ {body} }}", param.iter().map(|t|format!("{t}")).collect::<Vec<_>>().join(", ")),
            Self::VarDecl(name, Some(val)) => write!(f, "let {name} = {val}"),
            Self::VarDecl(name, None) => write!(f, "let {name} = <undefined>"),
            Self::BinExpr(op, lhs, rhs) => write!(f, "{lhs} {op} {rhs}"),
            Self::While(cond, body) => write!(f, "while {cond} {{ {body} }}"),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
enum Value<'s> {
    Bool(bool),
    Number(i64),
    String(String),
    Variable(&'s str),
    Func(&'s AST<'s>),
    Undefined
}

impl<'s> std::fmt::Display for Value<'s> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Bool(b) => write!(f, "{b}"),
            Self::Number(n) => write!(f, "{n}"),
            Self::String(s) => write!(f, "\"{s}\""),
            Self::Variable(v) => write!(f, "{v}"),
            Self::Func(func) => write!(f, "{func}"),
            Self::Undefined => write!(f, "<undefined>"),
        }
    }
}
#[derive(Debug)]
struct Interpreter<'s> {
    stack_depth: usize,
    variables: VecDeque<(usize, HashMap<&'s str, Value<'s>>)>,
}

impl<'s> Interpreter<'s> {
    fn new() -> Self {
        Self {
            stack_depth: 0,
            variables: VecDeque::new(),
        }
    }

    fn eval_lhs(&mut self, node: &AST<'s>) -> Option<Value<'s>> {
        let AST::Ident(Token::Ident(name)) = &node else {
            eprintln!("Unexpected value for LHS. Got {node}");
            return None
        };
        Some(Value::Variable(name))
    }

    fn run_node<'ast: 's>(&mut self, node: &'ast AST<'s>) -> Option<Value<'s>> {
        match node {
            AST::Block(block) => {
                self.variables.push_back((self.stack_depth, HashMap::new()));
                let mut res = Value::Undefined;
                for b in block {
                    res = self.run_node(&b)?;
                }
                self.variables.pop_back();
                Some(res)
            }
            AST::Number(Token::Number(number)) => {
                Some(Value::Number(number.parse::<i64>().unwrap()))
            }
            AST::StrLit(Token::StrLit(string_lit)) => {
                Some(Value::String(string_lit.to_string()))
            }
            AST::Ident(Token::Ident(ident)) => {
                let Some(scope) = self.get_scope_of_variable(&ident) else {
                    eprintln!("error: Attempted to access unknown variable `{ident}`");
                    return None;
                };
                let Some(val) = scope.get(ident) else {
                    unreachable!() // contains_key() is guaranteed
                };
                Some(val.clone())
            }
            AST::BinExpr(tkn, lhs, rhs) => {
                let lhs_eval = if let Token::Equal = tkn {
                    self.eval_lhs(&lhs)?
                } else {
                    self.run_node(&lhs)?
                };
                let rhs_eval = self.run_node(&rhs)?;
                match (tkn, lhs_eval, rhs_eval) {
                    (Token::Plus, Value::Number(n1), Value::Number(n2)) => Some(Value::Number(n1 + n2)),
                    (Token::Minus, Value::Number(n1), Value::Number(n2)) => Some(Value::Number(n1 - n2)),
                    (Token::Asterisk, Value::Number(n1), Value::Number(n2)) => Some(Value::Number(n1 * n2)),
                    (Token::Plus, Value::String(s1), Value::String(s2)) => Some(Value::String(s1.to_owned() + &s2)),
                    (Token::OpenSharp, Value::Number(n1), Value::Number(n2)) => Some(Value::Bool(n1 < n2)),
                    (Token::ClosingSharp, Value::Number(n1), Value::Number(n2)) => Some(Value::Bool(n1 > n2)),
                    (Token::Equal, Value::Variable(v), rhs_eval) => {
                        let Some(scope) = self.get_scope_of_variable(&v) else {
                            todo!() // Undeclared variable
                        };
                        scope.insert(v, rhs_eval.clone());
                        Some(rhs_eval)
                    },
                    (t, l, r) => {
                        eprintln!("error: Operation `{l} {t} {r}` is not defined!");
                        None
                    }
                }
            }
            AST::VarDecl(Token::Ident(name), val) => {
                let rhs_eval = if let Some(val) = val {
                    self.run_node(&val)?
                } else {
                    Value::Undefined
                };
                let current_scope = self.get_current_scope();
                if current_scope.insert(name, rhs_eval.clone()).is_some() {
                    todo!() // Do we wanna allow shadowing?
                }
                Some(rhs_eval)
            }
            AST::While(cond, body) => {
                let mut body_eval = Value::Undefined;
                while self.run_node(&cond)? == Value::Bool(true) {
                    body_eval = self.run_node(&body)?;
                }
                Some(body_eval)
            }
            AST::FuncDecl(..) => {
                Some(Value::Func(&node))
            }
            AST::FuncCall(func, args) => {
                let f = self.run_node(func)?;
                let Value::Func(func) = f else {
                    eprintln!("Expected function value for function call, got `{f}");
                    return None;
                };
                let AST::FuncDecl(params, body) = func else {
                    unreachable!()
                };
                let mut eval_args = Vec::new();
                for a in args {
                    eval_args.push(self.run_node(a)?);
                }
                self.stack_depth += 1;
                self.variables.push_back((self.stack_depth, HashMap::new()));
                let curr_scope = self.get_current_scope();
                for p in params {
                    let Token::Ident(p) = p else {
                        todo!() // Unreachable?
                    };
                    if curr_scope.insert(p, Value::Undefined).is_some() {
                        todo!()
                    }
                }
                for (a, p) in eval_args.iter().zip(params) {
                    let Token::Ident(p) = p else {
                        todo!() // Unreachable?
                    };
                    curr_scope.insert(p, a.clone());
                }
                let res = self.run_node(body)?;
                assert!(self.stack_depth != 0, "How did this happen?");
                self.stack_depth -= 1;
                self.variables.pop_back();
                Some(res)
            }
            n => todo!("{:?}", n)
        }
    }

    fn get_scope_of_variable(&mut self, var_name: &str) -> Option<&mut HashMap<&'s str, Value<'s>>> {
        for scope in self.variables.iter_mut().rev() {
            if scope.0 == self.stack_depth && scope.1.contains_key(var_name) {
                return Some(&mut scope.1);
            }
        }
        None
    }

    fn get_current_scope(&mut self) -> &mut HashMap<&'s str, Value<'s>> {
        let l = self.variables.len();
        assert!(l != 0);
        &mut self.variables.get_mut(l - 1).unwrap().1
    }

    fn run_block<'ast: 's>(&mut self, block: &'ast AST<'s>) -> Option<Value<'s>> {
        self.run_node(block)
    }
}

#[derive(Debug)]
struct Parser<'s> {
    lexer: &'s mut Lexer<'s>,
}

impl<'s> Parser<'s> {
    fn new(lexer: &'s mut Lexer<'s>) -> Self {
        Parser {
            lexer
        }
    }

    fn expect_ident(&mut self) -> Option<Token<'s>> {
        let pot = self.next_token()?;
        if let Token::Ident(_) = pot {
            Some(pot)
        } else {
            None
        }
    }

    fn expect(&mut self, tkn: &[Token<'s>]) -> Option<Token<'s>> {
        let pot = self.next_token()?;
        for t in tkn {
            if *t == pot {
                return Some(pot);
            }
        }
        None
    }

    fn at(&mut self, tkns: &[Token<'s>]) -> bool {
        if let Some(t) = self.peek_token() {
            tkns.contains(&t)
        } else {
            false
        }
    }
            

    fn eat(&mut self, tkn: Token<'s>) -> bool {
        if let Some(t) = self.peek_token() {
            if t == tkn {
                let _ = self.next_token();
                true
            } else {
                false
            }
        } else {
            false
        }
    }

    fn next_token(&mut self) -> Option<Token<'s>> {
        self.lexer.next()
    }

    fn peek_token(&mut self) -> Option<Token<'s>> {
        self.lexer.peek()
    }
    
    fn reached_eof(&mut self) -> bool {
        self.lexer.peek().is_none()
    }

    fn get_precedence(&self, tkn: &Token<'s>) -> u8 {
        match tkn {
            Token::Slash => 12,
            Token::Asterisk => 12,
            Token::Plus => 11,
            Token::Minus => 11,
            Token::OpenSharp => 9,
            Token::ClosingSharp => 9,
            Token::Equal => 2,
            e => todo!("{e}")
        }
    }

    fn get_associativity(&self, tkn: &Token<'s>) -> Assoc {
        if *tkn == Token::Equal {
            Assoc::Right
        } else {
            Assoc::Left
        }
    }

    fn matches_binary(&mut self) -> bool {
        if self.reached_eof() { false }
        else {
            match self.peek_token().unwrap() {
                Token::Plus => true,
                Token::Minus => true,
                Token::Slash => true,
                Token::Asterisk => true,
                Token::Equal => true,
                Token::OpenSharp => true,
                Token::ClosingSharp => true,
                _ => false
            }
        }
    }

    fn parse_while(&mut self) -> Option<AST<'s>> {
        self.expect(&[Token::Keyword("while")])?;
        let cond = self.parse_expr()?;
        let body = self.parse_expr()?;
        Some(AST::While(Box::new(cond), Box::new(body)))
    }

    fn parse_let(&mut self) -> Option<AST<'s>> {
        self.expect(&[Token::Keyword("let")])?;
        let name = self.expect_ident()?;
        let val = if self.eat(Token::Equal) {
            Some(Box::new(self.parse_expr()?))
        } else {
            None
        };
        Some(AST::VarDecl(name, val))
     }

    fn parse_fn(&mut self) -> Option<AST<'s>> {
        self.expect(&[Token::Keyword("f")])?;
        self.expect(&[Token::OpenParen])?;
        let mut params = Vec::new();
        while !self.at(&[Token::ClosingParen]) {
            let p = self.expect_ident()?;
            params.push(p);
            if !self.at(&[Token::ClosingParen]) {
                self.expect(&[Token::Comma])?;
            }
        }
        self.expect(&[Token::ClosingParen])?;
        let body = self.parse_expr()?;
        Some(AST::FuncDecl(params, Box::new(body)))
    }

    fn parse_expr(&mut self) -> Option<AST<'s>> {
        self.parse_expr_helper(0, Assoc::Left)
    }

    fn parse_expr_sec(&mut self, lhs: AST<'s>, prec: u8, assoc: Assoc) -> Option<AST<'s>> {
        assert!(self.matches_binary());
        let op = self.next_token()?;
        let rhs = self.parse_expr_helper(prec, assoc)?;
        Some(AST::BinExpr(op, Box::new(lhs), Box::new(rhs)))
    }

    fn parse_expr_helper(&mut self, min_prec: u8, assoc: Assoc) -> Option<AST<'s>> {
        let mut lhs = match self.peek_token()? {
            Token::Keyword("f") => self.parse_fn()?,
            Token::Keyword("let") => self.parse_let()?,
            Token::Keyword("while") => self.parse_while()?,
            Token::Number(_) => AST::Number(self.next_token().unwrap()),
            Token::Ident(_) => AST::Ident(self.next_token().unwrap()),
            Token::StrLit(_) => AST::StrLit(self.next_token().unwrap()),
            Token::OpenCurly => self.parse_block()?,
            e => {
                eprintln!("parse_expr: Unexpected token {e}");
                return None;
            }
        };
        if self.eat(Token::OpenParen) {
            let mut args = Vec::new();
            while !self.at(&[Token::ClosingParen]) {
                let arg = self.parse_expr()?;
                args.push(arg);
                if !self.at(&[Token::ClosingParen]) {
                    self.expect(&[Token::Comma])?;
                }
            }
            self.expect(&[Token::ClosingParen])?;
            lhs = AST::FuncCall(Box::new(lhs), args);
        }
        while self.matches_binary() {
            let tkn = self.peek_token()?;
            let new_prec = self.get_precedence(&tkn);
            if new_prec < min_prec {
                break;
            }
            if new_prec == min_prec && assoc == Assoc::Left {
                break;
            }
            let new_assoc = self.get_associativity(&tkn);
            lhs = self.parse_expr_sec(lhs, new_prec, new_assoc)?;
        }
        Some(lhs)
    }

    fn parse_block(&mut self) -> Option<AST<'s>> {
        let expect_curly = self.eat(Token::OpenCurly);
        let mut block = Vec::new();
        while !self.reached_eof() && (!expect_curly || !self.eat(Token::ClosingCurly)) {
            let ass = self.parse_expr()?;
            block.push(ass);
        }
        Some(AST::Block(block))
    }

    fn parse_program(&mut self) -> Option<AST<'s>> {
        let mut block = Vec::new();
        while !self.reached_eof() {
            let blk = self.parse_block()?;
            block.push(blk);
        }
        Some(AST::Block(block))
    }
}

#[derive(Debug)]
struct Lexer<'s> {
    content: &'s str,
    ptr: usize,
    len: usize,
}

impl<'s> Lexer<'s> {
    fn new(content: &'s str) -> Self {
        Self {
            content,
            ptr: 0,
            len: content.len()
        }
    }
    fn peek(&mut self) -> Option<Token<'s>> {
        let save = self.ptr;
        let orig = self.content;
        let t = self.next()?;
        self.ptr = save;
        self.content = orig;
        Some(t)
    }
    fn next(&mut self) -> Option<Token<'s>> {
        if self.ptr >= self.len {
            return None;
        }
        while self.content.starts_with(char::is_whitespace) {
            self.ptr += 1;
            self.content = &self.content[1..];
            if self.ptr >= self.len {
                return None;
            }
        }
        let t = if self.content.starts_with(|c:char|c.is_alphabetic()||c=='_') {
            let mut len = 0;
            let mut tmp = self.content;
            while tmp.starts_with(|c:char|c.is_alphabetic()||c=='_') {
                len += 1;
                tmp = &tmp[1..];
            }    
            let word = &self.content[0..len];
            self.ptr += len;
            self.content = tmp;
            if KEYWORDS.contains(&word) {
                Token::Keyword(word)
            } else {
                Token::Ident(word)
            }
        } else if self.content.starts_with(char::is_numeric) {
            let mut len = 0;
            let mut tmp = self.content;
            while tmp.starts_with(char::is_numeric) {
                len += 1;
                tmp = &tmp[1..];
            }
            let word = &self.content[0..len];
            self.ptr += len;
            self.content = tmp;
            Token::Number(word)
        } else if self.content.starts_with('"') {
            let mut len = 0;
            let mut tmp = &self.content[1..];
            while !tmp.starts_with('"') {
                len += 1;
                tmp = &tmp[1..];
            }
            let word = &self.content[1..(len+1)];
            self.ptr += len + 2;
            self.content = &tmp[1..];
            Token::StrLit(word)
        } else if self.content.starts_with('=') {
            self.content = &self.content[1..];
            self.ptr += 1;
            Token::Equal
        } else if self.content.starts_with('{') {
            self.content = &self.content[1..];
            self.ptr += 1;
            Token::OpenCurly
        } else if self.content.starts_with('}') {
            self.content = &self.content[1..];
            self.ptr += 1;
            Token::ClosingCurly
        } else if self.content.starts_with('(') {
            self.content = &self.content[1..];
            self.ptr += 1;
            Token::OpenParen
        } else if self.content.starts_with(')') {
            self.content = &self.content[1..];
            self.ptr += 1;
            Token::ClosingParen
        } else if self.content.starts_with('<') {
            self.content = &self.content[1..];
            self.ptr += 1;
            Token::OpenSharp
        } else if self.content.starts_with('>') {
            self.content = &self.content[1..];
            self.ptr += 1;
            Token::ClosingSharp
        } else if self.content.starts_with('+') {
            self.content = &self.content[1..];
            self.ptr += 1;
            Token::Plus
        } else if self.content.starts_with('-') {
            self.content = &self.content[1..];
            self.ptr += 1;
            Token::Minus
        } else if self.content.starts_with('/') {
            self.content = &self.content[1..];
            self.ptr += 1;
            Token::Slash
        } else if self.content.starts_with('*') {
            self.content = &self.content[1..];
            self.ptr += 1;
            Token::Asterisk
        } else if self.content.starts_with(',') {
            self.content = &self.content[1..];
            self.ptr += 1;
            Token::Comma
        } else {
            eprintln!("error: Unknown character `{0}` at byte {1}", &self.content[0..1], self.ptr);
            std::process::exit(1)
        };
        Some(t)
    }
}

fn main() {
    let mut args = env::args();
    let _prog_name = args.next().expect("Program must be provided");
    let Some(file_input) = args.next() else {
        eprintln!("No input file provided!");
        std::process::exit(1)
    };
    println!("Reading file `{file_input}`");
    match fs::read_to_string(file_input) {
        Ok(file) => {
            let mut lexer = Lexer::new(&file);
            let mut parser = Parser::new(&mut lexer);
            let mut interpreter = Interpreter::new();
            if let Some(program) = parser.parse_program() {
                println!("Block:\n{program}");
                let Some(result) = interpreter.run_block(&program) else {
                    eprintln!("error: Could not interpret the given program.");
                    std::process::exit(1)
                };
                println!("Result: {}", result);
            }
        },
        Err(e) => {
            panic!("{}", e);
        }
    }
}
