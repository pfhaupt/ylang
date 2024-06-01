use std::fs;
use std::env;
use std::collections::{HashMap, VecDeque};

const KEYWORDS: &[&str] = &["let", "f", "while", "use"];
const INTRINSICS: &[&str] = &["print", "noa", "nel", "len"];

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
    Comma,
    Dot,
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
            Self::Dot => write!(f, "."),
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
    FuncCall(Box<AST<'s>>, Box<AST<'s>>), // func, args
    FuncDecl(Vec<Token<'s>>, Box<AST<'s>>), // params, body
    VarDecl(Token<'s>, Option<Box<AST<'s>>>), // name, val
    BinExpr(Token<'s>, Box<AST<'s>>, Box<AST<'s>>), // op, lhs, rhs
    While(Box<AST<'s>>, Box<AST<'s>>), // condition, body
    ListDecl(Vec<AST<'s>>),
}

impl<'s> std::fmt::Display for AST<'s> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Block(block) => {
                let mut res = String::new();
                for b in block {
                    res.push_str(&format!("{} ", b));
                }
                write!(f, "{{ {res} }}")
            }
            Self::Number(n) => write!(f, "{n}"),
            Self::StrLit(s) => write!(f, "{s}"),
            Self::Ident(i) => write!(f, "{i}"),
            Self::FuncCall(func, args) => write!(f, "{func}{args}"),
            Self::FuncDecl(param, body) => write!(f, "f({0}) {{ {body} }}", param.iter().map(|t|format!("{t}")).collect::<Vec<_>>().join(", ")),
            Self::VarDecl(name, Some(val)) => write!(f, "let {name} = {val}"),
            Self::VarDecl(name, None) => write!(f, "let {name} = <undefined>"),
            Self::BinExpr(op, lhs, rhs) => write!(f, "{lhs} {op} {rhs}"),
            Self::While(cond, body) => write!(f, "while {cond} {{ {body} }}"),
            Self::ListDecl(elems) => write!(f, "({0})", elems.iter().map(|e|format!("{e}")).collect::<Vec<_>>().join(", ")),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
enum Value<'s> {
    Number(i64),
    String(String),
    List(Vec<Value<'s>>),
    Variable(&'s str),
    Func(&'s AST<'s>),
    Undefined
}

impl<'s> std::fmt::Display for Value<'s> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Number(n) => write!(f, "{n}"),
            Self::String(s) => write!(f, "{s}"),
            Self::Variable(v) => write!(f, "{v}"),
            Self::Func(func) => write!(f, "{func}"),
            Self::List(elems) => write!(f, "({0})", elems.iter().map(|e|format!("{e}")).collect::<Vec<_>>().join(", ")),
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

    fn run_function<'ast: 's>(&mut self, args: &Vec<Value<'s>>, params: &Vec<Token<'s>>, body: &'ast AST<'s>) -> Option<Value<'s>> {
        // TODO: Inline small functions like `let and=f(a,b)a*b`
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
        for (a, p) in args.iter().zip(params) {
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
                    (_, Value::Undefined, _) | (_, _, Value::Undefined) => {
                        println!("warning: Propagation of <undefined> at {node}");
                        Some(Value::Undefined)
                    },
                    (Token::Plus, Value::Number(n1), Value::Number(n2)) => Some(Value::Number(n1 + n2)),
                    (Token::Plus, Value::String(s1), Value::String(s2)) => Some(Value::String(s1.to_owned() + &s2)),
                    (Token::Plus, Value::List(elems), rhs) => {
                        let mut new_elems = elems;
                        new_elems.push(rhs);
                        Some(Value::List(new_elems))
                    }
                    (Token::Minus, Value::Number(n1), Value::Number(n2)) => Some(Value::Number(n1 - n2)),
                    (Token::Minus, Value::List(elems), rhs) => {
                        let mut new_elems = Vec::new();
                        for e in elems {
                            if e != rhs {
                                new_elems.push(e);
                            }
                        }
                        Some(Value::List(new_elems))
                    }
                    (Token::Asterisk, Value::Number(n1), Value::Number(n2)) => Some(Value::Number(n1 * n2)),
                    (Token::Asterisk, Value::Number(n), ref func @ Value::Func(AST::FuncDecl(params, body))) => {
                        if params.len() != 1 {
                            eprintln!("Operation `<number> {tkn} <func>` requires predicate that takes in one parameter (the index), found {0}", func);
                            return None;
                        }
                        if n < 0 {
                            eprintln!("Operation `<number> {tkn} <func>` requires positive number (the amount of elements in the created list), found {0}.", n);
                            return None;
                        }
                        let n = n as usize;
                        let mut new_elems = vec![Value::Undefined; n];
                        let mut args = vec![Value::Undefined];
                        for i in 0..n {
                            args[0] = Value::Number(i as i64);
                            new_elems[i] = self.run_function(&args, &params, &body)?;
                        }
                        Some(Value::List(new_elems))
                    }
                    (Token::Asterisk, Value::List(elems), ref func @ Value::Func(AST::FuncDecl(params, body))) => {
                        if params.len() != 1 {
                            eprintln!("List map operator `{tkn}` requires predicate that takes in one parameter (the element), found {0}", func);
                            return None;
                        }
                        let mut new_elems = Vec::new();
                        for e in elems {
                            let args = vec![e];
                            let res = self.run_function(&args, &params, &body)?;
                            new_elems.push(res);
                        }
                        Some(Value::List(new_elems))
                    }
                    (Token::Slash, Value::Number(n1), Value::Number(n2)) => Some(Value::Number(n1 / n2)),
                    (Token::Slash, Value::List(elems), ref func @ Value::Func(AST::FuncDecl(params, body))) => {
                        if params.len() != 1 {
                            eprintln!("List filter operator `{tkn}` requires predicate that takes in one parameter (the element), found {0}", func);
                            return None;
                        }
                        let mut new_elems = Vec::new();
                        for e in elems {
                            let args = vec![e.clone()];
                            let res = self.run_function(&args, &params, &body)?;
                            if res == Value::Number(1) {
                                new_elems.push(e);
                            }
                        }
                        Some(Value::List(new_elems))
                    }
                    (Token::Slash, Value::String(orig), Value::List(elems)) => {
                        let mut split_vals = Vec::with_capacity(elems.len());
                        // REVIEW: This looks stupidly over-engineered
                        for e in elems {
                            let Value::String(sep) = e else {
                                eprintln!("error: String splice operator `{tkn}` requires a list of single characters (the characters to split by), found `{0}`.", e);
                                return None;
                            };
                            if sep.len() != 1 {
                                eprintln!("error: String splice operator `{tkn}` requires a list of single characters (the characters to split by), found `{0}`.", sep);
                                return None;
                            };
                            let mut sep = sep.chars();
                            let c = sep.next().unwrap();
                            assert!(sep.next().is_none());

                            split_vals.push(c);
                        }
                        let res = orig.split(&*split_vals).filter(|e|e.len()!=0).map(|e|Value::String(e.to_string())).collect::<Vec<_>>();
                        Some(Value::List(res))
                    }
                    (Token::OpenSharp, Value::Number(n1), Value::Number(n2)) => Some(Value::Number((n1 < n2) as i64)),
                    (Token::ClosingSharp, Value::Number(n1), Value::Number(n2)) => Some(Value::Number((n1 > n2) as i64)),
                    (Token::Equal, Value::Variable(v), rhs_eval) => {
                        let Some(scope) = self.get_scope_of_variable(&v) else {
                            todo!() // Undeclared variable
                        };
                        scope.insert(v, rhs_eval.clone());
                        Some(rhs_eval)
                    },
                    (Token::Dot, Value::List(elems), Value::Number(n)) => {
                        if n < 0 || n >= elems.len() as i64 {
                            eprintln!("error: Attempted to access invalid index {n} of length {} list.", elems.len());
                            return None;
                        }
                        Some(elems[n as usize].clone())
                    }
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
                self.insert_variable(name, &rhs_eval)?;
                Some(rhs_eval)
            }
            AST::While(cond, body) => {
                let mut body_eval = Value::Undefined;
                while self.run_node(&cond)? == Value::Number(1) {
                    body_eval = self.run_node(&body)?;
                }
                Some(body_eval)
            }
            AST::FuncDecl(..) => {
                Some(Value::Func(&node))
            }
            AST::FuncCall(func, args) => {
                let Value::List(args) = self.run_node(&args)? else {
                    unreachable!()
                };
                if let AST::Ident(Token::Ident(maybe_intrinsic)) = &**func {
                    macro_rules! check_args {
                        ($vals:expr, $cnt:expr, $($exp:tt)*) => {
                            if $vals.len() < $cnt {
                                eprintln!("error: Intrinsic `{maybe_intrinsic}` requires at least {} argument(s), found {}.", $cnt, $vals.len());
                                return None;
                            }
                            let $($exp)* = &$vals[$cnt - 1] else {
                                eprintln!("error: Intrinsic `{maybe_intrinsic}` requires a special value as argument {}, got the wrong one. Oops.", $cnt);
                                return None;
                            };
                        }
                    }
                    match maybe_intrinsic {
                        &"print" => {
                            let v = args.iter().map(|a|format!("{a}")).collect::<Vec<_>>().join(" ");
                            println!("{v}");
                            Some(Value::Undefined)
                        },
                        &"noa" => {
                            check_args!(&args, 1, Value::String(arg));
                            Some(Value::Number(!arg.contains("a") as i64))
                        },
                        &"nel" => {
                            check_args!(&args, 1, Value::String(arg));
                            Some(Value::Number((arg != "l") as i64))
                        }
                        &"len" => {
                            check_args!(&args, 1, Value::List(arg));
                            Some(Value::Number(arg.len() as i64))
                        }
                        name => {
                            assert!(!INTRINSICS.contains(name));
                            let f = self.run_node(func)?;
                            let Value::Func(func) = f else {
                                eprintln!("Expected function value for function call, got `{f}`");
                                return None;
                            };
                            let AST::FuncDecl(params, body) = func else {
                                unreachable!()
                            };
                            self.run_function(&args, &params, &body)
                        }
                    }
                } else {
                    let f = self.run_node(func)?;
                    let Value::Func(func) = f else {
                        eprintln!("Expected function value for function call, got `{f}`");
                        return None;
                    };
                    let AST::FuncDecl(params, body) = func else {
                        unreachable!()
                    };
                    self.run_function(&args, &params, &body)
                }
            }
            AST::ListDecl(elems) => {
                let mut eval_elems = Vec::new();
                for e in elems {
                    let e = self.run_node(e)?;
                    eval_elems.push(e);
                }
                Some(Value::List(eval_elems))
            }
            n => todo!("{:?}", n)
        }
    }

    fn insert_variable(&mut self, var_name: &'s str, var_val: &Value<'s>) -> Option<()> {
        let current_scope = self.get_current_scope();
        if let Some(_) = current_scope.insert(var_name, var_val.clone()) {
            eprintln!("error: Redefinition of variable `{var_name}`");
            None
        } else {
            Some(())
        }
    }

    fn get_scope_of_variable(&mut self, var_name: &str) -> Option<&mut HashMap<&'s str, Value<'s>>> {
        for scope in self.variables.iter_mut().rev() {
            if let Some(v) = scope.1.get(var_name) {
                if let Value::Func(_) = v {
                    return Some(&mut scope.1);
                } else if scope.0 == self.stack_depth {
                    return Some(&mut scope.1);
                }
            }
        }
        None
    }

    fn get_current_scope(&mut self) -> &mut HashMap<&'s str, Value<'s>> {
        let l = self.variables.len();
        assert!(l != 0);
        &mut self.variables.get_mut(l - 1).unwrap().1
    }

    fn reset_variables(&mut self) {
        self.variables.clear();
        self.stack_depth = 0;
    }

    fn define_intrinsics(&mut self, intrinsics: &'s[(&'static str, AST<'s>)]) {
        assert!(intrinsics.len() == INTRINSICS.len(), "define_intrinsics() called with unexpected input!");
        self.reset_variables();
        assert!(self.variables.is_empty(), "define_intrinsics() called in wrong context!");
        assert!(self.stack_depth == 0, "define_intrinsics() called in wrong context!");
        self.variables.push_back((self.stack_depth, HashMap::new()));
        self.stack_depth = 1;
        for (name, ast) in intrinsics {
            let val = match ast {
                AST::FuncDecl(..) => Value::Func(&ast),
                AST::Number(..) => todo!(),
                _ => todo!(), // Unreachable
            };
            // is_some() Because Some(()) indicates success, and None indicates duplicate
            assert!(self.insert_variable(name, &val).is_some());
        }
    }

    fn run_program<'ast: 's>(&mut self, block: &'ast AST<'s>) -> Option<Value<'s>> {
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
            eprintln!("Expected identifier, found `{pot}`");
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
            Token::Dot => 13,
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
                Token::Dot => true,
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

    fn parse_list(&mut self) -> Option<AST<'s>> {
        self.expect(&[Token::OpenParen])?;
        let mut elems = Vec::new();
        while !self.at(&[Token::ClosingParen]) {
            let e = self.parse_expr()?;
            elems.push(e);
            if !self.at(&[Token::ClosingParen]) {
                self.expect(&[Token::Comma])?;
            }
        }
        self.expect(&[Token::ClosingParen])?;
        Some(AST::ListDecl(elems))
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
            Token::OpenParen => self.parse_list()?,
            e => {
                eprintln!("parse_expr: Unexpected token {e}");
                return None;
            }
        };
        if self.at(&[Token::OpenParen]) {
            let args = self.parse_list()?;
            lhs = AST::FuncCall(Box::new(lhs), Box::new(args));
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
        let mut instance = Self {
            content: "",
            ptr: 0,
            len: 0,
        };
        instance.load(content);
        instance
    }
    fn load(&mut self, content: &'s str) {
        self.content = content;
        self.ptr = 0;
        self.len = content.len();
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
        let t = if self.content.starts_with(char::is_alphabetic) {
            let mut len = 0;
            let mut tmp = self.content;
            while tmp.starts_with(char::is_alphabetic) {
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
        } else if self.content.starts_with('.') {
            self.content = &self.content[1..];
            self.ptr += 1;
            Token::Dot
        } else {
            eprintln!("error: Unknown character `{0}` at byte {1}", &self.content[0..1], self.ptr);
            std::process::exit(1)
        };
        Some(t)
    }
}

fn declare_intrinsics<'s>() -> Vec<(&'static str, AST<'s>)> {
    let mut decl_intr = Vec::new();
    macro_rules! empty_fn {
        () => {
            AST::FuncDecl(Vec::new(), Box::new(AST::Block(Vec::new())))
        }
    }
    decl_intr.push(("print", empty_fn!()));
    decl_intr.push(("noa", empty_fn!()));
    decl_intr.push(("nel", empty_fn!()));
    decl_intr.push(("len", empty_fn!()));

    for i in INTRINSICS {
        assert!(decl_intr.iter().find(|(n,_)|n==i).is_some(), "Could not find intrinsic `{i}` in declare_intrinsics()!");
    }
    assert!(decl_intr.len() == INTRINSICS.len());
    decl_intr
}

struct Preproc<'s, 'p> {
    lexer: &'p mut Lexer<'s>,
    import_paths: Vec<&'s str>,
}

impl<'s, 'p> Preproc<'s, 'p> {
    fn new(lexer: &'p mut Lexer<'s>) -> Self {
        Self {
            lexer,
            import_paths: Vec::new(),
        }
    }
    fn add_import(&mut self, path: &'s str) {
        self.import_paths.push(path);
    }
    fn process(&mut self, cycle: &mut VecDeque<String>, content: &'s str) -> Option<String> {
        let mut after = String::with_capacity(content.len());
        self.lexer.load(content);
        while let Some(t) = self.lexer.next() {
            match &t {
                Token::Keyword("use") => {
                    let Some(Token::StrLit(filename)) = self.lexer.next() else {
                        todo!()
                    };
                    let mut valid = false;
                    for path in &self.import_paths {
                        let mut filepath = format!("{path}/{filename}");
                        if !filepath.ends_with(".y") {
                            filepath += ".y";
                        }
                        if fs::metadata(&filepath).is_ok() {
                            let Ok(use_content) = fs::read_to_string(&filepath) else {
                                todo!() // Unreachable because metadata?
                            };
                            let Some(import) = pre_process(cycle, &self.import_paths, &filepath, &use_content) else {
                                eprintln!("error: Could not import file `{filepath}` used in `use {filename}`");
                                return None;
                            };
                            after.push_str(&import);
                            valid = true;
                            break;
                        }
                    }
                    if !valid {
                        println!("error: Could not import file `{filename}`");
                        return None;
                    }
                },
                _ => {
                    after.push_str(&t.to_string());
                }
            }
            after.push(' ');
        }
        Some(after)
    }
}

fn pre_process(cycle: &mut VecDeque<String>, imports: &Vec<&str>, path: &str, content: &str) -> Option<String> {
    if cycle.contains(&path.to_string()) {
        eprintln!("error: Circular import.");
        eprintln!("note: The cycle is:");
        for c in cycle {
            eprintln!("note: {c}");
        }
        std::process::exit(1)
    }
    cycle.push_back(path.to_string());
    println!("Reading file `{path}`");
    let mut lexer = Lexer::new(&content);
    let mut pp = Preproc::new(&mut lexer);
    for path in imports {
        pp.add_import(&path);
    }
    let tmp = pp.process(cycle, &content);
    cycle.pop_back();
    tmp
}

fn main() {
    let mut args = env::args();
    let _prog_name = args.next().expect("Program must be provided");
    let Some(file_input) = args.next() else {
        eprintln!("No input file provided!");
        std::process::exit(1)
    };
    if !file_input.ends_with(".y") {
        eprintln!("Expected file with extension `.y`");
        std::process::exit(1)
    }
    let mut import_paths = Vec::new();
    for a in args {
        if a.starts_with("-I") {
            import_paths.push(a.strip_prefix("-I").unwrap().to_string());
        } else {
            eprintln!("Unexpected argument {a}");
            std::process::exit(1)
        }
    }
    let mut ip = Vec::new();
    for i in &import_paths {
        ip.push(&**i);
    }
    // FIXME: Use absolute paths
    ip.push("./std");
    ip.push(".");
    let intrinsics = declare_intrinsics();
    match fs::read_to_string(&file_input) {
        Ok(file) => {
            let Some(content) = pre_process(&mut VecDeque::new(), &ip, &file_input, &file) else {
                eprintln!("error: Could not pre-process file `{file_input}`");
                std::process::exit(1)
            };
            let mut lexer = Lexer::new(&content);
            let mut parser = Parser::new(&mut lexer);
            let mut interpreter = Interpreter::new();
            interpreter.define_intrinsics(&intrinsics);
            // TODO: Add Bytecode interpreter? :o
            if let Some(program) = parser.parse_program() {
                let Some(result) = interpreter.run_program(&program) else {
                    eprintln!("error: Could not interpret the given program.");
                    std::process::exit(1)
                };
                println!("{result}");
            } else {
                println!("Could not parse program.");
                std::process::exit(1)
            }
        },
        Err(e) => {
            eprintln!("error: {}", e);
            std::process::exit(1)
        }
    }
}
