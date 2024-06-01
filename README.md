# ylang - The language that makes you go "y?"
## About
<p>This is a language that I created on a weekend and some train rides, to explore dynamic languages and Interpreters a bit more.</p>
<p>I now know how JavaScript was written in 10 days.</p>

- Is it revolutionary? No.
- Is it nice to use? Depends on your definition of nice.
- Is it stable? Not more than my mental state.
- Is it fast? Well, consider upgrading your hardware lol.
- Was it fun to make? Absolutely, I learned so much during this time!
## Features
- [x] Dynamically typed
- [x] [Turing Complete](./ex/rule110.y)
- [x] `f` is a builtin keyword
- [x] No Syntax Sugar, no bloat
  - No semicolons, no forced line breaks
  - Often used constructs such as `if` or logical `and` are defined in the [standard library](./std/std.y)
  - `while a > 0 a = a - 1` is a valid expression:
    - `a > 0` is the condition
    - `a = a - 1` is the body
- [x] Everything is an expression. Yes, everything.
- [x] Some special operations include:
  - `<list> * <fn>` returns a new list containing all elements of the original list, applied to the given function
  - `<list> / <fn>` returns a new list containing all elements that return `1` when applied to the given function
  - `<number> * <fn>` creates a list of length `<number>`,
  where each element is the result of the call to `fn(index)`
  - `<list> . <number>` *gets* the element at the given index
    - Note that you can't *set* the element (yet?), you'd have to create a new list for that.
- [x] Special support for intrinsics, my favorites:
  - `noa(<string>)` returns `1` if the argument does not contain a lowercase `a` (Thanks Noa!)
  - `nel(<string>)` returns `1` if the argument is not equal to lowercase `l` (Thanks Angelo!)