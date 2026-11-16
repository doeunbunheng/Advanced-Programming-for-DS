# TP-01 — Introduction to Python Programming Concepts

## Objective

Learn the basic Python concepts used in programming and Data Science.

## 1. Variable

A **variable** stores data that we can use later.

```python
name = "Bunheng"
age = 20
```

**Remember:** Variable = stores data.

---

## 2. Function

A **function** is a reusable block of code that performs a task.

```python
def add(a, b):
    return a + b

print(add(2, 3))
```

**Remember:** Function = performs a task.

---

## 3. Class

A **class** is a blueprint used to create objects.

```python
class Student:
    pass
```

**Remember:** Class = blueprint for objects.

---

## 4. Method

A **method** is a function inside a class.

```python
class Student:
    def study(self):
        print("Studying")
```

**Remember:** Method = function inside a class.

---

## 5. Constant

A **constant** is a value that should not change.

```python
PI = 3.14159
```

In Python, constants are usually written in **UPPERCASE**.

**Remember:** Constant = fixed value.

---

## 6. Module

A **module** is one Python file that contains reusable code.

Example:

```text
math_utils.py
```

It can be imported into another Python file:

```python
import math_utils
```

**Remember:** Module = one Python file.

---

## 7. Package

A **package** is a folder that contains multiple Python modules.

Example:

```text
my_package/
├── __init__.py
├── module1.py
└── module2.py
```

**Remember:** Package = folder of modules.

---

## Quick Summary

| Concept | Simple Meaning |
|---|---|
| Variable | Stores data |
| Function | Performs a reusable task |
| Class | Blueprint for objects |
| Method | Function inside a class |
| Constant | Value that should not change |
| Module | One Python file |
| Package | Folder of Python modules |

## Easy Structure to Remember

```text
Package
  ↓
Modules
  ↓
Classes
  ↓
Methods
```

Other basic concepts:

```text
Variable → stores data
Function → reusable action
Constant → fixed value
```

## What I Learned

After TP-01, I should understand the difference between:

- Variable
- Function
- Class
- Method
- Constant
- Module
- Package


---

# 02 — Clean Code, PEP 8, and Debugging

## Why Best Practices Matter

Writing code is one thing; writing **good code** is another.

Best practices help us improve the quality of our code.

---

### 1. Improve Readability

Make your code easier to understand for yourself and others.

Example:

Bad:

```python
x = 10
y = 2
z = x * y
```

Better:

```python
price = 10
quantity = 2
total_price = price * quantity
```

**Remember:** Readability = easy to read and understand.

---

### 2. Enhance Maintainability

Clean code makes updates and bug fixes easier.

Example:

```python
def calculate_total(price, quantity):
    return price * quantity
```

If the calculation changes later, we only need to change the function.

**Remember:** Maintainability = easy to update and fix.

---

### 3. Boost Efficiency

Write code that runs faster and uses fewer resources.

Example:

Instead of:

```python
total = 0

for number in numbers:
    total += number
```

Use:

```python
total = sum(numbers)
```

**Remember:** Efficiency = better performance with fewer resources.

---

### 4. Avoid Errors

Catch problems early and reduce debugging time.

Example:

```python
try:
    age = int(input("Enter age: "))
except ValueError:
    print("Please enter a valid number.")
```

This prevents the program from crashing when the input is wrong.

**Remember:** Avoid errors = find problems early.

---

## Quick Summary

| Point | Simple Meaning | Small Example |
|---|---|---|
| Improve Readability | Easy to understand | Use `total_price` instead of `x` |
| Enhance Maintainability | Easy to update | Put logic inside a function |
| Boost Efficiency | Use fewer resources | Use `sum(numbers)` |
| Avoid Errors | Catch problems early | Use `try-except` |

## Easy Way to Remember

```text
Good Code
   ↓
Easy to Read
   ↓
Easy to Update
   ↓
More Efficient
   ↓
Fewer Errors
```

## Main Idea

> Good code should work correctly and also be easy to read, update, improve, and debug.

## Small Examples to Remember

```python
# Readability
total_price = price * quantity

# Maintainability
def calculate_total(price, quantity):
    return price * quantity

# Efficiency
total = sum(numbers)

# Avoid errors
try:
    number = int(input("Enter number: "))
except ValueError:
    print("Invalid input")
```

# 03 — PEP 8 Tips and Tricks

PEP 8 is the Python style guide. It helps us write code that is clean, consistent, and easy to read.

## 1. Use Clear Names

Use meaningful names for variables and functions.

Bad:

```python
x = 100
y = 2
z = x * y
```

Better:

```python
price = 100
quantity = 2
total_price = price * quantity
```

**Remember:** Good names explain what the value means.

---

## 2. Use `snake_case` for Variables and Functions

```python
student_name = "Bunheng"

def calculate_total():
    pass
```

**Remember:** Variables and functions → `snake_case`

---

## 3. Use `PascalCase` for Classes

```python
class StudentProfile:
    pass
```

**Remember:** Classes → `PascalCase`

---

## 4. Use UPPERCASE for Constants

```python
PI = 3.14159
MAX_USERS = 100
```

**Remember:** Constants → `UPPERCASE`

---

## 5. Use 4 Spaces for Indentation

Bad:

```python
if score >= 50:
  print("Pass")
```

Better:

```python
if score >= 50:
    print("Pass")
```

**Remember:** Python indentation = 4 spaces.

---

## 6. Add Spaces Around Operators

Bad:

```python
total=price*quantity
```

Better:

```python
total = price * quantity
```

**Remember:** Spaces make code easier to read.

---

## 7. Add a Space After Commas

Bad:

```python
numbers = [1,2,3,4]
```

Better:

```python
numbers = [1, 2, 3, 4]
```

---

## 8. Keep Imports at the Top

```python
import math
import pandas as pd
```

Avoid placing imports randomly in the middle of the file unless there is a special reason.

---

## 9. Keep Lines Short

PEP 8 recommends keeping normal code lines to about **79 characters or less**.

Bad:

```python
result = calculate_total_price(price, quantity, discount, tax, shipping_cost, service_fee)
```

Better:

```python
result = calculate_total_price(
    price,
    quantity,
    discount,
    tax,
    shipping_cost,
    service_fee,
)
```

---

## 10. Use Blank Lines to Separate Code

```python
def add(a, b):
    return a + b


def subtract(a, b):
    return a - b
```

Blank lines help separate different parts of the program.

---

## 11. Write Useful Comments

Bad:

```python
# Add 1
count = count + 1
```

Better:

```python
# Count the number of valid records
count = count + 1
```

**Remember:** Comments should explain **why**, not obvious code.

---

## Quick Summary

| PEP 8 Rule | Easy Meaning |
|---|---|
| `snake_case` | Variables and functions |
| `PascalCase` | Classes |
| `UPPERCASE` | Constants |
| 4 spaces | Indentation |
| Spaces around operators | Improve readability |
| Space after commas | Cleaner lists and arguments |
| Imports at top | Better organization |
| Short lines | Easier to read |
| Blank lines | Separate code sections |
| Useful comments | Explain important logic |

## Easy Example

Bad:

```python
def Calc(a,b):
 return a+b
```

Better:

```python
def calculate_sum(a, b):
    return a + b
```

The second version follows PEP 8 and is easier to understand.
