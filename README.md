# 🚀 Advanced Programming for Data Science

## 📌 Course Overview

**Course:** Advanced Programming for Data Science  
**Short Name:** Advanced Programming for DS  
**Main Language:** Python  
**Purpose:** Learn advanced programming techniques and apply them to real Data Science problems.

This repository contains all materials related to the course, including:

- 📖 Lecture notes
- 🧪 TP / Lab exercises
- 📝 Assignments
- 🏠 Homework
- 💻 Coding practice
- 📊 Data Science projects
- 📂 Datasets
- 📚 Study notes
- ✅ Exam preparation

---

# 🎯 Learning Objectives

By the end of this course, I should be able to:

1. Write clean and efficient Python programs.
2. Understand advanced Python concepts.
3. Work with data using Python libraries.
4. Apply Object-Oriented Programming.
5. Handle files and datasets.
6. Debug and test Python programs.
7. Write reusable functions and modules.
8. Work with NumPy and Pandas.
9. Visualize and analyze data.
10. Build programming solutions for Data Science problems.
11. Organize larger Data Science projects.
12. Improve code performance and readability.

---

# 📂 Course Structure

```text
Advanced-Programming-for-DS/
│
├── README.md
│
├── 01_Lectures/
│   ├── Week01/
│   ├── Week02/
│   ├── Week03/
│   └── ...
│
├── 02_TP/
│   ├── TP01/
│   ├── TP02/
│   ├── TP03/
│   └── ...
│
├── 03_Assignments/
│   ├── Assignment01/
│   ├── Assignment02/
│   └── ...
│
├── 04_Homework/
│   ├── Homework01/
│   ├── Homework02/
│   └── ...
│
├── 05_Practice/
│   ├── Python/
│   ├── NumPy/
│   ├── Pandas/
│   └── Algorithms/
│
├── 06_Projects/
│   ├── Mini_Project/
│   └── Final_Project/
│
├── 07_Datasets/
│
├── 08_Notes/
│
└── 09_Exam_Preparation/
```

---

# 📖 1. Lectures

This folder contains materials learned during class.

For every lesson, I should record:

```text
Topic:
Date:
Main concepts:
Important syntax:
Example:
What I understand:
What I do not understand:
Questions to review:
```

### Example

```text
Week 01
Topic: Python Review

Topics:
- Variables
- Data types
- List
- Tuple
- Dictionary
- Set
- Condition
- Loop
- Function
```

---

# 🧪 2. TP / Lab

**TP = Travaux Pratiques / Practical Work**

TP is where I practice what I learned in class.

For each TP:

```text
TP Number:
Topic:
Objective:
Dataset:
Tasks:
Code:
Output:
Problems encountered:
Solution:
What I learned:
```

### TP Checklist

- [ ] Read the problem
- [ ] Understand the objective
- [ ] Write the code myself
- [ ] Run the program
- [ ] Check the output
- [ ] Fix errors
- [ ] Add comments
- [ ] Save the final version
- [ ] Review before exam

---

# 📝 3. Assignments

Assignments are larger tasks that may combine several lessons.

For each assignment:

```text
Assignment:
Topic:
Deadline:
Requirements:
Dataset:
Methods:
Code:
Results:
Conclusion:
```

### Assignment Workflow

```text
Understand Problem
        ↓
Analyze Requirements
        ↓
Prepare Data
        ↓
Write Code
        ↓
Test Code
        ↓
Fix Errors
        ↓
Analyze Results
        ↓
Write Explanation
        ↓
Submit
```

---

# 🏠 4. Homework

Homework is used to reinforce concepts learned in class.

For every homework:

```text
Homework:
Date:
Topic:
Questions:
My Solution:
Correct Solution:
Mistakes:
What I learned:
```

I should not only save the correct answer.

I should also record:

> Why was my first answer wrong?

This makes revision easier before the exam.

---

# 💻 5. Programming Practice

I should regularly practice small coding problems.

## Python

Practice:

- Variables
- Conditions
- Loops
- Functions
- Lists
- Dictionaries
- Sets
- Tuples
- File handling
- Exception handling
- Modules
- Classes
- Object-Oriented Programming

Example:

```python
def calculate_mean(values):
    return sum(values) / len(values)

data = [10, 20, 30, 40]

print(calculate_mean(data))
```

Output:

```text
25.0
```

---

# 🔢 6. NumPy

Important topics:

- Arrays
- Shape
- Dimension
- Indexing
- Slicing
- Broadcasting
- Vector operations
- Matrix operations
- Statistical functions

Example:

```python
import numpy as np

data = np.array([10, 20, 30, 40])

print(np.mean(data))
```

Output:

```text
25.0
```

---

# 🐼 7. Pandas

Important topics:

- Series
- DataFrame
- Reading CSV files
- Selecting rows and columns
- Filtering
- Missing values
- Sorting
- GroupBy
- Merge
- Join
- Aggregation

Example:

```python
import pandas as pd

df = pd.read_csv("data.csv")

print(df.head())
print(df.info())
print(df.describe())
```

---

# 🧹 8. Data Cleaning

Practice:

- Missing values
- Duplicate data
- Incorrect data types
- Outliers
- String cleaning
- Date conversion
- Encoding categories

Typical workflow:

```text
Raw Data
   ↓
Inspect Data
   ↓
Remove Duplicates
   ↓
Handle Missing Values
   ↓
Fix Data Types
   ↓
Handle Outliers
   ↓
Clean Dataset
```

---

# 📊 9. Data Visualization

Libraries:

```python
import matplotlib.pyplot as plt
```

Important plots:

- Line chart
- Bar chart
- Histogram
- Scatter plot
- Box plot

Before creating a visualization, ask:

1. What question am I trying to answer?
2. Which variables are important?
3. Which graph represents them clearly?

---

# 🧱 10. Object-Oriented Programming

Important concepts:

```text
Class
Object
Attribute
Method
Constructor
Inheritance
Encapsulation
Polymorphism
```

Example:

```python
class Student:
    def __init__(self, name, score):
        self.name = name
        self.score = score

    def display(self):
        print(self.name, self.score)


student1 = Student("Student A", 90)
student1.display()
```

---

# ⚠️ 11. Exception Handling

Example:

```python
try:
    number = int(input("Enter number: "))
    print(10 / number)

except ValueError:
    print("Please enter a valid number.")

except ZeroDivisionError:
    print("Cannot divide by zero.")
```

Important keywords:

```text
try
except
else
finally
raise
```

---

# 📁 12. File Handling

Practice:

```python
with open("data.txt", "r") as file:
    content = file.read()

print(content)
```

Learn how to work with:

```text
.txt
.csv
.json
.xlsx
```

---

# 🧩 13. Functions and Modules

Learn how to divide large programs into smaller reusable pieces.

Instead of:

```text
One huge Python file
```

Prefer:

```text
project/
│
├── main.py
├── data_loader.py
├── preprocessing.py
├── analysis.py
└── visualization.py
```

---

# 🐞 14. Debugging

When code does not work:

```text
Read Error
    ↓
Find Error Line
    ↓
Understand Error Type
    ↓
Check Variable Values
    ↓
Fix Small Part
    ↓
Run Again
```

Important error types:

```text
SyntaxError
TypeError
ValueError
IndexError
KeyError
NameError
FileNotFoundError
```

---

# 🧪 15. Testing

Simple example:

```python
def add(a, b):
    return a + b


assert add(2, 3) == 5
assert add(10, 5) == 15
```

Testing helps make sure our program behaves correctly.

---

# ⚡ 16. Code Efficiency

Compare different solutions based on:

- Execution time
- Memory
- Code readability
- Number of operations

Study concepts such as:

```text
Time Complexity
Space Complexity
Big O Notation
```

Common examples:

```text
O(1)
O(log n)
O(n)
O(n log n)
O(n²)
```

---

# 📊 17. Data Science Programming Workflow

A common Data Science workflow:

```text
Problem
   ↓
Collect Data
   ↓
Load Data
   ↓
Explore Data
   ↓
Clean Data
   ↓
Transform Data
   ↓
Analyze Data
   ↓
Visualize
   ↓
Model
   ↓
Evaluate
   ↓
Communicate Results
```

---

# 🧠 Study Method

For every new topic, I should follow:

```text
1. Learn
2. Understand
3. Write Code
4. Run Code
5. Change Code
6. Break Code
7. Fix Code
8. Explain Code
9. Practice Again
```

The most important step is:

> If I can explain the code without looking at my notes, I probably understand it.

---

# 📅 Weekly Study Routine

| Day | Activity |
|---|---|
| Monday | Review lecture |
| Tuesday | Practice examples |
| Wednesday | Work on TP |
| Thursday | Homework |
| Friday | Assignment / project |
| Saturday | Review weak topics |
| Sunday | Review the whole week |

---

# ✅ Weekly Checklist

At the end of every week:

- [ ] Lecture reviewed
- [ ] Notes completed
- [ ] Code examples tested
- [ ] TP completed
- [ ] Homework completed
- [ ] Assignment checked
- [ ] Errors documented
- [ ] Important concepts memorized
- [ ] Practice questions completed
- [ ] Difficult topics reviewed

---

# 📝 Topic Study Template

For every important concept:

## Topic

### 1. What is it?

Write a simple definition.

### 2. Why do we use it?

Explain the purpose.

### 3. When do we use it?

Give a real programming situation.

### 4. Syntax

```python
# syntax here
```

### 5. Example

```python
# example here
```

### 6. Output

```text
output here
```

### 7. Important Notes

Write important things to remember.

### 8. Common Mistakes

Write mistakes that I made.

### 9. Practice

Create at least 2–3 small exercises.

---

# 📚 Exam Preparation

Before the exam, review:

### Theory

- Definitions
- Concepts
- Differences between concepts
- Advantages / disadvantages
- When to use each method

### Programming

Practice writing code without copying:

```text
Loops
Functions
List / Dictionary
NumPy
Pandas
OOP
File Handling
Exception Handling
Data Cleaning
Algorithms
```

### Code Reading

Practice answering:

```text
What does this code do?

What is the output?

Why does this error happen?

How can we improve this code?
```

---

# 🎯 Assignment Tracker

| Assignment | Topic | Deadline | Status | Score |
|---|---|---|---|---|
| Assignment 01 | TBD | TBD | ⬜ | - |
| Assignment 02 | TBD | TBD | ⬜ | - |
| Assignment 03 | TBD | TBD | ⬜ | - |

Status:

```text
⬜ Not Started
🟡 In Progress
✅ Completed
```

---

# 🧪 TP Tracker

| TP | Topic | Status | Review |
|---|---|---|---|
| TP01 | TBD | ⬜ | ⬜ |
| TP02 | TBD | ⬜ | ⬜ |
| TP03 | TBD | ⬜ | ⬜ |
| TP04 | TBD | ⬜ | ⬜ |

---

# 🏠 Homework Tracker

| Homework | Topic | Status |
|---|---|---|
| HW01 | TBD | ⬜ |
| HW02 | TBD | ⬜ |
| HW03 | TBD | ⬜ |

---

# 📈 My Progress

```text
Python Fundamentals      [          ]
Advanced Python          [          ]
NumPy                    [          ]
Pandas                   [          ]
OOP                      [          ]
Data Cleaning            [          ]
Visualization            [          ]
Algorithms               [          ]
Testing                  [          ]
Data Science Projects    [          ]
```

Example:

```text
Python Fundamentals      [██████████] 100%
NumPy                    [██████░░░░] 60%
Pandas                   [████░░░░░░] 40%
```

---

# 🚀 Final Goal

By the end of **Advanced Programming for Data Science**, I want to be able to:

> Take a Data Science problem, organize the project, write clean Python code, process and analyze the data, debug problems, and produce a complete working solution independently.# Advanced-Programming-for-DS
