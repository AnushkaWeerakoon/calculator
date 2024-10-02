import streamlit as st
import sympy as sp
import math
import numpy as np
import matplotlib.pyplot as plt

# Function to display large results
def display_large_result(result):
    st.markdown(f"<h2 style='font-size:36px;'>{result}</h2>", unsafe_allow_html=True)

# Function to display chart
def display_chart(x, y, title):
    plt.figure(figsize=(10, 5))
    plt.plot(x, y, marker='o')
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    st.pyplot(plt)

# Sidebar for navigation
page = st.sidebar.selectbox("Select Page", ["Introduction", "Calculator"])

# Introduction Page
if page == "Introduction":
    st.title("Welcome to the Scientific Calculator App")
    st.write("""
        This application is designed to help you solve a wide range of mathematical problems, including:
        
        - **Basic Arithmetic**: Perform fundamental operations such as addition, subtraction, multiplication, and division.
        - **Algebraic Equations**: Solve algebraic equations with one variable and view the plots of the equations.
        - **Calculus**: Compute derivatives and integrals of mathematical functions, and visualize them with plots.
        - **Trigonometry**: Calculate trigonometric functions for given angles and view their graphical representations.
        - **Logarithmic & Exponential Functions**: Evaluate logarithms and exponentials, and observe their plots.
        - **Matrix Operations**: Perform matrix multiplication and view the results as heatmaps.

        Use the sidebar to navigate to the Calculator page where you can perform various calculations and see their visual representations.
    """)


# Calculator Page
elif page == "Calculator":
    st.title("Calculator Page")
    operation = st.selectbox(
        "Select the type of operation",
        ["Basic Arithmetic", "Algebraic Equation", "Calculus", "Trigonometry", "Logarithmic & Exponential", "Matrix Operations"]
    )

    # Layout containers
    input_container = st.container()
    result_container = st.container()

    # Basic Arithmetic Operations
    if operation == "Basic Arithmetic":
        with input_container:
            st.header("Basic Arithmetic Calculator")
            num1 = st.number_input("Enter first number", format="%.2f")
            num2 = st.number_input("Enter second number", format="%.2f")
            arithmetic_op = st.selectbox("Select operation", ["Addition", "Subtraction", "Multiplication", "Division", "Power"])

        with result_container:
            st.subheader("Result")
            if arithmetic_op == "Addition":
                result = num1 + num2
            elif arithmetic_op == "Subtraction":
                result = num1 - num2
            elif arithmetic_op == "Multiplication":
                result = num1 * num2
            elif arithmetic_op == "Division":
                result = num1 / num2 if num2 != 0 else "Undefined (cannot divide by zero)"
            elif arithmetic_op == "Power":
                result = num1 ** num2

            display_large_result(result)

    # Algebraic Equations Solver
    elif operation == "Algebraic Equation":
        with input_container:
            st.header("Algebraic Equation Solver")
            equation = st.text_input("Enter the equation (e.g., x**2 + 2*x + 1 = 0)")

        with result_container:
            st.subheader("Solution")
            if equation:
                try:
                    lhs, rhs = equation.split('=')
                    lhs_expr = sp.sympify(lhs.strip())
                    rhs_expr = sp.sympify(rhs.strip())
                    
                    eq = sp.Eq(lhs_expr, rhs_expr)
                    x = sp.Symbol('x')
                    solution = sp.solve(eq, x)
                    display_large_result(f"Solutions: {solution}")

                    # Plot the equation
                    x_vals = np.linspace(-10, 10, 400)
                    y_vals = [float(lhs_expr.subs(x, val)) for val in x_vals]
                    y_vals_rhs = [float(rhs_expr.subs(x, val)) for val in x_vals]
                    plt.figure(figsize=(10, 5))
                    plt.plot(x_vals, y_vals, label='LHS')
                    plt.plot(x_vals, y_vals_rhs, label='RHS', linestyle='--')
                    plt.title('Equation Plot')
                    plt.xlabel('x')
                    plt.ylabel('y')
                    plt.legend()
                    plt.grid(True)
                    st.pyplot(plt)

                except Exception as e:
                    st.error(f"Error: {e}")

    # Calculus (Differentiation, Integration)
    elif operation == "Calculus":
        with input_container:
            st.header("Calculus Calculator")
            calc_type = st.selectbox("Select Calculus Operation", ["Differentiation", "Integration"])
            expr = st.text_input("Enter the function (e.g., x**2 + 3*x)")

        with result_container:
            st.subheader("Result")
            if expr:
                x = sp.Symbol('x')
                function = sp.sympify(expr)
                if calc_type == "Differentiation":
                    result = sp.diff(function, x)
                    display_large_result(f"Derivative: {result}")
                elif calc_type == "Integration":
                    result = sp.integrate(function, x)
                    display_large_result(f"Integral: {result}")

                # Plot the function
                x_vals = np.linspace(-10, 10, 400)
                y_vals = [float(function.subs(x, val)) for val in x_vals]
                plt.figure(figsize=(10, 5))
                plt.plot(x_vals, y_vals, label='Function')
                plt.title(f'{calc_type} Plot')
                plt.xlabel('x')
                plt.ylabel('y')
                plt.legend()
                plt.grid(True)
                st.pyplot(plt)

    # Trigonometry
    elif operation == "Trigonometry":
        with input_container:
            st.header("Trigonometry Calculator")
            angle = st.number_input("Enter the angle (in degrees)", format="%.2f")
            trig_func = st.selectbox("Select trigonometric function", ["sin", "cos", "tan", "cot", "sec", "csc"])

        with result_container:
            st.subheader("Result")
            angle_rad = math.radians(angle)  # Convert to radians
            if trig_func == "sin":
                result = math.sin(angle_rad)
            elif trig_func == "cos":
                result = math.cos(angle_rad)
            elif trig_func == "tan":
                result = math.tan(angle_rad)
            elif trig_func == "cot":
                result = 1 / math.tan(angle_rad)
            elif trig_func == "sec":
                result = 1 / math.cos(angle_rad)
            elif trig_func == "csc":
                result = 1 / math.sin(angle_rad)

            display_large_result(f"{trig_func}({angle}) = {result}")

            # Plot the trigonometric function
            x_vals = np.linspace(-360, 360, 400)
            if trig_func == "sin":
                y_vals = np.sin(np.radians(x_vals))
            elif trig_func == "cos":
                y_vals = np.cos(np.radians(x_vals))
            elif trig_func == "tan":
                y_vals = np.tan(np.radians(x_vals))
            elif trig_func == "cot":
                y_vals = 1 / np.tan(np.radians(x_vals))
            elif trig_func == "sec":
                y_vals = 1 / np.cos(np.radians(x_vals))
            elif trig_func == "csc":
                y_vals = 1 / np.sin(np.radians(x_vals))


    # Logarithmic & Exponential Functions
    elif operation == "Logarithmic & Exponential":
        with input_container:
            st.header("Logarithmic & Exponential Calculator")
            number = st.number_input("Enter the number", format="%.2f")
            log_exp_func = st.selectbox("Select function", ["log", "log10", "exp"])

        with result_container:
            st.subheader("Result")
            x_vals = np.linspace(0.1, 10, 400)  # Avoid zero to prevent log(0)
            if log_exp_func == "log":
                result = np.log(number)
            elif log_exp_func == "log10":
                result = np.log10(number)
            elif log_exp_func == "exp":
                result = np.exp(number)

            display_large_result(result)

    # Matrix Operations
    elif operation == "Matrix Operations":
        with input_container:
            st.header("Matrix Operations Calculator")
            rows_a = st.number_input("Enter number of rows for Matrix A", min_value=1, max_value=10, value=2)
            cols_a = st.number_input("Enter number of columns for Matrix A", min_value=1, max_value=10, value=2)
            st.write("Enter values for Matrix A")
            matrix_a = []
            for i in range(rows_a):
                row = st.text_input(f"Row {i+1} (comma separated)", "0,0", key=f"matrix_a_row_{i}")
                matrix_a.append(list(map(float, row.split(","))))

            rows_b = st.number_input("Enter number of rows for Matrix B", min_value=1, max_value=10, value=2)
            cols_b = st.number_input("Enter number of columns for Matrix B", min_value=1, max_value=10, value=2)
            st.write("Enter values for Matrix B")
            matrix_b = []
            for i in range(rows_b):
                row = st.text_input(f"Row {i+1} (comma separated)", "0,0", key=f"matrix_b_row_{i}")
                matrix_b.append(list(map(float, row.split(","))))

        with result_container:
            st.subheader("Result")
            matrix_a_np = np.array(matrix_a)
            matrix_b_np = np.array(matrix_b)

            if matrix_a_np.shape[1] == matrix_b_np.shape[0]:
                result = np.dot(matrix_a_np, matrix_b_np)
                st.write(f"Matrix A * Matrix B = \n{result}")
                display_large_result(f"Matrix A * Matrix B = \n{result}")

                # Plotting the result as a heatmap
                plt.figure(figsize=(10, 5))
                plt.imshow(result, cmap='viridis', interpolation='nearest')
                plt.title('Matrix Result Heatmap')
                plt.colorbar()
                st.pyplot(plt)
            else:
                st.error("Matrix multiplication is not possible with these dimensions.")
