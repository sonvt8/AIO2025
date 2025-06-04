import math

def factorial(n):
    """
    Tính giai thừa của n
    """
    if n == 0:
        return 1
    return n * factorial(n - 1)

def my_sin(x, n):
    """
    sin(x) = x - x³/3! + x⁵/5! - x⁷/7! + x⁹/9! - ...
    """
    result = 0
    for i in range(n):
        # Số hạng thứ i: (-1)^i * x^(2i+1) / (2i+1)!
        power = 2 * i + 1
        term = ((-1) ** i) * (x ** power) / factorial(power)
        result += term
    return result

def my_cos(x, n):
    """
    cos(x) = 1 - x²/2! + x⁴/4! - x⁶/6! + x⁸/8! - x¹⁰/10! + ...
    """
    result = 0
    for i in range(n):
        # Số hạng thứ i: (-1)^i * x^(2i) / (2i)!
        power = 2 * i
        term = ((-1) ** i) * (x ** power) / factorial(power)
        result += term
    return result

def my_sinh(x, n):
    """
    sinh(x) = x + x³/3! + x⁵/5! + x⁷/7! + x⁹/9! + ...
    """
    result = 0
    for i in range(n):
        # Số hạng thứ i: x^(2i+1) / (2i+1)!
        power = 2 * i + 1
        term = (x ** power) / factorial(power)
        result += term
    return result

def my_cosh(x, n):
    """
    cosh(x) = 1 + x²/2! + x⁴/4! + x⁶/6! + x⁸/8! + x¹⁰/10! + ...
    """
    result = 0
    for i in range(n):
        # Số hạng thứ i: x^(2i) / (2i)!
        power = 2 * i
        term = (x ** power) / factorial(power)
        result += term
    return result

def calculate_functions():
    # Input x (radian)
    while True:
        try:
            x = float(input("Nhập giá trị x (radian): "))
            break
        except ValueError:
            print("Vui lòng nhập một số thực hợp lệ")
    
    # Input n (số lần lặp)
    while True:
        try:
            n = int(input("Nhập số lần lặp n (số nguyên dương): "))
            if n > 0:
                break
            else:
                print("n phải là số nguyên dương")
        except ValueError:
            print("Vui lòng nhập một số nguyên hợp lệ")
    
    # Tính toán các hàm
    sin_result = my_sin(x, n)
    cos_result = my_cos(x, n)
    sinh_result = my_sinh(x, n)
    cosh_result = my_cosh(x, n)
    
    # Hiển thị kết quả
    print(f"\n=== Kết quả với x = {x}, n = {n} ===")
    print(f"sin({x}) ≈ {sin_result:.10f}")
    print(f"cos({x}) ≈ {cos_result:.10f}")
    print(f"sinh({x}) ≈ {sinh_result:.10f}")
    print(f"cosh({x}) ≈ {cosh_result:.10f}")
    
    # So sánh với hàm built-in của Python
    print(f"\n=== So sánh với hàm built-in ===")
    print(f"sin({x}) (built-in) = {math.sin(x):.10f}")
    print(f"cos({x}) (built-in) = {math.cos(x):.10f}")
    print(f"sinh({x}) (built-in) = {math.sinh(x):.10f}")
    print(f"cosh({x}) (built-in) = {math.cosh(x):.10f}")
    
    # Tính sai số
    print(f"\n=== Sai số ===")
    print(f"Sai số sin: {abs(sin_result - math.sin(x)):.10f}")
    print(f"Sai số cos: {abs(cos_result - math.cos(x)):.10f}")
    print(f"Sai số sinh: {abs(sinh_result - math.sinh(x)):.10f}")
    print(f"Sai số cosh: {abs(cosh_result - math.cosh(x)):.10f}")

# Chạy chương trình
if __name__ == "__main__":
    calculate_functions()