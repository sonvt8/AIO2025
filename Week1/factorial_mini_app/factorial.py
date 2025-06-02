def factorial(n):
    """
    Tính giai thừa của số nguyên không âm n bằng đệ quy.
    Trả về n! = n * (n-1) * (n-2) * ... * 1.
    Ném ngoại lệ nếu n không hợp lệ.
    """
    if not isinstance(n, int):
        raise ValueError("Input phải là số nguyên")
    if n < 0:
        raise ValueError("Input phải là số nguyên không âm")
    if n == 0:
        return 1
    return n * factorial(n - 1)