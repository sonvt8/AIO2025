import math

# Hàm kiểm tra đầu vào có phải là số không
def is_number(n):
    try:
        float(n)  # Thử chuyển đổi thành float
        return True
    except ValueError:
        return False

# Hàm tính Sigmoid
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# Hàm tính ReLU
def relu(x):
    return max(0, x)  # ReLU: 0 nếu x <= 0, x nếu x > 0

# Hàm tính ELU (alpha = 0.01)
def elu(x, alpha=0.01):
    if x > 0:
        return x
    else:
        return alpha * (math.exp(x) - 1)  # ELU: alpha * (e^x - 1) nếu x <= 0

# Hàm chính để tương tác với người dùng
def main():
    print("Chương trình tính giá trị của các hàm kích hoạt")
    print("Hàm hỗ trợ: sigmoid, relu, elu")
    print("Nhấn Ctrl+C để thoát.\n")

    while True:
        try:
            # Nhập giá trị x
            x_input = input("Nhập giá trị x: ")
            if not is_number(x_input):
                raise ValueError("x phải là một số. Vui lòng nhập lại!")
            x = float(x_input)  # Chuyển đổi thành float

            # Nhập tên hàm kích hoạt
            activation = input("Nhập tên hàm kích hoạt (sigmoid/relu/elu): ").strip().lower()
            if activation not in ['sigmoid', 'relu', 'elu']:
                raise ValueError(f"{activation} không được hỗ trợ! Chỉ hỗ trợ sigmoid, relu, elu.")

            # Tính toán dựa trên hàm kích hoạt
            if activation == 'sigmoid':
                value = sigmoid(x)
            elif activation == 'relu':
                value = relu(x)
            else:  # elu
                value = elu(x, alpha=0.01)

            # Hiển thị kết quả
            print("\nKết quả:")
            print(f"Hàm {activation}({x}) = {value:.4f}")

            # Hỏi người dùng có muốn tiếp tục không
            choice = input("\nBạn có muốn tiếp tục tính toán? (y/n): ").strip().lower()
            if choice != 'y':
                print("Kết thúc chương trình.")
                break
            print()

        except ValueError as e:
            print(f"\nLỗi: {str(e)}\n")
        except KeyboardInterrupt:
            print("\n\nĐã thoát chương trình.")
            break
        except Exception as e:
            print(f"\nLỗi không xác định: {str(e)}\n")

if __name__ == "__main__":
    main()