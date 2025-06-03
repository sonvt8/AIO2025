def evaluate_f1_components(tp, fp, fn):
    # Kiểm tra kiểu dữ liệu: phải là int
    if not (isinstance(tp, int) and isinstance(fp, int) and isinstance(fn, int)):
        raise ValueError("Các giá trị TP, FP, FN phải là số nguyên (integer). Vui lòng nhập lại!")

    # Kiểm tra giá trị: phải >= 0
    if tp < 0 or fp < 0 or fn < 0:
        raise ValueError("Các giá trị TP, FP, FN không được nhỏ hơn 0. Vui lòng nhập lại!")

    # Kiểm tra trường hợp chia cho 0
    errors = []
    if tp + fp == 0:
        errors.append("Tổng TP + FP = 0 (Precision không xác định).")
    if tp + fn == 0:
        errors.append("Tổng TP + FN = 0 (Recall không xác định).")
    if errors:
        error_msg = "Không thể tính toán vì:\n" + "\n".join([f"- {error}" for error in errors]) + "\nVui lòng nhập lại!"
        raise ValueError(error_msg)

    # Tính Precision, Recall, F1-score
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if precision + recall == 0:
        f1_score = 0.0  # Trường hợp đặc biệt khi cả Precision và Recall đều là 0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1_score

def get_input(prompt):
    while True:
        try:
            value = input(prompt)
            return int(value)  # Chuyển đổi thành số nguyên
        except ValueError:
            print("Lỗi: Giá trị nhập vào phải là số nguyên. Vui lòng nhập lại!")

def main():
    print("Chương trình tính Precision, Recall và F1-score")
    print("Nhập các giá trị TP (True Positive), FP (False Positive), FN (False Negative).")
    print("Nhấn Ctrl+C để thoát.\n")

    while True:
        try:
            # Nhập dữ liệu từ người dùng
            tp = get_input("Nhập True Positive (TP): ")
            fp = get_input("Nhập False Positive (FP): ")
            fn = get_input("Nhập False Negative (FN): ")

            # Tính toán
            precision, recall, f1_score = evaluate_f1_components(tp, fp, fn)

            # Hiển thị kết quả
            print("\nKết quả:")
            print(f"Precision: {precision:.2f}")
            print(f"Recall: {recall:.2f}")
            print(f"F1-score: {f1_score:.2f}")

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